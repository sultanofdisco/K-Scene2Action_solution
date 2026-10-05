import json
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from f1.__main__ import export_schemas, main
from f1.bindings import (
    BindingError, bind_input, command_sha256, require_ready_context,
    validate_assessment_binding, validate_result_binding, verify_image_bytes,
)
from f1.schemas import (
    ChoiceProbabilities, F1Result, OCRResult, OAssessment, PUBLIC_SCHEMAS,
    Request, TAssessment, TaskContext, TInput,
)


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


def read(name):
    return json.loads((EXAMPLES / name).read_text(encoding="utf-8"))


@pytest.fixture
def request_data():
    return read("request.json")


@pytest.fixture
def context_data():
    return read("task-context.json")


@pytest.mark.parametrize("field,value", [
    ("command", "  \n"), ("command", 123), ("request_id", ""),
    ("mode", "D"), ("schema_version", "2.0"), ("contract", {}),
])
def test_request_rejects_invalid_fields(request_data, field, value):
    request_data[field] = value
    with pytest.raises(ValidationError):
        Request.model_validate(request_data)


@pytest.mark.parametrize("field,value", [
    ("width", 0), ("height", -1), ("width", True), ("width", "1280"),
    ("sha256", "<hash>"), ("sha256", "A" * 64),
])
def test_bad_image_metadata(request_data, field, value):
    request_data["image"][field] = value
    with pytest.raises(ValidationError):
        Request.model_validate(request_data)


def test_original_command_is_preserved(request_data):
    text = "  원문\r\n그대로  "
    request_data["command"] = text
    request = Request.model_validate(request_data)
    assert request.command == text
    assert command_sha256(text) != command_sha256(text.strip())
    assert command_sha256("가") != command_sha256("\u1100\u1161")
    with pytest.raises(ValidationError):
        request.command = "replacement"


def test_f1t_input_excludes_context(request_data, context_data):
    payload = {key: request_data[key] for key in ("command", "image")}
    TInput.model_validate(payload)
    payload["context"] = context_data
    with pytest.raises(ValidationError):
        TInput.model_validate(payload)


def test_ready_context_requires_semantics(context_data):
    context_data["explicit_goal"] = None
    with pytest.raises(ValidationError):
        TaskContext.model_validate(context_data)
    context_data["context_status"] = "INCOMPLETE"
    TaskContext.model_validate(context_data)


@pytest.mark.parametrize("change,expected", [
    ({"command_sha256": "0" * 64}, "CONTEXT_MISMATCH"),
    ({"environment_version": "lab-v2"}, "CONTEXT_MISMATCH"),
    ({"context_status": "REJECTED"}, "CONTEXT_REJECTED"),
    ({"context_status": "INCOMPLETE"}, "CONTEXT_INCOMPLETE"),
])
def test_context_identity_and_status(request_data, context_data, change, expected):
    context_data.update(change)
    with pytest.raises(BindingError) as error:
        require_ready_context(Request.model_validate(request_data), TaskContext.model_validate(context_data))
    assert error.value.reason_code == expected


def test_missing_context(request_data):
    with pytest.raises(BindingError, match="not available") as error:
        require_ready_context(Request.model_validate(request_data), None)
    assert error.value.reason_code == "CONTEXT_MISSING"


def test_image_bytes_are_bound(request_data):
    request = Request.model_validate(request_data)
    content = (EXAMPLES / "assets/io-only.png").read_bytes()
    verify_image_bytes(request.image, content)
    with pytest.raises(BindingError) as error:
        verify_image_bytes(request.image, content + b"changed")
    assert error.value.reason_code == "IMAGE_MISMATCH"


def test_examples_round_trip(request_data, context_data):
    request = Request.model_validate(request_data)
    context = TaskContext.model_validate(context_data)
    assert require_ready_context(request, context) == context
    for model, filename in ((TAssessment, "t-assessment.mock.json"), (OAssessment, "o-assessment.mock.json")):
        assessment = model.model_validate(read(filename))
        expected = bind_input(request, request.image, "identity-fixture-v1")
        validate_assessment_binding(assessment, expected, context)
        assert model.model_validate_json(assessment.model_dump_json()) == assessment
    for example in read("responses.mock.json"):
        result = F1Result.model_validate(example["result"])
        validate_result_binding(result, request, context, request.image)
        assert F1Result.model_validate_json(result.model_dump_json()) == result


def test_stale_assessment_rejected(request_data, context_data):
    assessment = TAssessment.model_validate(read("t-assessment.mock.json"))
    request_data["command"] += " 다시"
    request = Request.model_validate(request_data)
    with pytest.raises(BindingError) as error:
        validate_assessment_binding(assessment, bind_input(request, request.image, "identity-fixture-v1"))
    assert error.value.reason_code == "INPUT_BINDING_MISMATCH"


@pytest.mark.parametrize("change", [
    {"context_id": "other-context"}, {"context_status": "INCOMPLETE"},
])
def test_f1o_checks_context(request_data, context_data, change):
    request = Request.model_validate(request_data)
    context_data.update(change)
    assessment = OAssessment.model_validate(read("o-assessment.mock.json"))
    with pytest.raises(BindingError):
        validate_assessment_binding(assessment, bind_input(request, request.image, "identity-fixture-v1"),
                                    TaskContext.model_validate(context_data))


@pytest.mark.parametrize("decision,reasons", [
    ("BLOCK", ["F1O_TASK_OVERRIDE"]), ("DEFER", ["OCR_ERROR"]),
])
def test_nonpass_cannot_forward_input(decision, reasons):
    data = read("responses.mock.json")[0]["result"]
    data.update(decision=decision, reason_codes=reasons)
    with pytest.raises(ValidationError, match="approved_input to null"):
        F1Result.model_validate(data)


def test_pass_requires_approved_input():
    data = read("responses.mock.json")[0]["result"]
    data["approved_input"] = None
    with pytest.raises(ValidationError):
        F1Result.model_validate(data)


@pytest.mark.parametrize("field,value", [
    ("reason_codes", ["OCR_ERROR"]), ("reason_codes", ["unknown_reason"]),
    ("reason_codes", []), ("request_id", None),
])
def test_invalid_pass(field, value):
    data = read("responses.mock.json")[0]["result"]
    data[field] = value
    with pytest.raises(ValidationError):
        F1Result.model_validate(data)


@pytest.mark.parametrize("field,value", [
    ("context_id", "stale-context"), ("environment_version", "lab-v2"),
    ("source_image_sha256", "0" * 64),
])
def test_pass_binding_rejects_substitution(request_data, context_data, field, value):
    data = read("responses.mock.json")[0]["result"]
    data["approved_input"][field] = value
    request = Request.model_validate(request_data)
    with pytest.raises(BindingError):
        validate_result_binding(F1Result.model_validate(data), request,
                                TaskContext.model_validate(context_data), request.image)


def test_approved_command_cannot_change_silently():
    data = read("responses.mock.json")[0]["result"]
    data["approved_input"]["command"] += " 수정"
    with pytest.raises(ValidationError, match="command_sha256"):
        F1Result.model_validate(data)


def test_ocr_failure_is_not_no_text():
    data = read("o-assessment.mock.json")["ocr"]
    data.update(status="ERROR", regions=[], reason_codes=["NO_TEXT_DETECTED"])
    with pytest.raises(ValidationError):
        OCRResult.model_validate(data)


def test_empty_ocr_must_be_explicit():
    data = read("o-assessment.mock.json")["ocr"]
    data.update(regions=[], reason_codes=[])
    with pytest.raises(ValidationError):
        OCRResult.model_validate(data)
    data["reason_codes"] = ["NO_TEXT_DETECTED"]
    OCRResult.model_validate(data)


def test_unreadable_region_is_retained():
    data = read("o-assessment.mock.json")["ocr"]
    data["regions"][0]["raw_text"] = ""
    with pytest.raises(ValidationError):
        OCRResult.model_validate(data)
    data.update(status="INCOMPLETE", reason_codes=["OCR_UNREADABLE"])
    assert len(OCRResult.model_validate(data).regions) == 1


def test_ocr_coordinates_and_duplicate_ids():
    data = read("o-assessment.mock.json")["ocr"]
    out_of_bounds = deepcopy(data)
    out_of_bounds["regions"][0]["polygon"][0] = [99, 99]
    with pytest.raises(ValidationError):
        OCRResult.model_validate(out_of_bounds)
    data["regions"].append(deepcopy(data["regions"][0]))
    with pytest.raises(ValidationError):
        OCRResult.model_validate(data)


def clear_o_data():
    data = read("o-assessment.mock.json")
    data.update(status="CLEAR", reason_codes=["INPUT_CLEAR"])
    for name, question in data["questions"].items():
        value = "YES" if name == "benign_observation" else "NO"
        question.update(model_answer=value, decision=value,
                        probabilities={"yes": 1.0 if value == "YES" else 0.0,
                                       "no": 1.0 if value == "NO" else 0.0, "unknown": 0.0})
    return data


def test_clear_requires_all_four_questions():
    data = clear_o_data()
    OAssessment.model_validate(data)
    del data["questions"]["task_override"]
    with pytest.raises(ValidationError):
        OAssessment.model_validate(data)


def test_benign_answer_cannot_cancel_attack():
    data = read("o-assessment.mock.json")
    data["questions"]["benign_observation"].update(model_answer="YES", decision="YES")
    assert OAssessment.model_validate(data).status == "ATTACK"
    data.update(status="CLEAR", reason_codes=["INPUT_CLEAR"])
    with pytest.raises(ValidationError):
        OAssessment.model_validate(data)


def test_confirmed_attack_survives_partial_analysis():
    data = read("o-assessment.mock.json")
    data["questions"]["benign_observation"] = None
    data["analysis_complete"] = False
    assert OAssessment.model_validate(data).status == "ATTACK"
    data["analysis_complete"] = True
    with pytest.raises(ValidationError):
        OAssessment.model_validate(data)


def test_no_text_clear_skips_laya():
    data = clear_o_data()
    data.update(questions={}, reason_codes=["NO_TEXT_DETECTED"])
    data["ocr"].update(regions=[], reason_codes=["NO_TEXT_DETECTED"])
    data["model_version"] = "not-run:no-text"
    OAssessment.model_validate(data)


@pytest.mark.parametrize("scores", [
    {"yes": float("nan"), "no": 0.0, "unknown": 0.0},
    {"yes": float("inf"), "no": 0.0, "unknown": 0.0},
    {"yes": -0.1, "no": 1.0, "unknown": 0.1},
    {"yes": 0.1, "no": 0.1, "unknown": 0.1},
])
def test_bad_probabilities(scores):
    with pytest.raises(ValidationError):
        ChoiceProbabilities.model_validate(scores)


def test_exported_schemas_match_code(tmp_path):
    export_schemas(tmp_path)
    for model in PUBLIC_SCHEMAS:
        name = f"{model.__name__}.schema.json"
        assert json.loads((tmp_path / name).read_text()) == json.loads((ROOT / "schemas" / name).read_text())


def test_cli_does_not_emit_an_approval(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["f1", "validate-input", "--request", str(EXAMPLES / "request.json"),
                                    "--context", str(EXAMPLES / "task-context.json"),
                                    "--image", str(EXAMPLES / "assets/io-only.png")])
    assert main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "VALID_IO_ONLY"
    assert result["inference_executed"] is False
    assert "decision" not in result


def test_cli_invalid_input_does_not_echo_command(tmp_path, monkeypatch, capsys):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps({"command": "private input", "mode": "D"}))
    monkeypatch.setattr("sys.argv", ["f1", "validate-input", "--request", str(path)])
    assert main() == 2
    output = capsys.readouterr().err
    assert "private input" not in output
    assert json.loads(output)["status"] == "INVALID_IO"
