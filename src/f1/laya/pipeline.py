"""Map four experimental model answers to the existing OAssessment contract."""
import copy
import json
import math

from f1.bindings import BindingError, bind_input, require_ready_context
from f1.schemas import (Answer, ChoiceProbabilities, OAssessment, OCRStatus, OStatus,
                        QuestionAssessment, Questions, QUESTION_REASONS, ReasonCode)
from .backend import InputTooLong, InvalidOutput
from .questions import QUESTIONS, PROMPT_VERSION

POLICY_VERSION = "f1o-base-argmax-diagnostic-v1"
LABELS = {"A": Answer.YES, "B": Answer.NO, "C": Answer.UNKNOWN}


def make_state(request, context, ocr):
    groups = {}
    for region in sorted(ocr.regions, key=lambda r: (r.reading_order, r.region_id)):
        group = groups.setdefault(region.group_id, {"group_id": region.group_id, "regions": []})
        group["regions"].append({"region_id": region.region_id, "text": region.raw_text})
    return {
        "trusted_context": {"command": request.command, "target": context.target.description,
                            "explicit_goal": context.explicit_goal, "constraints": list(context.constraints)},
        "untrusted_observation": {"source": "image_ocr", "groups": list(groups.values())},
    }


def parse_answer(raw):
    if (not isinstance(raw, dict) or raw.get("type") != "choice"
            or not isinstance(raw.get("choice"), str) or raw["choice"] not in LABELS):
        raise InvalidOutput("Invalid choice result")
    probs = raw.get("probabilities")
    if not isinstance(probs, dict) or set(probs) != set(LABELS):
        raise InvalidOutput("Three choice probabilities are required")
    if any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in probs.values()):
        raise InvalidOutput("Invalid probability")
    total = sum(probs.values())
    # SDK rounds each probability to four decimals. Only correct that rounding,
    # not arbitrary malformed distributions; keep originals in the raw trace.
    if abs(total - 1) > 0.0002:
        raise InvalidOutput("Probability sum is invalid")
    choice = raw["choice"]
    if probs[choice] != max(probs.values()):
        raise InvalidOutput("Choice disagrees with probabilities")
    answer = LABELS[choice]
    decision = answer if sum(p == probs[choice] for p in probs.values()) == 1 else Answer.UNKNOWN
    return QuestionAssessment(model_answer=answer, decision=decision,
        probabilities=ChoiceProbabilities(yes=probs["A"]/total, no=probs["B"]/total, unknown=probs["C"]/total))


def assess(request, context, binding, ocr, backend, trace=None):
    """Diagnostic answers, not calibrated safety decisions or an F1Result."""
    require_ready_context(request, context)
    if binding != bind_input(request, ocr.image, binding.preprocessing_version):
        raise BindingError(ReasonCode.INPUT_BINDING_MISMATCH, "OCR binding differs from request")
    trace = trace if trace is not None else {}
    trace.update(deployment_ready=False, f1_approval=False, prompt_version=PROMPT_VERSION,
                 policy_version=POLICY_VERSION, question_count=0, laya_executed=False,
                 confidence_policy="UNCALIBRATED_ARGMAX_DIAGNOSTIC_ONLY")
    answers = {}
    issues = []

    def result(status, complete, reasons):
        return OAssessment(binding=binding, context_id=context.context_id, status=status,
                           analysis_complete=complete, reason_codes=tuple(dict.fromkeys(reasons)),
                           questions=Questions(**answers), ocr=ocr,
                           model_version=backend.version, policy_version=POLICY_VERSION)

    if ocr.status == OCRStatus.COMPLETE and not ocr.regions:
        return result(OStatus.CLEAR, True, [ReasonCode.NO_TEXT_DETECTED])
    if ocr.status == OCRStatus.COMPLETE and ocr.reason_codes:
        return result(OStatus.ERROR, False, [ReasonCode.F1O_INVALID_OUTPUT])
    if ocr.status != OCRStatus.COMPLETE:
        issues.extend(r for r in ocr.reason_codes if r in {
            ReasonCode.OCR_ERROR, ReasonCode.OCR_UNREADABLE, ReasonCode.OCR_BUDGET_EXCEEDED})
        issues.append(ReasonCode.OCR_ERROR if ocr.status == OCRStatus.ERROR else ReasonCode.OCR_UNREADABLE)
    if not ocr.regions:
        return result(OStatus.ERROR if ocr.status == OCRStatus.ERROR else OStatus.UNKNOWN, False, issues)

    state = make_state(request, context, ocr)
    trace["state"] = state
    trace["group_count"] = len(state["untrusted_observation"]["groups"])
    failed = False
    try:
        # Bound memory/tokenizer work before the exact token check in the backend.
        if len(ocr.regions) > 256 or sum(len(r.raw_text) for r in ocr.regions) > 8192 or len(json.dumps(state, ensure_ascii=False)) > 32768:
            raise InputTooLong("Input exceeds prototype processing budget")
        raw = backend.predict(state, QUESTIONS)
        trace["laya_executed"] = True
        trace["question_count"] = len(QUESTIONS)
        trace["raw_result"] = raw
        if not isinstance(raw, dict) or not isinstance(raw.get("answers"), dict):
            raise InvalidOutput("Missing answer mapping")
        for name in QUESTIONS:
            try:
                answers[name] = parse_answer(raw["answers"].get(name))
            except InvalidOutput:
                failed = True
                issues.append(ReasonCode.F1O_INVALID_OUTPUT)
        if set(raw["answers"]) - set(QUESTIONS):
            failed = True
            issues.append(ReasonCode.F1O_INVALID_OUTPUT)
    except InputTooLong:
        issues.append(ReasonCode.F1O_INPUT_TOO_LONG)
    except InvalidOutput:
        failed = True
        issues.append(ReasonCode.F1O_INVALID_OUTPUT)
    except Exception as exc:
        failed = True
        issues.append(ReasonCode.F1O_ERROR)
        trace["error_type"] = type(exc).__name__
    finally:
        trace["backend"] = copy.deepcopy(backend.metadata)

    positives = [reason for name, reason in QUESTION_REASONS.items()
                 if name in answers and answers[name].decision == Answer.YES]
    complete = ocr.status == OCRStatus.COMPLETE and len(answers) == 4 and not issues
    if positives:
        return result(OStatus.ATTACK, complete, positives + issues)
    if failed or ocr.status == OCRStatus.ERROR:
        return result(OStatus.ERROR, False, issues)
    decisions = tuple(answers[n].decision if n in answers else None for n in QUESTIONS)
    if complete and decisions == (Answer.NO, Answer.NO, Answer.NO, Answer.YES):
        return result(OStatus.CLEAR, True, [ReasonCode.INPUT_CLEAR])
    return result(OStatus.UNKNOWN, complete, issues + [ReasonCode.F1O_UNCERTAIN])
