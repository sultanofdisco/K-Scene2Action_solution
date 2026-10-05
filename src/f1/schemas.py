"""Validated wire formats for F1, without model execution or a gate."""

from enum import StrEnum
from hashlib import sha256
from math import isclose
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


# Do not strip or normalize strings: hashes refer to the original command.
Text = Annotated[str, StringConstraints(strict=True, min_length=1, pattern=r"\S")]
Sha256 = Annotated[str, StringConstraints(strict=True, pattern=r"^[0-9a-f]{64}$")]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]
Probability = Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]
Coordinate = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]


class Schema(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class Decision(StrEnum):
    PASS = "PASS"
    BLOCK = "BLOCK"
    DEFER = "DEFER"


class ContextStatus(StrEnum):
    READY = "READY"
    REJECTED = "REJECTED"
    INCOMPLETE = "INCOMPLETE"


class OStatus(StrEnum):
    CLEAR = "CLEAR"
    ATTACK = "ATTACK"
    UNKNOWN = "UNKNOWN"
    ERROR = "ERROR"


class OCRStatus(StrEnum):
    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"
    ERROR = "ERROR"


class Answer(StrEnum):
    YES = "YES"
    NO = "NO"
    UNKNOWN = "UNKNOWN"


class ReasonCode(StrEnum):
    INPUT_CLEAR = "INPUT_CLEAR"
    NO_TEXT_DETECTED = "NO_TEXT_DETECTED"
    INVALID_REQUEST = "INVALID_REQUEST"
    IMAGE_UNAVAILABLE = "IMAGE_UNAVAILABLE"
    IMAGE_DECODE_ERROR = "IMAGE_DECODE_ERROR"
    IMAGE_MISMATCH = "IMAGE_MISMATCH"
    INPUT_BINDING_MISMATCH = "INPUT_BINDING_MISMATCH"
    UNSUPPORTED_MODE = "UNSUPPORTED_MODE"
    CONTEXT_MISSING = "CONTEXT_MISSING"
    CONTEXT_INCOMPLETE = "CONTEXT_INCOMPLETE"
    CONTEXT_MISMATCH = "CONTEXT_MISMATCH"
    CONTEXT_REJECTED = "CONTEXT_REJECTED"
    F1T_HARMFUL_REQUEST = "F1T_HARMFUL_REQUEST"
    F1T_TEXT_TOO_LONG = "F1T_TEXT_TOO_LONG"
    F1T_UNCERTAIN = "F1T_UNCERTAIN"
    F1T_MODEL_NOT_READY = "F1T_MODEL_NOT_READY"
    F1T_ERROR = "F1T_ERROR"
    OCR_UNREADABLE = "OCR_UNREADABLE"
    OCR_BUDGET_EXCEEDED = "OCR_BUDGET_EXCEEDED"
    OCR_ERROR = "OCR_ERROR"
    F1O_HARMFUL_INSTRUCTION = "F1O_HARMFUL_INSTRUCTION"
    F1O_TASK_OVERRIDE = "F1O_TASK_OVERRIDE"
    F1O_AUTHORITY_SPOOFING = "F1O_AUTHORITY_SPOOFING"
    F1O_UNCERTAIN = "F1O_UNCERTAIN"
    F1O_INPUT_TOO_LONG = "F1O_INPUT_TOO_LONG"
    F1O_INVALID_OUTPUT = "F1O_INVALID_OUTPUT"
    F1O_ERROR = "F1O_ERROR"
    REQUIRED_CHECK_MISSING = "REQUIRED_CHECK_MISSING"


PASS_REASONS = frozenset({ReasonCode.INPUT_CLEAR, ReasonCode.NO_TEXT_DETECTED})
BLOCK_REASONS = frozenset({
    ReasonCode.CONTEXT_REJECTED, ReasonCode.F1T_HARMFUL_REQUEST,
    ReasonCode.F1O_HARMFUL_INSTRUCTION, ReasonCode.F1O_TASK_OVERRIDE,
    ReasonCode.F1O_AUTHORITY_SPOOFING,
})
ATTACK_REASONS = BLOCK_REASONS - {
    ReasonCode.CONTEXT_REJECTED, ReasonCode.F1T_HARMFUL_REQUEST,
}
DEFER_REASONS = frozenset(ReasonCode) - PASS_REASONS - BLOCK_REASONS
Reasons = Annotated[tuple[ReasonCode, ...], Field(min_length=1)]


class ImageRef(Schema):
    """Opaque asset reference; SHA-256 covers encoded file bytes, not pixels."""

    asset_id: Text
    sha256: Sha256
    width: PositiveInt
    height: PositiveInt


class Request(Schema):
    schema_version: Literal["1.0"] = "1.0"
    request_id: Text
    command: Text
    image: ImageRef
    environment_version: Text
    mode: Literal["C"] = "C"


class Target(Schema):
    description: Text
    entity_id: Text | None = None


class TaskContext(Schema):
    schema_version: Literal["1.0"] = "1.0"
    context_id: Text
    command_sha256: Sha256
    environment_version: Text
    context_status: ContextStatus
    # Rejected/incomplete contexts may be missing semantic fields.
    target: Target | None = None
    explicit_goal: Text | None = None
    constraints: tuple[Text, ...] = ()

    @model_validator(mode="after")
    def ready_has_goal(self) -> Self:
        if self.context_status == ContextStatus.READY:
            if self.target is None or self.explicit_goal is None:
                raise ValueError("READY requires target and explicit_goal")
        return self


class TInput(Schema):
    """Model input for F1-T; TaskContext and Contract are intentionally absent."""

    command: Text
    image: ImageRef


class InputBinding(Schema):
    request_id: Text
    command_sha256: Sha256
    environment_version: Text
    source_image: ImageRef
    inspected_image: ImageRef
    preprocessing_version: Text


class TAssessment(Schema):
    binding: InputBinding
    decision: Decision
    reason_codes: Reasons
    risk_score: Probability | None = None
    model_version: Text
    policy_version: Text

    @model_validator(mode="after")
    def valid_decision(self) -> Self:
        allowed = {
            Decision.PASS: {ReasonCode.INPUT_CLEAR},
            Decision.BLOCK: {ReasonCode.F1T_HARMFUL_REQUEST},
            Decision.DEFER: DEFER_REASONS,
        }[self.decision]
        if not set(self.reason_codes) <= allowed:
            raise ValueError("reason_codes do not match F1-T decision")
        if self.decision != Decision.DEFER and self.risk_score is None:
            raise ValueError("F1-T PASS/BLOCK requires a risk_score")
        return self


class OCRRegion(Schema):
    region_id: Text
    polygon: Annotated[tuple[tuple[Coordinate, Coordinate], ...], Field(min_length=3)]
    # Empty text is preserved for a detected but unreadable region.
    raw_text: Annotated[str, StringConstraints(strict=True)]
    recognition_score: Probability | None
    group_id: Text
    reading_order: Annotated[int, Field(strict=True, ge=0)]


class OCRResult(Schema):
    image: ImageRef
    status: OCRStatus
    regions: tuple[OCRRegion, ...]
    reason_codes: tuple[ReasonCode, ...] = ()
    model_version: Text

    @model_validator(mode="after")
    def validate_regions(self) -> Self:
        ids = [region.region_id for region in self.regions]
        if len(ids) != len(set(ids)):
            raise ValueError("OCR region IDs must be unique")
        for region in self.regions:
            if any(x > self.image.width or y > self.image.height for x, y in region.polygon):
                raise ValueError("OCR polygon exceeds inspected image bounds")
            if self.status == OCRStatus.COMPLETE:
                if not region.raw_text.strip() or region.recognition_score is None:
                    raise ValueError("unreadable regions cannot be COMPLETE")
        if self.status == OCRStatus.COMPLETE and not self.regions:
            if self.reason_codes != (ReasonCode.NO_TEXT_DETECTED,):
                raise ValueError("empty completed OCR requires NO_TEXT_DETECTED")
        elif ReasonCode.NO_TEXT_DETECTED in self.reason_codes:
            raise ValueError("NO_TEXT_DETECTED only applies to completed empty OCR")
        if self.status != OCRStatus.COMPLETE and not self.reason_codes:
            raise ValueError("incomplete/failed OCR requires a reason")
        return self


class ChoiceProbabilities(Schema):
    yes: Probability
    no: Probability
    unknown: Probability

    @model_validator(mode="after")
    def sums_to_one(self) -> Self:
        if not isclose(self.yes + self.no + self.unknown, 1.0, abs_tol=1e-5):
            raise ValueError("choice probabilities must sum to one")
        return self


class QuestionAssessment(Schema):
    model_answer: Answer
    # A future calibrated policy may abstain; do not overwrite the raw answer.
    decision: Answer
    probabilities: ChoiceProbabilities

    @model_validator(mode="after")
    def only_abstention_can_change_answer(self) -> Self:
        if self.decision not in (self.model_answer, Answer.UNKNOWN):
            raise ValueError("policy can abstain but cannot flip the model answer")
        return self


class Questions(Schema):
    # Partial answers are retained on ATTACK/ERROR, never treated as CLEAR.
    harmful_instruction: QuestionAssessment | None = None
    task_override: QuestionAssessment | None = None
    authority_spoofing: QuestionAssessment | None = None
    benign_observation: QuestionAssessment | None = None


QUESTION_REASONS = {
    "harmful_instruction": ReasonCode.F1O_HARMFUL_INSTRUCTION,
    "task_override": ReasonCode.F1O_TASK_OVERRIDE,
    "authority_spoofing": ReasonCode.F1O_AUTHORITY_SPOOFING,
}


class OAssessment(Schema):
    binding: InputBinding
    context_id: Text
    status: OStatus
    analysis_complete: bool = Field(strict=True)
    reason_codes: Reasons
    questions: Questions
    ocr: OCRResult
    model_version: Text
    policy_version: Text

    @model_validator(mode="after")
    def validate_summary(self) -> Self:
        if self.ocr.image != self.binding.inspected_image:
            raise ValueError("OCR image must match inspected image")
        positives = {
            reason for name, reason in QUESTION_REASONS.items()
            if (answer := getattr(self.questions, name)) is not None
            and answer.decision == Answer.YES
        }
        reasons = set(self.reason_codes)
        if positives:
            if self.status != OStatus.ATTACK or not positives <= reasons:
                raise ValueError("confirmed attack must be ATTACK with matching reasons")
        elif self.status == OStatus.ATTACK:
            raise ValueError("ATTACK requires a positive Q1-Q3 decision")
        if reasons & ATTACK_REASONS != positives:
            raise ValueError("attack reasons must match positive questions")
        answers = tuple(getattr(self.questions, name) for name in Questions.model_fields)
        no_text = self.ocr.status == OCRStatus.COMPLETE and not self.ocr.regions
        if no_text and any(a is not None for a in answers):
            raise ValueError("completed empty OCR must not have Laya answers")
        if self.analysis_complete and not no_text:
            if self.ocr.status != OCRStatus.COMPLETE or any(a is None for a in answers):
                raise ValueError("complete analysis requires completed OCR and all questions")
        if self.status == OStatus.CLEAR:
            if not self.analysis_complete or self.ocr.status != OCRStatus.COMPLETE:
                raise ValueError("CLEAR requires completed analysis and OCR")
            if no_text:
                if any(a is not None for a in answers) or reasons != {ReasonCode.NO_TEXT_DETECTED}:
                    raise ValueError("no-text CLEAR skips Laya and records NO_TEXT_DETECTED")
            else:
                decisions = tuple(a.decision if a is not None else None for a in answers)
                if decisions != (Answer.NO, Answer.NO, Answer.NO, Answer.YES):
                    raise ValueError("CLEAR requires NO/NO/NO/YES")
                if reasons != {ReasonCode.INPUT_CLEAR}:
                    raise ValueError("text CLEAR requires INPUT_CLEAR")
        elif reasons & PASS_REASONS:
            raise ValueError("non-CLEAR cannot contain success reasons")
        if self.status == OStatus.ERROR and self.analysis_complete:
            raise ValueError("ERROR cannot have completed analysis")
        return self


class ApprovedInput(Schema):
    command: Text
    command_sha256: Sha256
    source_image_sha256: Sha256
    image: ImageRef
    environment_version: Text
    context_id: Text

    @model_validator(mode="after")
    def command_matches_hash(self) -> Self:
        if sha256(self.command.encode("utf-8")).hexdigest() != self.command_sha256:
            raise ValueError("approved command does not match command_sha256")
        return self


class F1Result(Schema):
    schema_version: Literal["1.0"] = "1.0"
    # None is only for malformed requests where an ID could not be recovered.
    request_id: Text | None
    decision: Decision
    reason_codes: Reasons
    approved_input: ApprovedInput | None
    trace_id: Text

    @model_validator(mode="after")
    def approved_only_on_pass(self) -> Self:
        reasons = set(self.reason_codes)
        if len(reasons) != len(self.reason_codes):
            raise ValueError("duplicate reason codes")
        if self.decision == Decision.PASS:
            if self.approved_input is None or self.request_id is None:
                raise ValueError("PASS requires request_id and approved_input")
            if not reasons <= PASS_REASONS:
                raise ValueError("PASS cannot carry block/defer reasons")
        else:
            if self.approved_input is not None:
                raise ValueError("BLOCK/DEFER must set approved_input to null")
            if reasons & PASS_REASONS:
                raise ValueError("BLOCK/DEFER cannot carry success reasons")
            if self.decision == Decision.BLOCK and not reasons & BLOCK_REASONS:
                raise ValueError("BLOCK requires a confirmed block reason")
            if self.decision == Decision.DEFER and not reasons <= DEFER_REASONS:
                raise ValueError("DEFER cannot hide a confirmed block reason")
        if self.request_id is None:
            if self.decision != Decision.DEFER or ReasonCode.INVALID_REQUEST not in reasons:
                raise ValueError("missing request_id requires DEFER/INVALID_REQUEST")
        return self


PUBLIC_SCHEMAS = (Request, TaskContext, TInput, InputBinding, TAssessment, OCRResult,
                  OAssessment, ApprovedInput, F1Result)
