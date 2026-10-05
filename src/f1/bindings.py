"""Input identity checks, separate from model decisions and the future gate."""

from hashlib import sha256

from .schemas import (
    ContextStatus, Decision, F1Result, ImageRef, InputBinding,
    OAssessment, ReasonCode, Request, TAssessment, TaskContext,
)


class BindingError(ValueError):
    def __init__(self, reason_code: ReasonCode, message: str):
        super().__init__(message)
        self.reason_code = reason_code


def command_sha256(command: str) -> str:
    """Hash exact UTF-8 text without BOM, trimming, or Unicode normalization."""
    return sha256(command.encode("utf-8")).hexdigest()


def verify_image_bytes(image: ImageRef, content: bytes) -> None:
    """Verify encoded bytes. Image decoding/dimension checks belong to ingestion."""
    if sha256(content).hexdigest() != image.sha256:
        raise BindingError(ReasonCode.IMAGE_MISMATCH, "image bytes differ from the reference")


def bind_input(request: Request, inspected_image: ImageRef,
               preprocessing_version: str) -> InputBinding:
    return InputBinding(
        request_id=request.request_id,
        command_sha256=command_sha256(request.command),
        environment_version=request.environment_version,
        source_image=request.image,
        inspected_image=inspected_image,
        preprocessing_version=preprocessing_version,
    )


def require_ready_context(request: Request, context: TaskContext | None) -> TaskContext:
    """Check context binding and preparation status, not its semantic accuracy."""
    if context is None:
        raise BindingError(ReasonCode.CONTEXT_MISSING, "task context is not available")
    if (context.command_sha256 != command_sha256(request.command)
            or context.environment_version != request.environment_version):
        raise BindingError(ReasonCode.CONTEXT_MISMATCH, "context belongs to a different input")
    if context.context_status == ContextStatus.REJECTED:
        raise BindingError(ReasonCode.CONTEXT_REJECTED, "context provider rejected the task")
    if context.context_status == ContextStatus.INCOMPLETE:
        raise BindingError(ReasonCode.CONTEXT_INCOMPLETE, "context is not ready")
    return context


def validate_assessment_binding(
    assessment: TAssessment | OAssessment,
    expected: InputBinding,
    context: TaskContext | None = None,
) -> None:
    if assessment.binding != expected:
        raise BindingError(ReasonCode.INPUT_BINDING_MISMATCH, "assessment belongs to another input")
    if isinstance(assessment, OAssessment):
        if context is None:
            raise BindingError(ReasonCode.CONTEXT_MISSING, "F1-O requires task context")
        if (assessment.context_id != context.context_id
                or context.command_sha256 != expected.command_sha256
                or context.environment_version != expected.environment_version):
            raise BindingError(ReasonCode.CONTEXT_MISMATCH, "F1-O context binding differs")
        if context.context_status != ContextStatus.READY:
            reason = (ReasonCode.CONTEXT_REJECTED if context.context_status == ContextStatus.REJECTED
                      else ReasonCode.CONTEXT_INCOMPLETE)
            raise BindingError(reason, "F1-O context is not ready")


def validate_result_binding(result: F1Result, request: Request,
                            context: TaskContext | None,
                            inspected_image: ImageRef) -> None:
    """Check result identity. This does not authorize a VLM call."""
    if result.request_id != request.request_id:
        raise BindingError(ReasonCode.INPUT_BINDING_MISMATCH, "result request_id differs")
    if result.decision != Decision.PASS:
        return
    ready = require_ready_context(request, context)
    approved = result.approved_input
    # F1Result validation guarantees approved is present for PASS.
    assert approved is not None
    if (approved.command != request.command
            or approved.command_sha256 != command_sha256(request.command)
            or approved.source_image_sha256 != request.image.sha256
            or approved.image != inspected_image
            or approved.environment_version != request.environment_version
            or approved.context_id != ready.context_id):
        raise BindingError(ReasonCode.INPUT_BINDING_MISMATCH, "approved input differs from checked input")
