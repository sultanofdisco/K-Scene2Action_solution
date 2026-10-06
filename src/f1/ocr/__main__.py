"""CPU OCR extraction CLI. Output is OCRResult, never an F1 approval."""
import argparse
from contextlib import redirect_stdout
from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time

from f1.bindings import BindingError, bind_input
from f1.schemas import Request, OCRResult, OCRStatus, ReasonCode
from .backend import PaddleBackend, prepare_models
from .pipeline import OCRPolicy, PREPROCESSING_VERSION, prepare_image, extract_regions


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="F1-O OCR extraction, no attack classification")
    sub = parser.add_subparsers(dest="action", required=True)
    models = sub.add_parser("prepare-models")
    models.add_argument("--output", type=Path, default=Path("artifacts/ocr/models"))
    run = sub.add_parser("extract")
    run.add_argument("--image", type=Path, required=True)
    run.add_argument("--request", type=Path)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--models", type=Path, default=Path("artifacts/ocr/models"))
    run.add_argument("--min-score", type=float, default=0.8)
    run.add_argument("--cpu-threads", type=int, default=4)
    run.add_argument("--det-limit", type=int, default=1536)
    args = parser.parse_args()
    try:
        if args.action == "prepare-models":
            with redirect_stdout(sys.stderr):
                lock = prepare_models(args.output)
            print(json.dumps(lock, indent=2))
            return 0
        policy = OCRPolicy(min_recognition_score=args.min_score)
        if args.cpu_threads <= 0 or not 32 <= args.det_limit <= 4000:
            raise ValueError("Invalid CPU threads or detector size")
        request = Request.model_validate_json(args.request.read_text(encoding="utf-8")) if args.request else None
        # Read only a bounded amount once; all downstream models use these same bytes.
        with args.image.open("rb") as stream:
            content = stream.read(policy.max_file_bytes+1)
        prepared = prepare_image(content, request.image if request else None, policy)
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "inspected.png").write_bytes(prepared.png)
        metadata = {"preprocessing_version": PREPROCESSING_VERSION, "source_image": prepared.source.model_dump(),
                    "inspected_image": prepared.inspected.model_dump(), "policy": asdict(policy),
                    "cpu_threads": args.cpu_threads, "det_limit": args.det_limit,
                    "f1_approval": False, "laya_executed": False,
                    "packages": {p: version(p) for p in ["paddleocr", "paddlepaddle", "paddlex", "Pillow", "numpy"]}}
        if request:
            metadata["binding"] = bind_input(request, prepared.inspected, PREPROCESSING_VERSION).model_dump(mode="json")
        started = time.perf_counter()
        try:
            with redirect_stdout(sys.stderr):
                backend = PaddleBackend(args.models, args.cpu_threads, args.det_limit)
                metadata["model_lock"] = backend.lock
                result = extract_regions(prepared, backend, policy)
        except Exception as exc:
            metadata["error_type"] = type(exc).__name__
            print(f"OCR initialization failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            result = OCRResult(image=prepared.inspected, status=OCRStatus.ERROR, regions=(),
                               reason_codes=(ReasonCode.OCR_ERROR,), model_version="ocr-unavailable")
        metadata["elapsed_seconds_including_load"] = time.perf_counter()-started
        save(args.output / "ocr.json", result.model_dump(mode="json"))
        save(args.output / "run.json", metadata)
        print(json.dumps({"status": result.status, "regions": len(result.regions),
                          "reason_codes": result.reason_codes, "output": str(args.output),
                          "f1_approval": False}, ensure_ascii=False))
        return 0 if result.status == OCRStatus.COMPLETE else 2
    except (OSError, ValueError) as exc:
        reason = exc.reason_code.value if isinstance(exc, BindingError) else "OCR_ERROR"
        # Validation errors may contain raw commands; don't echo them.
        print(json.dumps({"status": "ERROR", "reason_code": reason, "error_type": type(exc).__name__}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
