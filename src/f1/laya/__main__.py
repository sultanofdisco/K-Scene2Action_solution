"""Consume an existing OCR run; this CLI does not call VLM or approve inputs."""
import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
import time

from f1.bindings import BindingError, bind_input, verify_image_bytes
from f1.schemas import ImageRef, OCRResult, Request, TaskContext
from .backend import LayaBackend, prepare_model
from .pipeline import assess


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Experimental F1-O Laya base-model adapter")
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare-model")
    prep.add_argument("--output", type=Path, default=Path("artifacts/laya/model"))
    prep.add_argument("--revision", default="main")
    run = sub.add_parser("assess")
    run.add_argument("--request", type=Path, required=True)
    run.add_argument("--context", type=Path, required=True)
    run.add_argument("--ocr-dir", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--model", type=Path, default=Path("artifacts/laya/model"))
    run.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    try:
        if args.action == "prepare-model":
            with redirect_stdout(sys.stderr):
                lock = prepare_model(args.output, args.revision)
            print(json.dumps({"repo": lock["repo"], "revision": lock["revision"], "output": str(args.output)}))
            return 0
        if args.output.exists():
            raise FileExistsError("Use a new output directory")
        request = Request.model_validate_json(args.request.read_text(encoding="utf-8"))
        context = TaskContext.model_validate_json(args.context.read_text(encoding="utf-8"))
        ocr = OCRResult.model_validate_json((args.ocr_dir / "ocr.json").read_text(encoding="utf-8"))
        previous = json.loads((args.ocr_dir / "run.json").read_text(encoding="utf-8"))
        if ImageRef.model_validate(previous["source_image"]) != request.image or ImageRef.model_validate(previous["inspected_image"]) != ocr.image:
            raise ValueError("OCR image references differ")
        verify_image_bytes(ocr.image, (args.ocr_dir / "inspected.png").read_bytes())
        binding = bind_input(request, ocr.image, previous["preprocessing_version"])
        if "binding" in previous and previous["binding"] != binding.model_dump(mode="json"):
            raise ValueError("Saved OCR binding differs")
        backend = LayaBackend(args.model, args.cpu_threads)
        trace = {}
        started = time.perf_counter()
        with redirect_stdout(sys.stderr):
            result = assess(request, context, binding, ocr, backend, trace)
        trace["elapsed_seconds_including_load"] = time.perf_counter() - started
        args.output.mkdir(parents=True, exist_ok=False)
        save(args.output / "assessment.json", result.model_dump(mode="json"))
        save(args.output / "run.json", trace)
        print(json.dumps({"status": result.status, "reason_codes": result.reason_codes,
                          "output": str(args.output), "deployment_ready": False, "f1_approval": False}))
        return 2 if result.status in {"ERROR", "UNKNOWN"} else 0
    except (OSError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "ERROR", "error_type": type(exc).__name__,
                          "reason_code": exc.reason_code if isinstance(exc, BindingError) else "F1O_ERROR",
                          "f1_approval": False}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
