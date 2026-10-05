"""CLI for schema export and format checks only; never runs inference."""

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from .bindings import BindingError, require_ready_context, verify_image_bytes
from .schemas import PUBLIC_SCHEMAS, Request, TaskContext


def export_schemas(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for model in PUBLIC_SCHEMAS:
        schema = model.model_json_schema()
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        (directory / f"{model.__name__}.schema.json").write_text(
            json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="F1 interface tools (no model inference)")
    commands = parser.add_subparsers(dest="action", required=True)
    export = commands.add_parser("export-schemas")
    export.add_argument("directory", type=Path)
    validate = commands.add_parser("validate-input")
    validate.add_argument("--request", type=Path, required=True)
    validate.add_argument("--context", type=Path)
    validate.add_argument("--image", type=Path)
    args = parser.parse_args()
    try:
        if args.action == "export-schemas":
            export_schemas(args.directory)
            print(f"Exported {len(PUBLIC_SCHEMAS)} schemas to {args.directory}")
            return 0
        request = Request.model_validate_json(args.request.read_text(encoding="utf-8"))
        checks = ["request_schema"]
        if args.context is not None:
            context = TaskContext.model_validate_json(args.context.read_text(encoding="utf-8"))
            require_ready_context(request, context)
            checks.append("context_schema_and_binding")
        if args.image is not None:
            verify_image_bytes(request.image, args.image.read_bytes())
            checks.append("encoded_image_sha256")
        print(json.dumps({
            "status": "VALID_IO_ONLY", "request_id": request.request_id,
            "checks": checks, "inference_executed": False,
        }, ensure_ascii=False))
        return 0
    except BindingError as error:
        payload = {"status": "INVALID_IO", "reason_code": error.reason_code.value,
                   "message": str(error)}
    except ValidationError as error:
        # Do not echo raw command/OCR payloads in error logs.
        payload = {"status": "INVALID_IO", "reason_code": "INVALID_REQUEST",
                   "errors": error.errors(include_input=False, include_context=False,
                                          include_url=False)}
    except (OSError, UnicodeError) as error:
        payload = {"status": "INVALID_IO", "message": str(error)}
    print(json.dumps(payload, ensure_ascii=False), file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
