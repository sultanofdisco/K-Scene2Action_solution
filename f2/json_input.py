"""Bounded, strict JSON decoding for model responses and host configuration."""
import json
import math
from pathlib import Path

MAX_JSON_BYTES = 1_048_576


def parse_json(raw: str | bytes):
    def pairs(items):
        output = {}
        for key, value in items:
            if key in output:
                raise ValueError('Duplicate JSON key: ' + key)
            output[key] = value
        return output

    def reject_constant(value):
        raise ValueError('Non-finite JSON value: ' + value)

    def finite_float(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError('Non-finite JSON number')
        return parsed

    if isinstance(raw, str):
        raw = raw.encode('utf-8')
    if not isinstance(raw, bytes) or len(raw) > MAX_JSON_BYTES:
        raise ValueError('Expected JSON text of at most 1 MiB')
    result = json.loads(raw.decode('utf-8-sig'), object_pairs_hook=pairs,
                        parse_constant=reject_constant, parse_float=finite_float)
    pending, count = [(result, 0)], 0
    while pending:
        value, depth = pending.pop()
        count += 1
        if depth > 16 or count > 20000:
            raise ValueError('JSON nesting/node budget exceeded')
        if isinstance(value, dict):
            pending.extend((v, depth + 1) for v in value.values())
        elif isinstance(value, list):
            pending.extend((v, depth + 1) for v in value)
    return result


def load_json(path):
    with Path(path).open('rb') as handle:
        raw = handle.read(MAX_JSON_BYTES + 1)
    return parse_json(raw)
