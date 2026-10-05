import argparse
import json
import sys

from .filter import Issue, Result
from . import inspect_plan, inspect_steps
from .json_input import load_json


def main():
    if len(sys.argv) > 1 and sys.argv[1] in {'setup', 'check'}:
        from .service_cli import main as contract_main
        return contract_main()
    parser = argparse.ArgumentParser(description='Legacy v0.1 checks. Use setup/check and GatedExecutor for Contract F2.')
    parser.add_argument('--plan', required=True)
    parser.add_argument('--context', required=True)
    parser.add_argument('--format', choices=('normalized', 'steps'), default='normalized')
    args = parser.parse_args()
    try:
        plan, context = load_json(args.plan), load_json(args.context)
        result = (inspect_steps if args.format == 'steps' else inspect_plan)(plan, context)
    except (OSError, ValueError, RecursionError, OverflowError) as exc:
        result = Result('DEFER', (Issue('INPUT_READ_ERROR', 'DEFER', str(exc), '$'),))
    print(json.dumps(result.to_dict(), ensure_ascii=True, allow_nan=False, indent=2))
    return {'PASS': 0, 'BLOCK': 2, 'DEFER': 3}[result.decision]


if __name__ == '__main__':
    sys.exit(main())
