"""Setup and inspect a Contract. No actuator invocation or remote deployment."""
import argparse
import json
import os
import sys

from .contract import ContractError, ContractManager, load_contract
from .filter import Issue
from .json_input import load_json
from .llm_setup import JsonSchemaMapper
from .runtime import RuntimeFilter, RuntimeResult, extract_plan


def main(argv=None):
    parser = argparse.ArgumentParser(description='Initial LLM Contract setup / CPU preflight checks')
    commands = parser.add_subparsers(dest='command', required=True)
    setup = commands.add_parser('setup')
    setup.add_argument('--task', required=True)
    setup.add_argument('--environment', required=True)
    setup.add_argument('--output', required=True)
    source = setup.add_mutually_exclusive_group(required=True)
    source.add_argument('--mapping', help='Offline mapping fixture; does not invoke an LLM')
    source.add_argument('--llm-url', help='Explicit trusted initial-setup endpoint')
    setup.add_argument('--model', default='Qwen/Qwen3-4B-Instruct-2507')
    setup.add_argument('--contrasts', help='Trusted normal/unsafe contrast cases')
    check = commands.add_parser('check')
    check.add_argument('--contract', required=True)
    check.add_argument('--task', required=True)
    check.add_argument('--environment', required=True)
    check.add_argument('--plan', required=True)
    check.add_argument('--state', required=True)
    check.add_argument('--contrasts', help='Revalidate semantic contrasts when no trusted expected mapping exists')
    check.add_argument('--stage-gated', action='store_true', help='Host promises GatedExecutor integration; check alone does not execute')
    args = parser.parse_args(argv)
    try:
        task, environment = load_json(args.task), load_json(args.environment)
        if args.command == 'setup':
            if args.mapping:
                mapping = load_json(args.mapping)
                mapper, ref = lambda payload, schema: mapping, 'offline-mapping-fixture-v1'
            else:
                mapper = JsonSchemaMapper(args.llm_url, args.model, api_key=os.environ.get('F2_LLM_API_KEY'))
                ref = args.model + ':json-schema-v1'
            manager = ContractManager(mapper, ref)
            contract = manager.prepare(task, environment, load_json(args.contrasts) if args.contrasts else ())
            contract.save(args.output)
            print(json.dumps({'activation_state': 'ACTIVE', 'contract_id': contract.contract_id,
                              'mapping_source': ref, 'llm_calls': manager.llm_calls if args.llm_url else 0}, indent=2))
            return 0
        contract = load_contract(args.contract, task, environment, contrasts=load_json(args.contrasts) if args.contrasts else ())
        result = RuntimeFilter(contract).inspect(extract_plan(load_json(args.plan)), load_json(args.state), step_gated=args.stage_gated)
    except (OSError, ValueError, TypeError, KeyError, OverflowError, RecursionError) as exc:
        result = RuntimeResult('DEFER', (Issue('SETUP_OR_INPUT_FAILED', 'DEFER', str(exc), '$'),))
    print(json.dumps(result.to_dict(), ensure_ascii=True, allow_nan=False, indent=2))
    return {'PASS': 0, 'BLOCK': 2, 'DEFER': 3}[result.decision]


if __name__ == '__main__':
    sys.exit(main())
