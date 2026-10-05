"""Contract/runtime demo with explicitly synthetic observations, no robot/LLM."""
import copy
import json
from pathlib import Path

import time

from f2 import RuntimeFilter, GatedExecutor, load_contract
from f2.contract import GUARDS
from f2.json_input import load_json

ROOT = Path(__file__).resolve().parent


def demo_state(contract, index=0, obj='red_ball', destination='zone1', request='demo-1'):
    now = time.monotonic()
    facts = {}
    for key in ('controller_ready', 'initial_gripper_empty', 'motion_safety_clear',
                'holding_confirmed:' + obj, 'support_confirmed:' + obj + ':' + destination):
        facts[key] = dict(value=True, source='sensor', observed_at=now,
                          step_index=index-1, evidence='SYNTHETIC demo observation, not real robot evidence')
    for target, (goal, _, _) in contract.goals.items():
        facts[f'outcome_confirmed:{target}:{goal}'] = dict(value=True, source='sensor', observed_at=now,
            step_index=index-1, evidence='SYNTHETIC final outcome, not real robot evidence')
    return dict(bindings=contract.bindings_dict(), episode_id='synthetic-demo', request_id=request,
                observed_at=now, rgb_stamp=10.0, depth_stamp=10.0, facts=facts,
                capabilities=sorted({g for guards in GUARDS.values() for g in guards}))


def demo_inputs():
    folder = ROOT / 'examples/contract'
    contract = load_contract(folder / 'contract_fixture.json', load_json(folder/'task.json'), load_json(folder/'environment.json'))
    return contract, load_json(folder/'normal_plan.json')


def main():
    contract, normal = demo_inputs()
    recorded = load_json(ROOT / 'examples/recorded_drop_plans.json')
    runpod = load_json(ROOT / 'examples/contract/recorded_runpod_plans.json')
    runtime = RuntimeFilter(contract)
    calls = []
    def observe(index):
        obj, dest = ('red_ball', 'zone1') if index < 8 else ('blue_ball', 'zone2')
        return demo_state(contract, index, obj, dest)
    result = GatedExecutor(runtime).run(normal, observe, lambda step: calls.append(step) or True, lambda: True)
    output = {'mode': 'LOCAL_SYNTHETIC_OBSERVATIONS_NO_ACTUATORS', 'runtime_llm_calls': 0,
              'normal': {'decision': result.decision, 'completed_steps': result.completed_steps}, 'recorded_cases': {}}
    for case in recorded + runpod:
        result = runtime.inspect(case['plan'], demo_state(contract), step_gated=True)
        key = case.get('model_label', 'sonnet') + '/' + case['case_id']
        output['recorded_cases'][key] = {'decision': result.decision, 'reason_codes': result.reason_codes}
    def missing_support(index):
        s = observe(index)
        if index == 6: del s['facts']['support_confirmed:red_ball:zone1']
        return s
    halted = []
    result = GatedExecutor(runtime).run(normal, missing_support, lambda step: halted.append(step) or True, lambda: True)
    output['missing_support'] = {'decision': result.decision, 'completed_steps': result.completed_steps,
                                 'release_executed': any(s['action']=='release' for s in halted)}
    output['normal_adapter_step_count'] = len(calls)
    print(json.dumps(output, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
