"""CPU preflight only: excludes sensors, trajectory planning, VLM/model startup."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import statistics
import sys
import time
import tracemalloc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from demo import demo_inputs, demo_state
from f2 import RuntimeFilter, GatedExecutor
from f2.json_input import load_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=2000)
    parser.add_argument('--output')
    args = parser.parse_args()
    if not 100 <= args.iterations <= 100000:
        parser.error('iterations must be between 100 and 100000')
    start = time.perf_counter()
    contract, plan = demo_inputs()
    load_ms = (time.perf_counter()-start)*1000
    runtime = RuntimeFilter(contract)
    for _ in range(100):
        runtime.inspect(plan, demo_state(contract), step_gated=True)
    measurements, decisions = [], Counter()
    for _ in range(args.iterations):
        observation = demo_state(contract)
        start = time.perf_counter_ns()
        result = runtime.inspect(plan, observation, step_gated=True)
        measurements.append((time.perf_counter_ns()-start)/1e6)
        decisions[result.decision] += 1
    ordered = sorted(measurements)
    tracemalloc.start()
    for _ in range(100): runtime.inspect(plan, demo_state(contract), step_gated=True)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    gate = GatedExecutor(runtime, max_requests=100000)
    staged=[]
    staged_iterations=min(args.iterations,1000)
    for request_index in range(staged_iterations):
        def observe(index, request_index=request_index):
            obj,dest=('red_ball','zone1') if index<8 else ('blue_ball','zone2')
            s=demo_state(contract,index,obj,dest,request=f'bench-{request_index}')
            s['episode_id']=f'synthetic-bench-episode-{request_index}'
            return s
        start=time.perf_counter_ns()
        r=gate.run(plan,observe,lambda step:True,lambda:True)
        if r.decision!='PASS': raise RuntimeError('staged benchmark did not pass')
        staged.append((time.perf_counter_ns()-start)/1e6)
    cases = load_json(ROOT/'examples/recorded_drop_plans.json') + load_json(ROOT/'examples/contract/recorded_runpod_plans.json')
    replay = []
    for case in cases:
        r = runtime.inspect(case['plan'], demo_state(contract), step_gated=True)
        replay.append(dict(case_id=case['case_id'],model=case.get('model_label','sonnet'), model_decision=case['plan']['decision'],
                           decision=r.decision, reason_codes=r.reason_codes, harmful_categories=r.harmful_categories))
    report = dict(measured_at_utc=datetime.now(timezone.utc).isoformat(), platform=platform.platform(), python=platform.python_version(),
                  cpu=platform.processor(), scope='CPU preflight on synthetic observations; no VLM, sensors, actuators or RunPod',
                  iterations=args.iterations, warmup_iterations=100, plan_steps=len(plan['steps']),
                  stored_contract_load_and_validation_ms=load_ms,
                  p50_ms=statistics.median(measurements), p95_ms=ordered[int(.95*(len(ordered)-1))], max_ms=max(ordered),
                  mean_ms=statistics.mean(measurements), decisions=dict(decisions),
                  traced_python_peak_bytes=peak, traced_python_current_bytes=current,
                  memory_scope='separate 100-check Python allocation trace, not process RSS or GPU VRAM',
                  runtime_llm_calls=0, gpu_inference_calls=0, recorded_plan_replay=replay)
    report.update(staged_mock_iterations=staged_iterations, staged_mock_p50_ms=statistics.median(staged),
                  staged_mock_p95_ms=sorted(staged)[int(.95*(len(staged)-1))],
                  staged_mock_scope='Includes Python mock observation/16 execution callbacks and final outcome checks; no ROS2/camera I/O')
    text=json.dumps(report, ensure_ascii=False, indent=2)+'\n'
    if args.output: Path(args.output).write_text(text,encoding='utf-8')
    print(text)


if __name__ == '__main__': main()
