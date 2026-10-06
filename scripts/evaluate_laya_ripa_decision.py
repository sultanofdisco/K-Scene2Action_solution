"""Reload selected native decision delta and verify on untouched diagnostic rows."""
import json,runpy,time
from pathlib import Path
from f1.laya.finetuned import DecisionHeadTunedLayaBackend
from f1.laya.questions_v2 import QUESTIONS,PROMPT_VERSION
from f1.laya.pipeline import assess

ROOT=Path('artifacts/laya/ripa-decision-cpu-v1');DATA=Path('data/laya_ripa_domain_v1')
def main():
    h=runpy.run_path('scripts/train_laya_scorer.py');read,dump,sha=h['read'],h['dump'],h['sha'];helpers=runpy.run_path('scripts/compare_laya_prompts.py');metrics=runpy.run_path('scripts/evaluate_laya_gold.py')['metrics']
    train=json.loads((ROOT/'training-summary.json').read_text());lock=json.loads((DATA/'data-lock.json').read_text());assert not (ROOT/'native-summary.json').exists()
    backend=DecisionHeadTunedLayaBackend(ROOT,cpu_threads=4);backend._load();model=backend.agent.model
    assert h['tensors_hash'](model.encoder.named_parameters())==train['initial_encoder_parameter_hash']
    restored={k:v for k,v in model.state_dict().items() if k.startswith(('head.','scorer.'))}
    from safetensors.torch import load_file
    import torch
    assert all(torch.equal(v,restored[k]) for k,v in load_file(str(ROOT/'decision.safetensors')).items())
    adapter=helpers['CandidateBackend'](backend,QUESTIONS);baseline_weight_hash=sha(Path('artifacts/laya/model/model.safetensors'))
    frozen={'decision.safetensors':sha(ROOT/'decision.safetensors'),'decision-lock.json':sha(ROOT/'decision-lock.json')};dump(ROOT/'pre-diagnostic-freeze.json',frozen)
    jobs={'legacy_validation':read(DATA/'legacy_validation.jsonl'),'ripa_validation':read(DATA/'ripa_validation.jsonl'),
          'stress-v1':read(Path('artifacts/laya/fp-profile-cpu-v1/stress-cases.jsonl')),
          'stress-v2':read(Path('artifacts/laya/fp-support-cpu-v1/stress-v2-cases.jsonl')),
          'guards':[c for p in ['data/laya_gold_v1/cases.jsonl','data/laya_validation_v2/cases.jsonl'] for c in read(Path(p)) if c['suite']=='guard']}
    summary={'selected_epoch':train['selected_epoch'],'encoder_exactly_restored':True,'decision_delta_roundtrip_exact':True,'sets':{},'deployment_ready':False,'frozen':frozen}
    start=time.perf_counter()
    for name,cases in jobs.items():
        path=ROOT/(name+'-native.jsonl');rows=read(path) if path.exists() else []
        assert [r['case_id'] for r in rows]==[c['case_id'] for c in cases[:len(rows)]]
        for i,c in enumerate(cases[len(rows):],start=len(rows)):
            trace={};t=time.perf_counter();result=assess(*helpers['inputs'](c),adapter,trace);a=result.questions.model_dump(mode='json')
            row={**c,'actual_status':str(result.status),'status_correct':str(result.status)==c['expected_status'],'answers':a,
                 'question_errors':[q for q,v in (c['expected_questions'] or {}).items() if not a[q] or a[q]['decision']!=v],
                 'laya_executed':trace['laya_executed'],'analysis_complete':result.analysis_complete,'seconds':time.perf_counter()-t,
                 'reason_codes':[str(x) for x in result.reason_codes],'prompt_version':PROMPT_VERSION}
            if c['suite']=='guard':assert row['status_correct'] and row['laya_executed']==c['expected_laya_executed']
            else:assert result.analysis_complete and trace['laya_executed'] and all(not b['truncated'] for b in trace['backend']['token_budget'].values())
            rows.append(row)
            with path.open('a',encoding='utf8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
            if (i+1)%10==0 or i+1==len(cases):print(json.dumps({'stage':'native_evaluation','set':name,'done':i+1,'total':len(cases),'seconds':time.perf_counter()-start}),flush=True)
        summary['sets'][name]={'metrics':metrics(rows),'question_accuracy':sum(4-len(r['question_errors']) for r in rows)/(len(rows)*4) if name!='guards' else None,
           'guards_passed':all(r['status_correct'] and r['laya_executed']==r['expected_laya_executed'] for r in rows if r['suite']=='guard')}
        if name in ['legacy_validation','ripa_validation']:
            cached=read(ROOT/(name+'-predictions.jsonl'));assert [r['case_id'] for r in cached]==[r['case_id'] for r in rows]
            for r,c in zip(rows,cached):
                assert r['actual_status']==c['actual_status']
                assert all(r['answers'][q]['decision']==c['answers'][q]['decision'] for q in QUESTIONS)
            summary['sets'][name]['all_native_answers_match_cache']=True
        dump(ROOT/'native-summary-progress.json',summary)
    for n,v in frozen.items():assert sha(ROOT/n)==v
    for p,v in lock['protected_files'].items():assert sha(Path(p))==v
    assert sha(Path('artifacts/laya/model/model.safetensors'))==baseline_weight_hash
    summary['completed']=True;summary['seconds']=time.perf_counter()-start;dump(ROOT/'native-summary.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
