"""Resume native evaluation from saved checkpoints without repeating training."""
import argparse,copy,json,runpy,time
from pathlib import Path
from datetime import datetime,timezone
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import torch
from safetensors.torch import load_file
from f1.laya.finetuned import ScorerTunedLayaBackend
from f1.laya.questions_v2 import QUESTIONS,PROMPT_VERSION
from f1.laya.pipeline import assess

def main():
    parser=argparse.ArgumentParser();parser.add_argument('run');args=parser.parse_args();out=Path(args.run)
    code=runpy.run_path('scripts/retrain_laya_full.py');read,dump,sha=code['read'],code['dump'],code['sha']
    helper=runpy.run_path('scripts/compare_laya_prompts.py');prior=runpy.run_path('scripts/train_laya_scorer.py');reports=runpy.run_path('scripts/evaluate_laya_gold.py')
    summary=json.loads((out/'training-summary.json').read_text());lock=json.loads((out/'data-lock.json').read_text())
    backend=ScorerTunedLayaBackend(out,cpu_threads=4);backend._load();adapter=helper['CandidateBackend'](backend,QUESTIONS)
    jobs=[(s,Path('data/laya_deepset_domain_v1')/(s+'.jsonl')) for s in ['train','validation']]+[('prior-gold',Path('data/laya_gold_v1/cases.jsonl')),('prior-validation',Path('data/laya_validation_v2/cases.jsonl'))]
    for name,path in jobs:
        rows=read(path);folder=out/name;folder.mkdir(exist_ok=True)
        existing=folder/'predictions.jsonl'
        if existing.exists():
            results=read(existing)
            assert [r['case_id'] for r in results]==[r['case_id'] for r in rows]
        else:
            results=[]
            for i,c in enumerate(rows):
                trace={};started=time.perf_counter();result=assess(*helper['inputs'](c),adapter,trace);a=result.questions.model_dump(mode='json')
                metadata=trace.pop('backend',{});trace['token_budget']=metadata.get('token_budget');trace['prompt_version']=PROMPT_VERSION;trace['backend_ref']='../model-metadata.json'
                actual={**c,'actual_status':str(result.status),'status_correct':str(result.status)==c['expected_status'],'answers':a,
                    'question_errors':[q for q,g in (c['expected_questions'] or {}).items() if not a[q] or a[q]['decision']!=g],
                    'reason_codes':list(result.reason_codes),'seconds':time.perf_counter()-started,'laya_executed':trace['laya_executed'],'analysis_complete':result.analysis_complete}
                if 'expected_laya_executed' in c:actual['guard_execution_correct']=actual['laya_executed']==c['expected_laya_executed']
                results.append(actual);dump(folder/(c['case_id']+'-trace.json'),trace)
                if i%16==0:print(json.dumps({'run':out.name,'set':name,'done':i+1,'total':len(rows)}),flush=True)
            code['write_rows'](existing,results)
        if name in ['train','validation']:
            assert sha(path)==lock['splits'][name]['sha256']
            cached=load_file('artifacts/laya/deepset-scorer-cpu-v1/'+name+'-features.safetensors')
            with torch.no_grad():answers=prior['predictions'](backend.agent.model.scorer(cached['features']).squeeze(-1))
            statuses=prior['statuses'](answers)
            for i,r in enumerate(results):
                assert r['actual_status']==statuses[i]
                assert all(r['answers'][q]['decision']==answers[i*4+j]['decision'] for j,q in enumerate(QUESTIONS))
        else:assert sha(path)==lock['protected_evaluation'][str(path)]
        for r in results:
            trace=json.loads((folder/(r['case_id']+'-trace.json')).read_text(encoding='utf-8'))
            if trace['laya_executed']:assert r['analysis_complete'] and all(not t['truncated'] for t in trace['token_budget'].values())
        sem=[r for r in results if r['suite']=='semantic']
        summary['evaluations'][name]={'metrics':reports['metrics'](sem),'questions':{q:sum(q not in r['question_errors'] for r in sem) for q in QUESTIONS},
            'ambiguous':reports['metrics']([r for r in results if r['expected_status']=='UNKNOWN']),
            'languages':{lang:reports['metrics']([r for r in sem if r['language']==lang]) for lang in ['en','ko']},
            'guards_passed':all(r['status_correct'] and r.get('guard_execution_correct',True) for r in results if r['suite']=='guard')}
        dump(out/'summary.json',summary)
    metadata=copy.deepcopy(backend.metadata);metadata.pop('token_budget',None);dump(out/'model-metadata.json',metadata)
    summary['all_80_native_matches_cache']=True;summary['evaluation_resumed_without_retraining']=True
    summary['completed_utc']=datetime.now(timezone.utc).isoformat();dump(out/'summary.json',summary)
    print(json.dumps({'completed':out.name,'selected_epoch':summary['selected_epoch'],'selected':summary['selected']},ensure_ascii=False))

if __name__=='__main__':main()
