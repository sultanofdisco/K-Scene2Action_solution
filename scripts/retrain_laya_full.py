"""Frozen 58/22 scorer retraining with a prespecified validation selection rule."""
import argparse, copy, hashlib, html, json, math, os, runpy, time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import torch
import torch.nn.functional as F
from safetensors.torch import load_file,save_file
from f1.laya.backend import LayaBackend
from f1.laya.finetuned import apply_scorer_delta,lock_hash
from f1.laya.pipeline import assess
from f1.laya.questions_v2 import QUESTIONS,PROMPT_VERSION

OUT=Path('artifacts/laya/full-retrain-cpu-v2')
DATA=Path('data/laya_deepset_domain_v1')
OLD=Path('artifacts/laya/deepset-scorer-cpu-v1')
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def read(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write_rows(p,rows):p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')

def selection_key(metrics,question_correct,loss):
    # First retain all known validation attacks, then improve normal utility.
    return (metrics['attack_detect_count']==metrics['attack_n'],metrics['attack_detect_count'],
            -metrics['attack_clear_count'],metrics['benign_clear_count'],metrics['ambiguous_correct'],
            question_correct,-loss)

def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default=str(OUT))
    parser.add_argument('--balanced',action='store_true')
    args=parser.parse_args();OUT=Path(args.output)
    OUT.mkdir(parents=True,exist_ok=False)
    lock=json.loads((DATA/'data-lock.json').read_text(encoding='utf-8'))
    assert lock['questions']==QUESTIONS
    cases={s:read(DATA/(s+'.jsonl')) for s in ['train','validation']}
    for s in cases:assert sha(DATA/(s+'.jsonl'))==lock['splits'][s]['sha256']
    groups={};texts={s:set() for s in cases}
    for s,rows in cases.items():
        for r in rows:
            assert groups.setdefault(r['group_id'],s)==s
            texts[s].add(tuple(r['ocr_texts']))
            answers=list(r['expected_questions'].values())
            status='ATTACK' if 'YES' in answers[:3] else 'CLEAR' if answers==['NO','NO','NO','YES'] else 'UNKNOWN'
            assert status==r['expected_status']
    assert not texts['train']&texts['validation']
    audit={'reviewer':'Codex','human_reviewed':False,'labels_changed':False,'splits_preserved':True,
           'no_exact_transcript_overlap':True,'groups_preserved':True,'aggregation_labels_consistent':True,
           'counts':{s:dict(Counter(r['expected_status'] for r in rows)) for s,rows in cases.items()},
           'question_counts':{s:{q:dict(Counter(r['expected_questions'][q] for r in rows)) for q in QUESTIONS} for s,rows in cases.items()},
           'interpretation_sensitive_cases':[r['case_id'] for rows in cases.values() for r in rows if r['origin']=='synthetic_ambiguous'],
           'limitations':['Ambiguous replacement phrases may count as task override under a stricter policy; provisional labels retained for comparability.',
                          'Quoted attacks are labeled benign only under explicit educational quotation framing.',
                          'Public text adapted to illustrative robot context, not original robot observations.',
                          'Rare harm and authority positives are synthetic; human label review pending.']}
    dump(OUT/'data-audit.json',audit);dump(OUT/'data-lock.json',lock)
    plan={'created_utc':datetime.now(timezone.utc).isoformat(),'start':'original pinned Laya base model; not memorization weights',
          'train_cases':58,'validation_cases':22,'trainable':'native scorer only','trainable_parameters':592897,
          'epochs':6000,'batch_size':232,'optimizer':'AdamW','weight_decay':0.0,'loss':'unweighted cross entropy',
          'learning_rate_start':0.001,'learning_rate_end':0.00001,'schedule':'cosine decay over all updates',
          'eval_every':50,'seed':42,'cpu_threads':4,'selection':'lexicographic: detect all 9 validation attacks, attack detection count, fewer attack CLEAR, more benign CLEAR, more correct ambiguous cases, question correct count, lower validation CE',
          'baseline_is_candidate':True,'selection_uses_prior_diagnostic_sets':False,'threshold_tuning':False,
          'data_lock_sha256':sha(DATA/'data-lock.json'),'deployment_ready':False}
    if args.balanced:
        plan['loss']='question/class balanced cross entropy; weights from train only'
        plan['reason']='Unweighted run lost validation attack detection; test loss weighting only with the same data, schedule and selection rule.'
    dump(OUT/'plan.json',plan)
    torch.manual_seed(42);torch.set_num_threads(4)
    prior=runpy.run_path('scripts/train_laya_scorer.py');helpers=runpy.run_path('scripts/compare_laya_prompts.py');reports=runpy.run_path('scripts/evaluate_laya_gold.py')
    backend=LayaBackend(cpu_threads=4);backend._load();adapter=helpers['CandidateBackend'](backend,QUESTIONS);model=backend.agent.model
    old_manifest=json.loads((OLD/'scorer-lock.json').read_text())
    assert old_manifest['base_model_lock_sha256']==lock_hash(backend.metadata['model_lock'])
    assert old_manifest['training_data_lock_sha256']==plan['data_lock_sha256']
    for p in model.parameters():p.requires_grad_(False)
    frozen_hash=prior['tensors_hash']((n,p) for n,p in model.named_parameters() if not n.startswith('scorer.'))
    original=copy.deepcopy(model.scorer.state_dict())
    cache={};cache_audit={}
    for split in cases:
        p=OLD/(split+'-features.safetensors');features=load_file(str(p))
        expected=torch.tensor([['YES','NO','UNKNOWN'].index(r['expected_questions'][q]) for r in cases[split] for q in QUESTIONS])
        assert torch.equal(features['labels'],expected)
        cache[split]=(features['features'],features['labels'])
        cache_audit[split]={'sha256':sha(p),'label_order_verified':True,'native_spot_checks':[]}
    # Previous experiment checked all cached predictions before writing the cache.
    # Recheck representative normal, attack and ambiguous native features now.
    captured=[];hook=model.scorer.register_forward_pre_hook(lambda m,a:captured.append(a[0].detach().clone()))
    for split in cases:
        indices=sorted({0,len(cases[split])-1,*[next(i for i,c in enumerate(cases[split]) if c['expected_status']==s) for s in ['CLEAR','ATTACK','UNKNOWN']]})
        for i in indices:
            captured.clear();trace={};r=assess(*helpers['inputs'](cases[split][i]),adapter,trace)
            assert r.analysis_complete and trace['laya_executed']
            difference=float((captured[0]-cache[split][0][i*4:i*4+4]).abs().max())
            assert difference<1e-5
            cache_audit[split]['native_spot_checks'].append({'case_id':cases[split][i]['case_id'],'max_abs_difference':difference})
    hook.remove();dump(OUT/'cache-audit.json',cache_audit)
    scorer=model.scorer
    for p in scorer.parameters():p.requires_grad_(True)
    def evaluate(split,with_rows=False):
        x,y=cache[split];scorer.eval()
        with torch.no_grad():
            logits=scorer(x).squeeze(-1);loss=float(F.cross_entropy(logits,y));pred=logits.argmax(-1).reshape(-1,4)
            statuses=['ATTACK' if 0 in p[:3] else 'CLEAR' if p==[1,1,1,0] else 'UNKNOWN' for p in pred.tolist()]
            question_correct=int((pred.flatten()==y).sum())
        rows=cases[split]
        metrics={'n':len(rows),'correct':sum(c['expected_status']==s for c,s in zip(rows,statuses)),
                 'attack_n':sum(c['expected_status']=='ATTACK' for c in rows),
                 'attack_detect_count':sum(c['expected_status']=='ATTACK' and s=='ATTACK' for c,s in zip(rows,statuses)),
                 'attack_clear_count':sum(c['expected_status']=='ATTACK' and s=='CLEAR' for c,s in zip(rows,statuses)),
                 'benign_n':sum(c['expected_status']=='CLEAR' for c in rows),
                 'benign_clear_count':sum(c['expected_status']=='CLEAR' and s=='CLEAR' for c,s in zip(rows,statuses)),
                 'ambiguous_correct':sum(c['expected_status']=='UNKNOWN' and s=='UNKNOWN' for c,s in zip(rows,statuses)),
                 'unknown_count':statuses.count('UNKNOWN')}
        result={'loss':loss,'question_correct':question_correct,'metrics':metrics}
        if with_rows:
            answers=prior['predictions'](logits);assert statuses==prior['statuses'](answers)
            result['rows']=[{**c,'actual_status':statuses[i],'status_correct':statuses[i]==c['expected_status'],
                            'answers':dict(zip(QUESTIONS,answers[i*4:i*4+4])),
                            'question_errors':[q for j,q in enumerate(QUESTIONS) if answers[i*4+j]['decision']!=c['expected_questions'][q]]} for i,c in enumerate(rows)]
        return result
    baseline={s:evaluate(s,True) for s in cases}
    for s in cases:write_rows(OUT/(s+'-baseline.jsonl'),baseline[s].pop('rows'))
    best=copy.deepcopy(original);best_epoch=0;v=baseline['validation'];best_key=selection_key(v['metrics'],v['question_correct'],v['loss'])
    history=[{'epoch':0,'train':baseline['train'],'validation':baseline['validation']}]
    optimizer=torch.optim.AdamW(scorer.parameters(),lr=plan['learning_rate_start'],weight_decay=0)
    x,y=cache['train'];started=time.perf_counter()
    sample_weights=torch.ones(len(y))
    if args.balanced:
        weights=torch.zeros(4,3)
        for q in range(4):
            counts=torch.bincount(y[q::4],minlength=3)
            for label in range(3):
                if counts[label]:weights[q,label]=len(y)/(4*int((counts>0).sum())*int(counts[label]))
        sample_weights=weights[torch.arange(len(y))%4,y]
        dump(OUT/'class-weights.json',dict(zip(QUESTIONS,weights.tolist())))
    for epoch in range(1,plan['epochs']+1):
        lr=plan['learning_rate_end']+(plan['learning_rate_start']-plan['learning_rate_end'])*(1+math.cos(math.pi*(epoch-1)/(plan['epochs']-1)))/2
        optimizer.param_groups[0]['lr']=lr
        scorer.train();logits=scorer(x).squeeze(-1);loss=(F.cross_entropy(logits,y,reduction='none')*sample_weights).sum()/sample_weights.sum();optimizer.zero_grad();loss.backward()
        assert torch.isfinite(loss)
        torch.nn.utils.clip_grad_norm_(scorer.parameters(),1.0);optimizer.step()
        if epoch%plan['eval_every']==0:
            t=evaluate('train');v=evaluate('validation');key=selection_key(v['metrics'],v['question_correct'],v['loss'])
            if key>best_key:best_key=key;best_epoch=epoch;best=copy.deepcopy(scorer.state_dict())
            history.append({'epoch':epoch,'learning_rate':lr,'train':t,'validation':v,'selected_epoch_so_far':best_epoch})
            if epoch%250==0:
                dump(OUT/'history.json',history);print(json.dumps({'epoch':epoch,'seconds':time.perf_counter()-started,'train':t,'validation':v,'selected_epoch':best_epoch}),flush=True)
    last={s:evaluate(s,True) for s in cases}
    for s in cases:write_rows(OUT/(s+'-last-predictions.jsonl'),last[s].pop('rows'))
    save_file({k:v.contiguous() for k,v in scorer.state_dict().items()},str(OUT/'last-scorer.safetensors'))
    scorer.load_state_dict(best);model.eval()
    assert frozen_hash==prior['tensors_hash']((n,p) for n,p in model.named_parameters() if not n.startswith('scorer.'))
    save_file({k:v.contiguous() for k,v in best.items()},str(OUT/'scorer.safetensors'))
    manifest={'base_model_lock_sha256':lock_hash(backend.metadata['model_lock']),'base_revision':backend.metadata['model_lock']['revision'],
              'scorer_sha256':sha(OUT/'scorer.safetensors'),'trainable_module':'scorer','prompt_version':PROMPT_VERSION,
              'selected_epoch':best_epoch,'selection':plan['selection'],'training_data_lock_sha256':plan['data_lock_sha256'],'deployment_ready':False}
    dump(OUT/'scorer-lock.json',manifest);apply_scorer_delta(backend,OUT)
    summary={'plan':plan,'training_seconds':time.perf_counter()-started,'selected_epoch':best_epoch,'baseline':baseline,'last':last,'selected':{},'evaluations':{},
             'frozen_weights_unchanged':True,'scorer_changed':any(not torch.equal(original[k],v) for k,v in scorer.state_dict().items()),'deployment_ready':False}
    expected={}
    for s in cases:
        result=evaluate(s,True);expected[s]=result.pop('rows');summary['selected'][s]=result
    dump(OUT/'training-summary.json',summary)
    # Checkpoint is now fixed. Native replay of all 80 cases, then prior diagnostics.
    jobs=[(s,cases[s]) for s in cases]
    for name,p in [('prior-gold',Path('data/laya_gold_v1/cases.jsonl')),('prior-validation',Path('data/laya_validation_v2/cases.jsonl'))]:
        assert sha(p)==lock['protected_evaluation'][str(p)];jobs.append((name,read(p)))
    for name,rows in jobs:
        folder=OUT/name;folder.mkdir();results=[]
        for i,c in enumerate(rows):
            trace={};r=assess(*helpers['inputs'](c),adapter,trace);a=r.questions.model_dump(mode='json')
            meta=trace.pop('backend',{});trace['token_budget']=meta.get('token_budget');trace['prompt_version']=PROMPT_VERSION
            trace['backend_ref']='../model-metadata.json'
            actual={**c,'actual_status':str(r.status),'status_correct':str(r.status)==c['expected_status'],'answers':a,
                    'question_errors':[q for q,gold in (c['expected_questions'] or {}).items() if not a[q] or a[q]['decision']!=gold],
                    'reason_codes':list(r.reason_codes),'laya_executed':trace['laya_executed'],'analysis_complete':r.analysis_complete}
            if name in expected:
                assert str(r.status)==expected[name][i]['actual_status']
                assert all(a[q]['decision']==expected[name][i]['answers'][q]['decision'] for q in QUESTIONS)
            if 'expected_laya_executed' in c:actual['guard_execution_correct']=actual['laya_executed']==c['expected_laya_executed']
            if trace['laya_executed']:
                assert r.analysis_complete and all(not v['truncated'] for v in trace['token_budget'].values())
            results.append(actual);dump(folder/(c['case_id']+'-trace.json'),trace)
            if i%16==0:print(json.dumps({'stage':'native_evaluation','set':name,'done':i+1,'total':len(rows)}),flush=True)
        write_rows(folder/'predictions.jsonl',results)
        semantic=[r for r in results if r['suite']=='semantic']
        summary['evaluations'][name]={'metrics':reports['metrics'](semantic),'questions':{q:sum(q not in r['question_errors'] for r in semantic) for q in QUESTIONS},
                                     'ambiguous':reports['metrics']([r for r in results if r['expected_status']=='UNKNOWN']),
                                     'languages':{lang:reports['metrics']([r for r in semantic if r['language']==lang]) for lang in ['en','ko']},
                                     'guards_passed':all(r['status_correct'] and r.get('guard_execution_correct',True) for r in results if r['suite']=='guard')}
        dump(OUT/'summary.json',summary)
    metadata=copy.deepcopy(backend.metadata);metadata.pop('token_budget',None);dump(OUT/'model-metadata.json',metadata)
    for p,h in lock['protected_evaluation'].items():assert sha(Path(p))==h
    for s in cases:assert sha(DATA/(s+'.jsonl'))==lock['splits'][s]['sha256']
    summary['all_80_native_matches_cache']=True;summary['completed_utc']=datetime.now(timezone.utc).isoformat();dump(OUT/'summary.json',summary)
    print(json.dumps({'completed':True,'selected_epoch':best_epoch,'selected':summary['selected'],'evaluations':summary['evaluations']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
