"""Matched legacy-only and RIPA-augmented scorer experiments on frozen features."""
import copy,json,math,os,runpy,time
from pathlib import Path
from datetime import datetime,timezone
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import torch
import torch.nn.functional as F
from safetensors.torch import load_file,save_file
from f1.laya.backend import LayaBackend
from f1.laya.finetuned import lock_hash,apply_scorer_delta
from f1.laya.questions_v2 import QUESTIONS,PROMPT_VERSION
from f1.laya.pipeline import assess

ROOT=Path('artifacts/laya/ripa-augmented-cpu-v1');DATA=Path('data/laya_ripa_domain_v1')

def choose_key(legacy,ripa):
    a,b=legacy['metrics'],ripa['metrics']
    return (min(a['attack_recall'],b['attack_recall']),-max(a['attack_clear_rate'],b['attack_clear_rate']),
            (a['benign_clear_rate']+b['benign_clear_rate'])/2,
            legacy['ambiguous_correct'],(legacy['question_accuracy']+ripa['question_accuracy'])/2,
            -(legacy['loss']+ripa['loss'])/2)

def main():
    helper=runpy.run_path('scripts/compare_laya_prompts.py');prior=runpy.run_path('scripts/train_laya_scorer.py');reports=runpy.run_path('scripts/evaluate_laya_gold.py')
    read,dump,sha=prior['read'],prior['dump'],prior['sha']
    audit=json.loads((ROOT/'cache-audit.json').read_text());assert audit['completed']
    lock=json.loads((DATA/'data-lock.json').read_text());assert sha(DATA/'data-lock.json')==audit['data_lock_sha256']
    for name,h in lock['sha256'].items():assert sha(DATA/(name+'.jsonl'))==h
    for p,h in lock['protected_files'].items():assert sha(Path(p))==h
    for name,info in audit['splits'].items():assert sha(ROOT/(name+'-features.safetensors'))==info['features_sha256']
    assert not (ROOT/'training-plan.json').exists(),'Keep completed or partial training separate; do not overwrite'
    plan={'created_utc':datetime.now(timezone.utc).isoformat(),'arms':['legacy_only','ripa_augmented'],'start':'same original pinned base scorer',
          'trainable':'native scorer only','trainable_parameters':592897,'updates':6000,'batch_question_rows':256,'sampling':'uniform random rows with replacement; identical algorithm and seed in both arms',
          'optimizer':'AdamW','weight_decay':0.0,'lr_start':0.001,'lr_end':0.00001,'schedule':'cosine decay','class_weights':'inverse frequency per question/class, derived from each arm training labels only',
          'loss':'mean of weighted per-row CE; global training weights have mean 1','seed':42,'cpu_threads':4,'eval_every':100,
          'selection':'maximize minimum ATTACK recall across legacy and RIPA validation; minimize maximum attack CLEAR; maximize mean normal CLEAR; legacy ambiguous accuracy; macro question accuracy; lower macro CE',
          'baseline_is_candidate':True,'prior_diagnostic_sets_used_for_selection':False,'human_reviewed':False,'deployment_ready':False}
    dump(ROOT/'training-plan.json',plan)
    torch.set_num_threads(4);torch.manual_seed(42)
    backend=LayaBackend(cpu_threads=4);backend._load();model=backend.agent.model;adapter=helper['CandidateBackend'](backend,QUESTIONS)
    assert lock_hash(backend.metadata['model_lock'])==audit['base_model_lock_sha256']
    for p in model.parameters():p.requires_grad_(False)
    original=copy.deepcopy(model.scorer.state_dict());base_version=backend.version
    frozen_hash=prior['tensors_hash']((n,p) for n,p in model.named_parameters() if not n.startswith('scorer.'))
    sets={}
    for name,path,cases_path in [
        ('legacy_train',Path('artifacts/laya/deepset-scorer-cpu-v1/train-features.safetensors'),Path('data/laya_deepset_domain_v1/train.jsonl')),
        ('legacy_validation',Path('artifacts/laya/deepset-scorer-cpu-v1/validation-features.safetensors'),DATA/'legacy_validation.jsonl'),
        ('ripa_train',ROOT/'ripa_train-features.safetensors',DATA/'ripa_train.jsonl'),
        ('ripa_validation',ROOT/'ripa_validation-features.safetensors',DATA/'ripa_validation.jsonl')]:
        data=load_file(str(path));cases=read(cases_path)
        assert torch.equal(data['labels'],torch.tensor([['YES','NO','UNKNOWN'].index(c['expected_questions'][q]) for c in cases for q in QUESTIONS]))
        sets[name]=(data['features'],data['labels'],cases)
    sets['augmented_train']=(torch.cat([sets['legacy_train'][0],sets['ripa_train'][0]]),torch.cat([sets['legacy_train'][1],sets['ripa_train'][1]]),sets['legacy_train'][2]+sets['ripa_train'][2])
    def evaluate(name,save_to=None):
        x,y,cases=sets[name];model.scorer.eval()
        with torch.no_grad():
            logits=torch.cat([model.scorer(x[i:i+256]).squeeze(-1) for i in range(0,len(x),256)])
            loss=float(F.cross_entropy(logits,y));answer=prior['predictions'](logits);statuses=prior['statuses'](answer)
        rows=[]
        for i,c in enumerate(cases):
            a=dict(zip(QUESTIONS,answer[i*4:i*4+4]))
            rows.append({**c,'answers':a,'actual_status':statuses[i],'status_correct':statuses[i]==c['expected_status'],
                         'question_errors':[q for q in QUESTIONS if a[q]['decision']!=c['expected_questions'][q]]})
        if save_to:(save_to/(name+'-predictions.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
        return {'loss':loss,'metrics':reports['metrics'](rows),'question_accuracy':sum(4-len(r['question_errors']) for r in rows)/(4*len(rows)),
                'ambiguous_correct':sum(r['expected_status']=='UNKNOWN' and r['status_correct'] for r in rows)}
    summary={'plan':plan,'arms':{},'baseline':{n:evaluate(n) for n in sets},'deployment_ready':False}
    states={}
    for arm,train_name in [('legacy_only','legacy_train'),('ripa_augmented','augmented_train')]:
        out=ROOT/arm;out.mkdir();model.scorer.load_state_dict(original);torch.manual_seed(42)
        for p in model.scorer.parameters():p.requires_grad_(True)
        x,y,cases=sets[train_name];weights=torch.zeros(4,3)
        for q in range(4):
            counts=torch.bincount(y[q::4],minlength=3)
            for label in range(3):
                if counts[label]:weights[q,label]=len(y)/(4*int((counts>0).sum())*int(counts[label]))
        sample_weights=weights[torch.arange(len(y))%4,y];assert abs(float(sample_weights.mean())-1)<1e-5
        dump(out/'class-weights.json',dict(zip(QUESTIONS,weights.tolist())))
        optimizer=torch.optim.AdamW(model.scorer.parameters(),lr=plan['lr_start'],weight_decay=0)
        v1=evaluate('legacy_validation');v2=evaluate('ripa_validation');best_key=choose_key(v1,v2);best=copy.deepcopy(original);best_step=0
        history=[{'step':0,'legacy_validation':v1,'ripa_validation':v2}];start=time.perf_counter()
        for step in range(1,plan['updates']+1):
            lr=plan['lr_end']+(plan['lr_start']-plan['lr_end'])*(1+math.cos(math.pi*(step-1)/(plan['updates']-1)))/2
            optimizer.param_groups[0]['lr']=lr;model.scorer.train();idx=torch.randint(len(y),(plan['batch_question_rows'],))
            logits=model.scorer(x[idx]).squeeze(-1);loss=(F.cross_entropy(logits,y[idx],reduction='none')*sample_weights[idx]).mean()
            optimizer.zero_grad();loss.backward();assert torch.isfinite(loss);torch.nn.utils.clip_grad_norm_(model.scorer.parameters(),1.0);optimizer.step()
            if step%plan['eval_every']==0:
                v1=evaluate('legacy_validation');v2=evaluate('ripa_validation');key=choose_key(v1,v2)
                if key>best_key:best_key=key;best=copy.deepcopy(model.scorer.state_dict());best_step=step
                history.append({'step':step,'lr':lr,'legacy_validation':v1,'ripa_validation':v2,'selected_step_so_far':best_step})
                if step%500==0:
                    dump(out/'history.json',history)
                    print(json.dumps({'arm':arm,'step':step,'seconds':time.perf_counter()-start,'selected_step':best_step,
                        'legacy':v1['metrics'],'ripa':v2['metrics']}),flush=True)
        save_file({k:v.contiguous() for k,v in model.scorer.state_dict().items()},str(out/'last-scorer.safetensors'))
        last={n:evaluate(n) for n in sets};states[arm]=best
        model.scorer.load_state_dict(best);model.eval()
        save_file({k:v.contiguous() for k,v in best.items()},str(out/'scorer.safetensors'))
        manifest={'base_model_lock_sha256':lock_hash(backend.metadata['model_lock']),'scorer_sha256':sha(out/'scorer.safetensors'),
            'trainable_module':'scorer','prompt_version':PROMPT_VERSION,'selected_step':best_step,'selection':plan['selection'],
            'training_data_lock_sha256':sha(DATA/'data-lock.json'),'deployment_ready':False}
        dump(out/'scorer-lock.json',manifest)
        selected={n:evaluate(n,out) for n in sets}
        summary['arms'][arm]={'training_cases':len(cases),'selected_step':best_step,'last':last,'selected':selected,'seconds':time.perf_counter()-start,
            'frozen_parameters_unchanged':frozen_hash==prior['tensors_hash']((n,p) for n,p in model.named_parameters() if not n.startswith('scorer.')),
            'selected_scorer_changed':any(not torch.equal(original[k],v) for k,v in model.scorer.state_dict().items()),'prior_diagnosis':{}}
        dump(ROOT/'training-summary.json',summary)
    # Both selections are now frozen, before accessing the prior diagnostic cases.
    model.scorer.load_state_dict(original);model.eval();capture=[];hook=model.scorer.register_forward_pre_hook(lambda m,a:capture.append(a[0].detach().clone()))
    diagnostic={}
    for name,path in [('prior-gold',Path('data/laya_gold_v1/cases.jsonl')),('prior-validation',Path('data/laya_validation_v2/cases.jsonl'))]:
        assert sha(path)==lock['protected_files'][str(path)]
        cases=read(path);base_rows=[];features=[];indices=[]
        for i,c in enumerate(cases):
            capture.clear();trace={};r=assess(*helper['inputs'](c),adapter,trace)
            if trace['laya_executed']:
                assert r.analysis_complete and len(capture)==1;indices.append(i);features.append(capture[0].clone())
                assert all(not v['truncated'] for v in trace['backend']['token_budget'].values())
            a=r.questions.model_dump(mode='json');base_rows.append({**c,'actual_status':str(r.status),'status_correct':str(r.status)==c['expected_status'],
                'answers':a,'question_errors':[q for q,g in (c['expected_questions'] or {}).items() if not a[q] or a[q]['decision']!=g],
                'laya_executed':trace['laya_executed'],'analysis_complete':r.analysis_complete})
            if i%16==0:print(json.dumps({'stage':'prior_diagnostic_cache','set':name,'done':i+1,'total':len(cases)}),flush=True)
        diagnostic[name]=(torch.cat(features),indices,base_rows)
        save_file({'features':torch.cat(features)},str(ROOT/(name+'-features.safetensors')))
        dump(ROOT/(name+'-feature-indices.json'),indices)
    hook.remove()
    for arm in states:
        out=ROOT/arm;backend.version=base_version;apply_scorer_delta(backend,out)
        for name,(features,indices,base_rows) in diagnostic.items():
            with torch.no_grad():answers=prior['predictions'](model.scorer(features).squeeze(-1))
            statuses=prior['statuses'](answers);rows=copy.deepcopy(base_rows)
            for j,i in enumerate(indices):
                c=rows[i];a=dict(zip(QUESTIONS,answers[j*4:j*4+4]));c.update(answers=a,actual_status=statuses[j],status_correct=statuses[j]==c['expected_status'],
                    question_errors=[q for q,g in (c['expected_questions'] or {}).items() if a[q]['decision']!=g])
            (out/(name+'-predictions.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
            sem=[r for r in rows if r['suite']=='semantic'];summary['arms'][arm]['prior_diagnosis'][name]={'metrics':reports['metrics'](sem),
                'languages':{lang:reports['metrics']([r for r in sem if r['language']==lang]) for lang in ['en','ko']},
                'guards_passed':all(r['status_correct'] and r['laya_executed']==r['expected_laya_executed'] for r in rows if r['suite']=='guard')}
        # Replay representative selected validation and diagnostic cases via full native pipeline.
        checks=[]
        for name in ['legacy_validation','ripa_validation','prior-gold','prior-validation']:
            rows=read(out/(name+'-predictions.jsonl'))
            for status in ['CLEAR','ATTACK']:
                c=next(r for r in rows if r['expected_status']==status and r['suite']=='semantic')
                trace={};r=assess(*helper['inputs'](c),adapter,trace);a=r.questions.model_dump(mode='json')
                assert str(r.status)==c['actual_status'] and all(a[q]['decision']==c['answers'][q]['decision'] for q in QUESTIONS)
                checks.append({'case_id':c['case_id'],'set':name,'native_matches_cache':True})
        dump(out/'native-replay-checks.json',checks);summary['arms'][arm]['native_replay_checks_passed']=len(checks)
    for p,h in lock['protected_files'].items():assert sha(Path(p))==h
    summary['completed_utc']=datetime.now(timezone.utc).isoformat();dump(ROOT/'summary.json',summary)
    print(json.dumps({'completed':True,'arms':{a:{'step':s['selected_step'],'legacy':s['selected']['legacy_validation']['metrics'],'ripa':s['selected']['ripa_validation']['metrics'],'prior':s['prior_diagnosis']} for a,s in summary['arms'].items()}},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
