"""Sixteen-case memorization check. Results are training fit, never test accuracy."""
import argparse, copy, hashlib, json, os, runpy, time
from pathlib import Path
from datetime import datetime, timezone
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from f1.laya.backend import LayaBackend
from f1.laya.pipeline import assess
from f1.laya.questions_v2 import QUESTIONS, PROMPT_VERSION
from f1.laya.finetuned import lock_hash

OUT=Path('artifacts/laya/memorization-16-cpu-v1')
GROUPS=['destination','object','system_claim','root_claim','shatter','throw','safety_rule','authority_goal']

def read(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default=str(OUT))
    parser.add_argument('--epochs',type=int,default=500)
    parser.add_argument('--learning-rate',type=float,default=0.001)
    parser.add_argument('--warm-start')
    parser.add_argument('--arms',nargs='+',choices=['unweighted','question_class_balanced'],default=['unweighted','question_class_balanced'])
    args=parser.parse_args()
    OUT=Path(args.output)
    OUT.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(42);torch.set_num_threads(4)
    source=Path('data/laya_deepset_domain_v1/train.jsonl')
    all_cases=read(source)
    ids=['domain-'+g+'-'+kind for g in GROUPS for kind in ['normal','attack']]
    indices=[next(i for i,c in enumerate(all_cases) if c['case_id']==cid) for cid in ids]
    cases=[all_cases[i] for i in indices]
    assert len(cases)==16 and sum(c['expected_status']=='CLEAR' for c in cases)==8
    plan={'created_utc':datetime.now(timezone.utc).isoformat(),'purpose':'Same-case memorization diagnostic, not generalization evaluation','seed':42,'cpu_threads':4,'source_sha256':sha(source),'case_ids':ids,'trainable':'native scorer only','frozen':['encoder','type_emb','head','act_head'],'optimizer':'AdamW','learning_rate':0.001,'weight_decay':0.0,'batch_size':64,'max_epochs':500,'eval_interval':10,'early_stop':'64/64 question labels AND 16/16 statuses for three consecutive checks','arms':['unweighted','question_class_balanced'],'prompt_version':PROMPT_VERSION,'human_reviewed':False,'deployment_ready':False}
    plan['max_epochs']=args.epochs
    plan['learning_rate']=args.learning_rate
    plan['arms']=args.arms
    if args.warm_start:
        plan['warm_start']={'path':args.warm_start,'sha256':sha(Path(args.warm_start)),'optimizer_state':'reset identically for paired continuation experiments'}
    dump(OUT/'plan.json',plan)
    (OUT/'cases.jsonl').write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in cases),encoding='utf-8')
    helpers=runpy.run_path('scripts/compare_laya_prompts.py');prior=runpy.run_path('scripts/train_laya_scorer.py')
    backend=LayaBackend(cpu_threads=4);backend._load();adapter=helpers['CandidateBackend'](backend,QUESTIONS);model=backend.agent.model
    original=copy.deepcopy(model.scorer.state_dict())
    for p in model.parameters():p.requires_grad_(False)
    frozen=prior['tensors_hash']((n,p) for n,p in model.named_parameters() if not n.startswith('scorer.'))
    cached=load_file('artifacts/laya/deepset-scorer-cpu-v1/train-features.safetensors')
    rows=torch.tensor([i*4+q for i in indices for q in range(4)])
    x=cached['features'][rows].clone();y=cached['labels'][rows].clone()
    expected=torch.tensor([['YES','NO','UNKNOWN'].index(c['expected_questions'][q]) for c in cases for q in QUESTIONS])
    assert torch.equal(y,expected)
    # Re-extract each selected case to detect stale caches or case/question ordering mistakes.
    capture=[]
    hook=model.scorer.register_forward_pre_hook(lambda m,a:capture.append(a[0].detach().clone()))
    baseline=[];max_difference=0.0
    for i,c in enumerate(cases):
        capture.clear();trace={};result=assess(*helpers['inputs'](c),adapter,trace)
        assert result.analysis_complete and trace['laya_executed'] and len(capture)==1
        difference=float((capture[0]-x[i*4:i*4+4]).abs().max());max_difference=max(max_difference,difference)
        assert difference<1e-5,(c['case_id'],difference)
        baseline.append({**c,'actual_status':str(result.status),'answers':result.questions.model_dump(mode='json')})
    hook.remove()
    if args.warm_start:
        original=load_file(args.warm_start)
    dump(OUT/'input-audit.json',{'all_16_cache_matches_native':True,'max_abs_difference':max_difference,'label_order_verified':True,'choice_mapping':{'YES':0,'NO':1,'UNKNOWN':2},'labels_not_in_state':True,'cases_sha256':sha(OUT/'cases.jsonl'),'cache_sha256':sha(Path('artifacts/laya/deepset-scorer-cpu-v1/train-features.safetensors'))})
    weights=torch.zeros(4,3)
    for q in range(4):
        counts=torch.bincount(y[q::4],minlength=3)
        for label in range(3):
            if counts[label]:weights[q,label]=len(y)/(4*int((counts>0).sum())*int(counts[label]))
    summary={'plan':plan,'arms':{},'baseline_status_correct':sum(r['actual_status']==r['expected_status'] for r in baseline),'baseline_question_correct':sum(r['answers'][q]['decision']==r['expected_questions'][q] for r in baseline for q in QUESTIONS)}
    for arm in plan['arms']:
        folder=OUT/arm;folder.mkdir();model.scorer.load_state_dict(original);torch.manual_seed(42)
        for p in model.scorer.parameters():p.requires_grad_(True)
        w=torch.ones(len(y)) if arm=='unweighted' else weights[torch.arange(len(y))%4,y]
        opt=torch.optim.AdamW(model.scorer.parameters(),lr=plan['learning_rate'],weight_decay=0)
        def evaluate():
            model.scorer.eval()
            with torch.no_grad():
                logits=model.scorer(x).squeeze(-1);answers=prior['predictions'](logits);statuses=prior['statuses'](answers)
                loss=float((F.cross_entropy(logits,y,reduction='none')*w).sum()/w.sum())
            return {'loss':loss,'question_correct':int((logits.argmax(-1)==y).sum()),'status_correct':sum(a==c['expected_status'] for a,c in zip(statuses,cases))},answers,statuses
        initial,_,_=evaluate();history=[{'epoch':0,**initial}];stable=0;best=None;best_key=None;started=time.perf_counter();gradient_audit={}
        for epoch in range(1,plan['max_epochs']+1):
            model.scorer.train();logits=model.scorer(x).squeeze(-1);loss=(F.cross_entropy(logits,y,reduction='none')*w).sum()/w.sum()
            opt.zero_grad();loss.backward()
            if epoch==1:
                gradient_audit={n:float(p.grad.norm()) for n,p in model.scorer.named_parameters()}
                assert all(torch.isfinite(p.grad).all() for p in model.scorer.parameters()) and sum(gradient_audit.values())>0
            torch.nn.utils.clip_grad_norm_(model.scorer.parameters(),1.0);opt.step()
            if epoch%10==0:
                metrics,answers,statuses=evaluate();history.append({'epoch':epoch,**metrics})
                key=(metrics['question_correct'],metrics['status_correct'],-metrics['loss'])
                if best_key is None or key>best_key:
                    best_key=key;best=copy.deepcopy(model.scorer.state_dict());best_epoch=epoch
                stable=stable+1 if metrics['question_correct']==64 and metrics['status_correct']==16 else 0
                dump(folder/'history.json',history)
                if epoch%50==0 or stable:print(json.dumps({'arm':arm,'epoch':epoch,**metrics,'seconds':time.perf_counter()-started}),flush=True)
                if stable>=3:break
        model.scorer.load_state_dict(best);model.eval();final,answers,statuses=evaluate()
        save_file({k:v.contiguous() for k,v in best.items()},str(folder/'scorer.safetensors'))
        manifest={'base_model_lock_sha256':lock_hash(backend.metadata['model_lock']),'scorer_sha256':sha(folder/'scorer.safetensors'),'trainable_module':'scorer','prompt_version':PROMPT_VERSION,'selected_epoch':best_epoch,'deployment_ready':False,'memorization_only':True}
        dump(folder/'scorer-lock.json',manifest)
        # Restore serialized weights, then run the complete native inference on all 16 cases.
        model.scorer.load_state_dict(load_file(str(folder/'scorer.safetensors')),strict=True)
        actual=[]
        for i,c in enumerate(cases):
            trace={};r=assess(*helpers['inputs'](c),adapter,trace)
            a=r.questions.model_dump(mode='json')
            assert str(r.status)==statuses[i]
            assert all(a[q]['decision']==answers[i*4+j]['decision'] for j,q in enumerate(QUESTIONS))
            trace['prompt_version']=PROMPT_VERSION;trace['memorization_arm']=arm
            trace.pop('backend',None)
            dump(folder/(c['case_id']+'-trace.json'),trace)
            actual.append({**c,'actual_status':str(r.status),'status_correct':str(r.status)==c['expected_status'],'answers':a,'question_errors':[q for q in QUESTIONS if a[q]['decision']!=c['expected_questions'][q]]})
        (folder/'predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in actual),encoding='utf-8')
        changed=any(not torch.equal(original[k],v) for k,v in model.scorer.state_dict().items())
        assert changed and frozen==prior['tensors_hash']((n,p) for n,p in model.named_parameters() if not n.startswith('scorer.'))
        summary['arms'][arm]={'initial':initial,'final':final,'selected_epoch':best_epoch,'last_epoch':epoch,'seconds':time.perf_counter()-started,'gradient_norms_first_step':gradient_audit,'scorer_weights_changed':changed,'frozen_weights_unchanged':True,'all_16_native_roundtrip_matches':True,'perfect_fit':final['question_correct']==64 and final['status_correct']==16,'normal_clear_count':sum(r['expected_status']=='CLEAR' and r['actual_status']=='CLEAR' for r in actual),'attack_detect_count':sum(r['expected_status']=='ATTACK' and r['actual_status']=='ATTACK' for r in actual)}
        dump(OUT/'summary.json',summary)
        print(json.dumps({'arm_complete':arm,**summary['arms'][arm]}),flush=True)
    summary['completed_utc']=datetime.now(timezone.utc).isoformat();dump(OUT/'summary.json',summary)

if __name__=='__main__':main()
