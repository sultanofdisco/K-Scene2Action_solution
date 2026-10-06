"""Train Laya's existing scorer on frozen native marker features, then evaluate."""
import copy,hashlib,json,os,random,runpy,time
from datetime import datetime,timezone
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file,load_file
from f1.laya.backend import LayaBackend
from f1.laya.finetuned import apply_scorer_delta,lock_hash
from f1.laya.pipeline import assess,parse_answer
from f1.laya.questions_v2 import QUESTIONS,PROMPT_VERSION

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def read(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
def tensors_hash(items):
    h=hashlib.sha256()
    for name,t in items:
        h.update(name.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def predictions(logits):
    # Match the SDK's four-decimal probabilities and existing adapter mapping.
    p=logits.softmax(-1).detach().numpy();out=[]
    for row in p:
        probs={key:round(float(v),4) for key,v in zip('ABC',row)}
        answer=parse_answer({'type':'choice','choice':'ABC'[int(row.argmax())],'probabilities':probs})
        out.append(answer.model_dump(mode='json'))
    return out

def statuses(answers):
    result=[]
    for i in range(0,len(answers),4):
        a=[x['decision'] for x in answers[i:i+4]]
        result.append('ATTACK' if 'YES' in a[:3] else 'CLEAR' if a==['NO','NO','NO','YES'] else 'UNKNOWN')
    return result

def main():
    out=Path('artifacts/laya/deepset-scorer-cpu-v1');out.mkdir(parents=True,exist_ok=False)
    data=Path('data/laya_deepset_domain_v1')
    lock=json.loads((data/'data-lock.json').read_text(encoding='utf-8'))
    assert lock['questions']==QUESTIONS
    for split in ['train','validation']:assert sha(data/(split+'.jsonl'))==lock['splits'][split]['sha256']
    dump(out/'data-lock.json',lock)
    plan=lock['training_plan']
    random.seed(plan['seed']);np.random.seed(plan['seed']);torch.manual_seed(plan['seed']);torch.set_num_threads(plan['cpu_threads'])
    helpers=runpy.run_path('scripts/compare_laya_prompts.py')
    reporting=runpy.run_path('scripts/evaluate_laya_gold.py')
    backend=LayaBackend(cpu_threads=plan['cpu_threads']);backend._load()
    adapter=helpers['CandidateBackend'](backend,QUESTIONS)
    model=backend.agent.model
    assert backend.agent.temperature_by_options.get('choice:3-5',backend.agent.temperature[0])==1.0
    for p in model.parameters():p.requires_grad_(False)
    frozen_hash=tensors_hash((n,p) for n,p in model.named_parameters() if not n.startswith('scorer.'))
    original_scorer={k:v.detach().clone() for k,v in model.scorer.state_dict().items()}
    original_scorer_hash=tensors_hash(original_scorer.items())
    buffers=[]
    def capture(module,args):buffers.append(args[0].detach().cpu().clone())
    hook=model.scorer.register_forward_pre_hook(capture)
    cache={};baseline={};cache_start=time.perf_counter()
    for split in ['train','validation']:
        cases=read(data/(split+'.jsonl'));features=[];rows=[]
        for index,case in enumerate(cases):
            buffers.clear();trace={}
            result=assess(*helpers['inputs'](case),adapter,trace)
            assert trace['laya_executed'] and result.analysis_complete
            assert len(buffers)==1 and buffers[0].shape[:2]==(4,3)
            feature=buffers[0].detach().clone() # outside inference_mode
            features.append(feature)
            answer=result.questions.model_dump(mode='json')
            rows.append({**case,'actual_status':str(result.status),'status_correct':str(result.status)==case['expected_status'],'answers':answer,
                         'question_errors':[q for q in QUESTIONS if answer[q]['decision']!=case['expected_questions'][q]],'laya_executed':True})
            if index%8==0:print(json.dumps({'stage':'cache','split':split,'done':index+1,'total':len(cases)}),flush=True)
        x=torch.cat(features).clone();y=torch.tensor([['YES','NO','UNKNOWN'].index(c['expected_questions'][q]) for c in cases for q in QUESTIONS])
        with torch.no_grad():
            reconstructed=predictions(model.scorer(x).squeeze(-1))
        for i,c in enumerate(rows):
            for j,q in enumerate(QUESTIONS):
                assert reconstructed[i*4+j]['model_answer']==c['answers'][q]['model_answer']
                assert max(abs(reconstructed[i*4+j]['probabilities'][k]-c['answers'][q]['probabilities'][k]) for k in ['yes','no','unknown'])<0.001
        cache[split]=(x,y,cases);baseline[split]=rows
        save_file({'features':x,'labels':y},str(out/(split+'-features.safetensors')))
    hook.remove();buffers.clear()
    cache_seconds=time.perf_counter()-cache_start
    scorer=model.scorer
    for p in scorer.parameters():p.requires_grad_(True)
    trainable_count=sum(p.numel() for p in scorer.parameters())
    total_count=sum(p.numel() for p in model.parameters())
    x,y,cases=cache['train'];vx,vy,vcases=cache['validation']
    weights=torch.zeros(4,3)
    for q in range(4):
        counts=torch.bincount(y[q::4],minlength=3)
        for label in range(3):
            if counts[label]:weights[q,label]=len(y)/(4*int((counts>0).sum())*int(counts[label]))
    train_weights=weights[torch.arange(len(y))%4,y]
    val_weights=weights[torch.arange(len(vy))%4,vy]
    assert (val_weights>0).all()
    optimizer=torch.optim.AdamW(scorer.parameters(),lr=plan['learning_rate'],weight_decay=plan['weight_decay'])
    def evaluate(features,labels,weight):
        with torch.no_grad():
            logits=scorer(features).squeeze(-1)
            loss=float((F.cross_entropy(logits,labels,reduction='none')*weight).sum()/weight.sum())
            answer=predictions(logits)
        return loss,answer
    base_val_loss,_=evaluate(vx,vy,val_weights)
    best_loss=float('inf');best_epoch=None;best_state=None;history=[]
    started=datetime.now(timezone.utc).isoformat()
    for epoch in range(1,plan['epochs']+1):
        scorer.train();order=torch.randperm(len(y))
        for start in range(0,len(y),plan['batch_size']):
            idx=order[start:start+plan['batch_size']]
            logits=scorer(x[idx]).squeeze(-1)
            loss=(F.cross_entropy(logits,y[idx],reduction='none')*train_weights[idx]).sum()/train_weights[idx].sum()
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(scorer.parameters(),1.0);optimizer.step()
        scorer.eval()
        train_loss,train_answers=evaluate(x,y,train_weights)
        val_loss,val_answers=evaluate(vx,vy,val_weights)
        row=dict(epoch=epoch,train_loss=train_loss,validation_loss=val_loss,
                 train_status_accuracy=sum(a==c['expected_status'] for a,c in zip(statuses(train_answers),cases))/len(cases),
                 validation_status_accuracy=sum(a==c['expected_status'] for a,c in zip(statuses(val_answers),vcases))/len(vcases))
        history.append(row)
        if val_loss<best_loss:
            best_loss=val_loss;best_epoch=epoch;best_state={k:v.detach().clone() for k,v in scorer.state_dict().items()}
        dump(out/'history.json',history);print(json.dumps({'stage':'train',**row}),flush=True)
    scorer.load_state_dict(best_state);model.eval()
    assert tensors_hash((n,p) for n,p in model.named_parameters() if not n.startswith('scorer.'))==frozen_hash
    assert tensors_hash(scorer.state_dict().items())!=original_scorer_hash
    save_file({k:v.contiguous() for k,v in best_state.items()},str(out/'scorer.safetensors'))
    ckpt=dict(base_model_lock_sha256=lock_hash(backend.metadata['model_lock']),base_repo=backend.metadata['model_lock']['repo'],base_revision=backend.metadata['model_lock']['revision'],scorer_sha256=sha(out/'scorer.safetensors'),trainable_module='scorer',trainable_parameters=trainable_count,total_parameters=total_count,prompt_version=PROMPT_VERSION,selected_epoch=best_epoch,selection_uses='training validation only',training_data_lock_sha256=sha(data/'data-lock.json'),deployment_ready=False,frozen_parameters_sha256=frozen_hash)
    dump(out/'scorer-lock.json',ckpt)
    apply_scorer_delta(backend,out)
    for key,value in load_file(str(out/'scorer.safetensors')).items():assert torch.equal(value,scorer.state_dict()[key])
    summary=dict(started_utc=started,completed_training_utc=datetime.now(timezone.utc).isoformat(),cache_seconds=cache_seconds,
        training_plan=plan,selected_epoch=best_epoch,base_validation_loss=base_val_loss,best_validation_loss=best_loss,
        trainable_parameters=trainable_count,total_parameters=total_count,frozen_parameters_unchanged=True,
        native_scorer_changed=True,checkpoint_roundtrip_exact=True,deployment_ready=False,human_reviewed=False,training_results={},evaluations={})
    for split,(features,labels,c) in cache.items():
        _,answer=evaluate(features,labels,train_weights if split=='train' else val_weights)
        status=statuses(answer)
        rows=[]
        for i,case in enumerate(c):
            answers=dict(zip(QUESTIONS,answer[i*4:i*4+4]))
            rows.append({**case,'actual_status':status[i],'status_correct':status[i]==case['expected_status'],
                         'question_errors':[q for q in QUESTIONS if answers[q]['decision']!=case['expected_questions'][q]],'answers':answers})
        summary['training_results'][split]={'before':reporting['metrics'](baseline[split]),'after':reporting['metrics'](rows)}
        (out/(split+'-predictions.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    dump(out/'training-summary.json',summary)
    # Checkpoint is now frozen. These diagnosis sets never select an epoch or train weights.
    for name,path,prior in [
        ('prior-gold',Path('data/laya_gold_v1/cases.jsonl'),Path('artifacts/laya/prompt-comparison-cpu-v2/development-v2/predictions.jsonl')),
        ('prior-validation',Path('data/laya_validation_v2/cases.jsonl'),Path('artifacts/laya/prompt-comparison-cpu-v2/validation-v2/predictions.jsonl')),
    ]:
        assert sha(path)==lock['protected_evaluation'][str(path)]
        folder=out/name;folder.mkdir();results=[]
        prior_rows=read(prior)
        for index,case in enumerate(read(path)):
            trace={};t=time.perf_counter()
            result=assess(*helpers['inputs'](case),adapter,trace)
            trace['prompt_version']=PROMPT_VERSION
            meta=trace.pop('backend',{});trace['token_budget']=meta.get('token_budget');trace['backend_ref']='../model-metadata.json'
            answers=result.questions.model_dump(mode='json')
            row={**case,'actual_status':str(result.status),'status_correct':str(result.status)==case['expected_status'],
                 'answers':answers,'question_errors':[q for q,gold in (case['expected_questions'] or {}).items() if not answers[q] or answers[q]['decision']!=gold],
                 'reason_codes':list(result.reason_codes),'seconds':time.perf_counter()-t,'laya_executed':trace['laya_executed'],'analysis_complete':result.analysis_complete}
            if 'expected_laya_executed' in case:row['guard_execution_correct']=row['laya_executed']==case['expected_laya_executed']
            results.append(row)
            dump(folder/(case['case_id']+'-assessment.json'),result.model_dump(mode='json'))
            dump(folder/(case['case_id']+'-trace.json'),trace)
            with (folder/'predictions.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
            if index%8==0:print(json.dumps({'stage':'evaluate','set':name,'done':index+1,'total':len(prior_rows)}),flush=True)
        sem=[r for r in results if r['suite']=='semantic']
        comparison=helpers['compare'](prior_rows,results,reporting['metrics'])
        comparison['before_finetuning']=comparison.pop('v1');comparison['after_finetuning']=comparison.pop('v2')
        comparison['questions']={q:{'correct':sum(q not in r['question_errors'] for r in sem),'n':len(sem)} for q in QUESTIONS}
        comparison['suites']={s:reporting['metrics']([r for r in results if r['suite']==s]) for s in ['semantic','ambiguous','guard']}
        comparison['languages']={lang:reporting['metrics']([r for r in sem if r['language']==lang]) for lang in ['en','ko']}
        comparison['guard_execution_checks_passed']=all(r.get('guard_execution_correct',True) for r in results)
        summary['evaluations'][name]=comparison
        reporting['render_review'](folder,results)
        dump(out/'summary.json',summary)
    metadata=copy.deepcopy(backend.metadata);metadata.pop('token_budget',None);dump(out/'model-metadata.json',metadata)
    for path,expected in lock['protected_evaluation'].items():assert sha(Path(path))==expected
    summary['completed_utc']=datetime.now(timezone.utc).isoformat();dump(out/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
