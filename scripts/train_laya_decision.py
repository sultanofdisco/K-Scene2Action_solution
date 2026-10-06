"""Expand training to Laya's native two-layer head, keeping encoder frozen."""
import copy,hashlib,json,os,random,runpy,time
from datetime import datetime,timezone
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from f1.laya.backend import LayaBackend
from f1.laya.finetuned import lock_hash,apply_decision_delta
from f1.laya.pipeline import assess
from f1.laya.questions_v2 import QUESTIONS,PROMPT_VERSION

def read(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    out=Path('artifacts/laya/deepset-decision-cpu-v1');out.mkdir(parents=True,exist_ok=False)
    data=Path('data/laya_deepset_domain_v1');lock=json.loads((data/'data-lock.json').read_text(encoding='utf-8'))
    for split in ['train','validation']:assert sha(data/(split+'.jsonl'))==lock['splits'][split]['sha256']
    assert lock['questions']==QUESTIONS
    plan=dict(seed=42,epochs=10,learning_rate=0.00003,weight_decay=0.01,question_batch_size=8,cpu_threads=4,
              trainable=['head','scorer'],frozen=['encoder','type_emb','act_head'],
              selection='Minimum validation cross entropy with question/class weights derived from train only',
              reason='Scorer-only validation retained zero normal clearance; expand native decision layers',
              created_utc=datetime.now(timezone.utc).isoformat(),data_lock_sha256=sha(data/'data-lock.json'))
    dump(out/'plan.json',plan);dump(out/'data-lock.json',lock)
    torch.manual_seed(42);random.seed(42);torch.set_num_threads(4)
    helpers=runpy.run_path('scripts/compare_laya_prompts.py');reports=runpy.run_path('scripts/evaluate_laya_gold.py');prior=runpy.run_path('scripts/train_laya_scorer.py')
    backend=LayaBackend(cpu_threads=4);backend._load();adapter=helpers['CandidateBackend'](backend,QUESTIONS);model=backend.agent.model
    for p in model.parameters():p.requires_grad_(False)
    frozen_hash=prior['tensors_hash']((n,p) for n,p in model.named_parameters() if not n.startswith(('head.','scorer.')))
    state_before=prior['tensors_hash']((n,p) for n,p in model.named_parameters() if n.startswith(('head.','scorer.')))
    capture={}
    def pre(module,args):
        capture['mask']=args[1].detach().clone();capture['markers']=args[2].detach().clone();capture['qtype']=args[4].detach().clone()
    def enc(module,args,output):capture['h']=output.last_hidden_state.detach().clone()
    h1=model.register_forward_pre_hook(pre);h2=model.encoder.register_forward_hook(enc)
    datasets={};base_metrics={};cache_start=time.perf_counter()
    for split in ['train','validation']:
        cases=read(data/(split+'.jsonl'));items=[];base_rows=[]
        for i,c in enumerate(cases):
            capture.clear();trace={};result=assess(*helpers['inputs'](c),adapter,trace)
            assert trace['laya_executed'] and result.analysis_complete
            for q in range(4):
                length=int(capture['mask'][q].sum())
                items.append({'h':capture['h'][q,:length].clone(),'markers':capture['markers'][q].clone(),
                    'qtype':int(capture['qtype'][q]),'q':q,'label':['YES','NO','UNKNOWN'].index(c['expected_questions'][list(QUESTIONS)[q]])})
            base_rows.append({**c,'actual_status':str(result.status),'status_correct':str(result.status)==c['expected_status']})
            if i%8==0:print(json.dumps({'stage':'encoder_cache','split':split,'done':i+1,'total':len(cases)}),flush=True)
        datasets[split]=(items,cases);base_metrics[split]=reports['metrics'](base_rows)
        save_file({f'h{i}':x['h'].contiguous() for i,x in enumerate(items)},str(out/(split+'-encoder-features.safetensors')))
        dump(out/(split+'-encoder-index.json'),[{k:v.tolist() if isinstance(v,torch.Tensor) else v for k,v in item.items() if k!='h'} for item in items])
    h1.remove();h2.remove();capture.clear()
    def forward(items):
        lengths=[len(x['h']) for x in items];maximum=max(lengths)
        h=torch.stack([F.pad(x['h'],(0,0,0,maximum-len(x['h']))) for x in items])
        mask=torch.arange(maximum)[None,:]<torch.tensor(lengths)[:,None]
        h=h+model.type_emb(torch.tensor([x['qtype'] for x in items]))[:,None,:]
        for layer in model.head.layers:h=layer(h,src_key_padding_mask=~mask)
        markers=torch.stack([x['markers'] for x in items])
        m=torch.gather(h,1,markers[:,:,None].expand(-1,-1,h.size(-1)))
        return model.scorer(m).squeeze(-1)
    # Verify cached execution against a native complete prediction before updating weights.
    with torch.no_grad():
        sample=datasets['train'][0][:4];cached=prior['predictions'](forward(sample))
        t={};r=assess(*helpers['inputs'](datasets['train'][1][0]),adapter,t)
        for i,q in enumerate(QUESTIONS):
            assert cached[i]['model_answer']==getattr(r.questions,q).model_answer
    for name,p in model.named_parameters():p.requires_grad_(name.startswith(('head.','scorer.')))
    params=[p for p in model.parameters() if p.requires_grad];trainable=sum(p.numel() for p in params)
    weights=torch.zeros(4,3);train=datasets['train'][0]
    for q in range(4):
        counts=torch.bincount(torch.tensor([x['label'] for x in train if x['q']==q]),minlength=3)
        for c in range(3):
            if counts[c]:weights[q,c]=len(train)/(4*int((counts>0).sum())*int(counts[c]))
    def evaluate(split):
        items,cases=datasets[split];logits=[]
        model.head.eval();model.scorer.eval()
        with torch.no_grad():
            for i in range(0,len(items),8):logits.append(forward(items[i:i+8]))
            logits=torch.cat(logits);labels=torch.tensor([x['label'] for x in items]);w=torch.tensor([weights[x['q'],x['label']] for x in items])
            assert (w>0).all()
            loss=float((F.cross_entropy(logits,labels,reduction='none')*w).sum()/w.sum())
            answers=prior['predictions'](logits);status=prior['statuses'](answers)
        rows=[]
        for i,c in enumerate(cases):
            a=dict(zip(QUESTIONS,answers[i*4:i*4+4]))
            rows.append({**c,'answers':a,'actual_status':status[i],'status_correct':status[i]==c['expected_status'],
                         'question_errors':[q for q in QUESTIONS if a[q]['decision']!=c['expected_questions'][q]]})
        return loss,rows
    initial_val_loss,_=evaluate('validation')
    optimizer=torch.optim.AdamW(params,lr=plan['learning_rate'],weight_decay=plan['weight_decay'])
    best_loss=float('inf');best_epoch=0;best_state=None;history=[]
    train_started=datetime.now(timezone.utc).isoformat()
    for epoch in range(1,plan['epochs']+1):
        t=time.perf_counter();model.head.train();model.scorer.train();indices=torch.randperm(len(train)).tolist()
        for step,start in enumerate(range(0,len(train),8)):
            items=[train[j] for j in indices[start:start+8]]
            logits=forward(items);labels=torch.tensor([x['label'] for x in items]);w=torch.tensor([weights[x['q'],x['label']] for x in items])
            loss=(F.cross_entropy(logits,labels,reduction='none')*w).sum()/w.sum()
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(params,1.0);optimizer.step()
            if step%10==0:print(json.dumps({'stage':'decision_training','epoch':epoch,'step':step+1,'steps':(len(train)+7)//8}),flush=True)
        train_loss,train_rows=evaluate('train');val_loss,val_rows=evaluate('validation')
        row=dict(epoch=epoch,seconds=time.perf_counter()-t,train_loss=train_loss,validation_loss=val_loss,train_metrics=reports['metrics'](train_rows),validation_metrics=reports['metrics'](val_rows));history.append(row)
        if val_loss<best_loss:
            best_loss=val_loss;best_epoch=epoch;best_state={n:p.detach().clone() for n,p in model.state_dict().items() if n.startswith(('head.','scorer.'))}
            save_file(best_state,str(out/'decision.safetensors'))
        dump(out/'history.json',history)
        print(json.dumps({'stage':'epoch_complete','epoch':epoch,'seconds':row['seconds'],'train_loss':train_loss,'validation_loss':val_loss,'validation':row['validation_metrics']}),flush=True)
    for name in ['head','scorer']:getattr(model,name).load_state_dict({k[len(name)+1:]:v for k,v in best_state.items() if k.startswith(name+'.')},strict=True)
    model.eval()
    assert prior['tensors_hash']((n,p) for n,p in model.named_parameters() if not n.startswith(('head.','scorer.')))==frozen_hash
    assert prior['tensors_hash']((n,p) for n,p in model.named_parameters() if n.startswith(('head.','scorer.')))!=state_before
    manifest=dict(base_model_lock_sha256=lock_hash(backend.metadata['model_lock']),base_revision=backend.metadata['model_lock']['revision'],weights_sha256=sha(out/'decision.safetensors'),trainable_modules=['head','scorer'],trainable_parameters=trainable,prompt_version=PROMPT_VERSION,selected_epoch=best_epoch,selection='training validation loss only',data_lock_sha256=sha(data/'data-lock.json'),frozen_parameters_sha256=frozen_hash,deployment_ready=False)
    dump(out/'decision-lock.json',manifest);apply_decision_delta(backend,out)
    summary=dict(training_started_utc=train_started,training_finished_utc=datetime.now(timezone.utc).isoformat(),plan=plan,selected_epoch=best_epoch,initial_validation_loss=initial_val_loss,best_validation_loss=best_loss,trainable_parameters=trainable,frozen_parameters_unchanged=True,native_decision_weights_changed=True,checkpoint_roundtrip_exact=all(torch.equal(v,model.state_dict()[k]) for k,v in best_state.items()),deployment_ready=False,human_reviewed=False,training_results={},evaluations={})
    for split in ['train','validation']:
        _,rows=evaluate(split);summary['training_results'][split]={'before':base_metrics[split],'after':reports['metrics'](rows)}
        (out/(split+'-predictions.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    dump(out/'training-summary.json',summary)
    for name,path,base_path in [('prior-gold',Path('data/laya_gold_v1/cases.jsonl'),Path('artifacts/laya/prompt-comparison-cpu-v2/development-v2/predictions.jsonl')),('prior-validation',Path('data/laya_validation_v2/cases.jsonl'),Path('artifacts/laya/prompt-comparison-cpu-v2/validation-v2/predictions.jsonl'))]:
        assert sha(path)==lock['protected_evaluation'][str(path)]
        folder=out/name;folder.mkdir();rows=[];cases=read(path)
        for i,c in enumerate(cases):
            trace={};t=time.perf_counter();result=assess(*helpers['inputs'](c),adapter,trace)
            trace['prompt_version']=PROMPT_VERSION;meta=trace.pop('backend',{});trace['token_budget']=meta.get('token_budget');trace['backend_ref']='../model-metadata.json'
            a=result.questions.model_dump(mode='json');row={**c,'actual_status':str(result.status),'status_correct':str(result.status)==c['expected_status'],'answers':a,'question_errors':[q for q,gold in (c['expected_questions'] or {}).items() if not a[q] or a[q]['decision']!=gold],'reason_codes':list(result.reason_codes),'seconds':time.perf_counter()-t,'laya_executed':trace['laya_executed'],'analysis_complete':result.analysis_complete}
            if 'expected_laya_executed' in c:row['guard_execution_correct']=row['laya_executed']==c['expected_laya_executed']
            rows.append(row);dump(folder/(c['case_id']+'-trace.json'),trace);dump(folder/(c['case_id']+'-assessment.json'),result.model_dump(mode='json'))
            with (folder/'predictions.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
            if i%8==0:print(json.dumps({'stage':'evaluation','set':name,'done':i+1,'total':len(cases)}),flush=True)
        sem=[r for r in rows if r['suite']=='semantic'];comparison=helpers['compare'](read(base_path),rows,reports['metrics'])
        comparison['before_finetuning']=comparison.pop('v1');comparison['after_finetuning']=comparison.pop('v2')
        comparison['questions']={q:{'correct':sum(q not in r['question_errors'] for r in sem),'n':len(sem)} for q in QUESTIONS}
        comparison['suites']={s:reports['metrics']([r for r in rows if r['suite']==s]) for s in ['semantic','ambiguous','guard']}
        comparison['languages']={lang:reports['metrics']([r for r in sem if r['language']==lang]) for lang in ['en','ko']}
        comparison['guard_execution_checks_passed']=all(r.get('guard_execution_correct',True) for r in rows)
        summary['evaluations'][name]=comparison;reports['render_review'](folder,rows);dump(out/'summary.json',summary)
    metadata=copy.deepcopy(backend.metadata);metadata.pop('token_budget',None);dump(out/'model-metadata.json',metadata)
    summary['completed_utc']=datetime.now(timezone.utc).isoformat();dump(out/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
