"""RIPA native decision-head training with upstream-inspired RLCD + CE."""
import os,json,runpy,time,math,copy,random,gc
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file,load_file
from laya.common import proper_reward
from f1.laya.backend import LayaBackend
from f1.laya.questions_v2 import QUESTIONS,PROMPT_VERSION
from f1.laya.pipeline import assess
from f1.laya.finetuned import lock_hash,apply_decision_delta

ROOT=Path('artifacts/laya/ripa-decision-cpu-v1');DATA=Path('data/laya_ripa_domain_v1')

def rlcd_loss(logits,labels,qtypes,sigma):
    target=F.one_hot(labels,3).float();mask=torch.ones_like(logits,dtype=torch.bool)
    eps=torch.randn((4,)+logits.shape,device=logits.device)*sigma
    eps=eps-eps.mean(-1,keepdim=True)
    noisy=logits.detach().unsqueeze(0)+eps
    with torch.no_grad():
        reward=proper_reward(noisy.softmax(-1),target.unsqueeze(0),qtypes,mask,w_sph=.75,w_rps=1.)
        advantage=reward-reward.mean(0,keepdim=True);advantage=advantage/(advantage.std()+1e-6)
    logp=-((noisy-logits.unsqueeze(0))**2).sum(-1)/(2*sigma**2)
    rl=-(advantage*logp).mean();ce=F.cross_entropy(logits,labels)
    return rl+ce,rl,ce

def decision_forward(model,items):
    maxlen=max(len(i['h']) for i in items);lengths=torch.tensor([len(i['h']) for i in items])
    h=torch.stack([F.pad(i['h'],(0,0,0,maxlen-len(i['h']))) for i in items])
    mask=torch.arange(maxlen)[None,:]<lengths[:,None]
    h=h+model.type_emb(torch.tensor([i['qtype'] for i in items]))[:,None,:]
    for layer in model.head.layers:h=layer(h,src_key_padding_mask=~mask)
    markers=torch.tensor([i['markers'] for i in items])
    return model.scorer(torch.gather(h,1,markers[:,:,None].expand(-1,-1,h.shape[-1]))).squeeze(-1)

def main():
    prior=runpy.run_path('scripts/train_laya_scorer.py');helper=runpy.run_path('scripts/compare_laya_prompts.py');metrics=runpy.run_path('scripts/evaluate_laya_gold.py')['metrics']
    read,dump,sha=prior['read'],prior['dump'],prior['sha'];cache=json.loads((ROOT/'cache-manifest.json').read_text());assert cache['completed']
    lock=json.loads((DATA/'data-lock.json').read_text());assert sha(DATA/'data-lock.json')==cache['data_lock_sha256']
    for n,v in lock['sha256'].items():assert sha(DATA/(n+'.jsonl'))==v
    for p,v in lock['protected_files'].items():assert sha(Path(p))==v
    upstream=json.loads((ROOT/'upstream/source-lock.json').read_text())
    assert sha(ROOT/'upstream/laya_finetune_typed_decisions_mps.py')==upstream['sha256']
    plan={'source':upstream,'start':'original pinned multilingual base; no prior fine-tuned or auxiliary readout weights',
          'train_cases':418,'train_question_rows':1672,'validation_cases':{'legacy':22,'ripa':90},'epochs':4,'batch_question_rows':8,
          'optimizer':'AdamW','lr_start':1e-4,'lr_end':1e-6,'schedule':'cosine over optimizer updates','weight_decay':.01,'gradient_clip':1.,
          'loss':'RLCD proper_reward(log+spherical, w_sph=.75) + one-hot cross entropy, no class weights',
          'noise_samples':4,'noise_sigma_start':.4,'noise_sigma_end':.1,'gradient_accumulation':1,'seed':42,'cpu_threads':4,
          'trainable':['head','scorer'],'trainable_parameters':14768641,'frozen':['encoder','type_emb','act_head'],
          'encoder_memory':'frozen outputs cached in float32; encoder released during training; exact original base restored for final native evaluation',
          'selection':'lowest arithmetic mean of legacy and RIPA validation question cross entropy; report attack misses separately, no all-attack preference',
          'temperature_calibration':False,'reason_no_temperature_calibration':'argmax comparison only; scores are not calibrated safety probabilities',
          'upstream_differences':['frozen encoder, type embedding and act_head per requested scope','local one-hot F1-O labels instead of typed-decisions dataset','no mixed precision','no accumulation needed for batch 8','group-preserving existing validation instead of random question calibration split'],
          'deployment_ready':False,'data_lock_sha256':sha(DATA/'data-lock.json')}
    if (ROOT/'training-plan.json').exists():assert json.loads((ROOT/'training-plan.json').read_text())==plan
    else:dump(ROOT/'training-plan.json',plan)
    assert not (ROOT/'training-summary.json').exists(),'Completed training is not overwritten'
    torch.set_num_threads(4);torch.manual_seed(42);random.seed(42)
    handles={};sets={};casepaths={'legacy_train':Path('data/laya_deepset_domain_v1/train.jsonl'),'legacy_validation':DATA/'legacy_validation.jsonl','ripa_train':DATA/'ripa_train.jsonl','ripa_validation':DATA/'ripa_validation.jsonl'}
    for name,desc in cache['splits'].items():
        cases=read(casepaths[name]);items=[]
        for shard in desc['shards']:
            fp,ip=Path(shard['features']),Path(shard['index']);assert sha(fp)==shard['features_sha256'] and sha(ip)==shard['index_sha256']
            handle=safe_open(str(fp),framework='pt',device='cpu');handles[str(fp)]=handle
            for j,idx in enumerate(json.loads(ip.read_text())):
                t=idx.get('tensor',f'h{j}');items.append({**idx,'path':str(fp),'tensor':t,'length':handle.get_slice(t).get_shape()[0]})
        assert len(items)==len(cases)*4
        for i,c in enumerate(cases):
            for q,key in enumerate(QUESTIONS):
                item=items[4*i+q];assert item['q']==q and item['label']==['YES','NO','UNKNOWN'].index(c['expected_questions'][key])
                if 'case_id' in item:assert item['case_id']==c['case_id']
        sets[name]=(items,cases)
    sets['train']=(sets['legacy_train'][0]+sets['ripa_train'][0],sets['legacy_train'][1]+sets['ripa_train'][1])
    def fetch(items):return [{**i,'h':handles[i['path']].get_tensor(i['tensor'])} for i in items]
    backend=LayaBackend(cpu_threads=4);backend._load();model=backend.agent.model;adapter=helper['CandidateBackend'](backend,QUESTIONS)
    assert lock_hash(backend.metadata['model_lock'])==cache['base_model_lock_sha256'];model.eval()
    for p in model.parameters():p.requires_grad_(False)
    initial_encoder_hash=prior['tensors_hash'](model.encoder.named_parameters());base_file_hash=sha(Path('artifacts/laya/model/model.safetensors'))
    native_checks=[]
    for name in ['legacy_train','legacy_validation','ripa_train','ripa_validation']:
        items,cases=sets[name];trace={};r=assess(*helper['inputs'](cases[0]),adapter,trace);assert r.analysis_complete
        with torch.no_grad():answers=prior['predictions'](decision_forward(model,fetch(items[:4])))
        for q,a in zip(QUESTIONS,answers):
            real=getattr(r.questions,q).model_dump(mode='json');assert a['decision']==real['decision']
            assert max(abs(a['probabilities'][k]-real['probabilities'][k]) for k in ['yes','no','unknown'])<.001
        native_checks.append({'set':name,'case_id':cases[0]['case_id'],'four_question_cache_matches_native':True})
    dump(ROOT/'pretrain-native-checks.json',native_checks)
    frozen_hash=prior['tensors_hash']((n,p) for n,p in model.named_parameters() if n.startswith(('type_emb.','act_head.')))
    # No encoder gradients are needed; release its RAM while using disk-backed outputs.
    model.encoder=None;gc.collect()
    for n,p in model.named_parameters():p.requires_grad_(n.startswith(('head.','scorer.')))
    params=[p for p in model.parameters() if p.requires_grad];assert sum(p.numel() for p in params)==14768641
    optimizer=torch.optim.AdamW(params,lr=plan['lr_start'],weight_decay=plan['weight_decay'])
    def evaluate(name,save=False):
        items,cases=sets[name];model.head.eval();model.scorer.eval();parts=[]
        with torch.no_grad():
            for off in range(0,len(items),8):parts.append(decision_forward(model,fetch(items[off:off+8])))
            logits=torch.cat(parts);labels=torch.tensor([i['label'] for i in items]);loss=float(F.cross_entropy(logits,labels));answers=prior['predictions'](logits);statuses=prior['statuses'](answers)
        rows=[]
        for i,c in enumerate(cases):
            a=dict(zip(QUESTIONS,answers[i*4:i*4+4]));rows.append({**c,'actual_status':statuses[i],'status_correct':statuses[i]==c['expected_status'],'answers':a,'question_errors':[q for q in QUESTIONS if a[q]['decision']!=c['expected_questions'][q]]})
        if save:(ROOT/(name+'-predictions.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf8')
        return {'loss':loss,'metrics':metrics(rows),'question_accuracy':sum(4-len(r['question_errors']) for r in rows)/(len(rows)*4)}
    def state():return {k:v.detach().clone().contiguous() for k,v in model.state_dict().items() if k.startswith(('head.','scorer.'))}
    history=[];start_epoch=1;step=0;best_loss=float('inf');best_epoch=0;initial_hash=prior['tensors_hash'](state().items())
    if (ROOT/'resume.pt').exists():
        saved=torch.load(ROOT/'resume.pt',map_location='cpu',weights_only=True)
        assert saved['plan_sha256']==sha(ROOT/'training-plan.json')
        for name in ['head','scorer']:getattr(model,name).load_state_dict({k[len(name)+1:]:v for k,v in saved['weights'].items() if k.startswith(name+'.')})
        optimizer.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['rng']);step=saved['step'];start_epoch=saved['epoch']+1;history=saved['history'];best_loss=saved['best_loss'];best_epoch=saved['best_epoch']
    else:
        base={name:evaluate(name) for name in ['legacy_validation','ripa_validation']};dump(ROOT/'baseline.json',base)
    items,cases=sets['train'];updates=math.ceil(len(items)/8)*plan['epochs'];started=time.perf_counter()
    for epoch in range(start_epoch,plan['epochs']+1):
        t=time.perf_counter();sigma=.4+(.1-.4)*(epoch-1)/max(1,plan['epochs']-1);rng=random.Random(42+epoch)
        order=list(range(len(items)));rng.shuffle(order);batches=[]
        for off in range(0,len(order),64):
            bucket=sorted(order[off:off+64],key=lambda i:items[i]['length'])
            batches.extend(bucket[j:j+8] for j in range(0,len(bucket),8))
        rng.shuffle(batches);tot_ce=tot_rl=0.
        for bi,batch in enumerate(batches):
            model.head.train();model.scorer.train();selected=[items[i] for i in batch]
            lr=plan['lr_end']+(plan['lr_start']-plan['lr_end'])*(1+math.cos(math.pi*step/max(1,updates-1)))/2;optimizer.param_groups[0]['lr']=lr
            logits=decision_forward(model,fetch(selected));labels=torch.tensor([i['label'] for i in selected]);qtypes=torch.tensor([i['qtype'] for i in selected])
            loss,rl,ce=rlcd_loss(logits,labels,qtypes,sigma);assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(norm);optimizer.step();step+=1;tot_ce+=float(ce.detach());tot_rl+=float(rl.detach())
            if bi%20==0 or bi+1==len(batches):print(json.dumps({'stage':'training','epoch':epoch,'batch':bi+1,'batches':len(batches),'total_update':step,'ce':float(ce.detach()),'seconds':time.perf_counter()-t}),flush=True)
        train=evaluate('train');legacy=evaluate('legacy_validation');ripa=evaluate('ripa_validation');val_loss=(legacy['loss']+ripa['loss'])/2
        row={'epoch':epoch,'updates':step,'sigma':sigma,'lr_end':lr,'mean_batch_ce':tot_ce/len(batches),'mean_batch_rl':tot_rl/len(batches),'train':train,'legacy_validation':legacy,'ripa_validation':ripa,'macro_validation_ce':val_loss,'seconds':time.perf_counter()-t};history.append(row)
        weights=state()
        if val_loss<best_loss:best_loss=val_loss;best_epoch=epoch;save_file(weights,str(ROOT/'decision.safetensors'))
        save_file(weights,str(ROOT/'last-decision.safetensors'));dump(ROOT/'history.json',history)
        torch.save({'weights':weights,'optimizer':optimizer.state_dict(),'rng':torch.get_rng_state(),'step':step,'epoch':epoch,'history':history,'best_loss':best_loss,'best_epoch':best_epoch,'plan_sha256':sha(ROOT/'training-plan.json')},ROOT/'resume.pt')
        print(json.dumps({'epoch_completed':epoch,'seconds':row['seconds'],'selected_epoch':best_epoch,'legacy':legacy['metrics'],'ripa':ripa['metrics']}),flush=True)
    best=load_file(str(ROOT/'decision.safetensors'))
    for name in ['head','scorer']:getattr(model,name).load_state_dict({k[len(name)+1:]:v for k,v in best.items() if k.startswith(name+'.')})
    assert initial_hash!=prior['tensors_hash'](state().items());assert frozen_hash==prior['tensors_hash']((n,p) for n,p in model.named_parameters() if n.startswith(('type_emb.','act_head.')))
    assert sha(Path('artifacts/laya/model/model.safetensors'))==base_file_hash
    manifest={'base_model_lock_sha256':cache['base_model_lock_sha256'],'weights_sha256':sha(ROOT/'decision.safetensors'),'trainable_modules':['head','scorer'],'trainable_parameters':14768641,'prompt_version':PROMPT_VERSION,'selected_epoch':best_epoch,'selection':plan['selection'],'data_lock_sha256':sha(DATA/'data-lock.json'),'deployment_ready':False}
    dump(ROOT/'decision-lock.json',manifest)
    summary={'plan':plan,'selected_epoch':best_epoch,'updates':step,'frozen_nonencoder_parameters_unchanged':True,'base_file_unchanged':True,'initial_encoder_parameter_hash':initial_encoder_hash,'selected':{},'seconds':time.perf_counter()-started,'deployment_ready':False}
    for name in ['train','legacy_validation','ripa_validation']:summary['selected'][name]=evaluate(name,True)
    dump(ROOT/'training-summary.json',summary);print(json.dumps({'training_completed':True,'epoch':best_epoch,'selected':summary['selected']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
