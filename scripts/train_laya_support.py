"""Train-only neighborhood support gate; stress v1 is now development diagnosis."""
import json,runpy,time,shutil
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics.pairwise import euclidean_distances
from safetensors.torch import load_file,save_file
from threadpoolctl import threadpool_limits
from f1.laya.backend import LayaBackend,check_budget
from f1.laya.pipeline import make_state
from f1.laya.questions_v2 import QUESTIONS

ROOT=Path('artifacts/laya/fp-support-cpu-v1');PREV=Path('artifacts/laya/fp-profile-cpu-v1');OLD=Path('artifacts/laya/ripa-augmented-cpu-v1')
def main():
    h=runpy.run_path('scripts/improve_laya_fp.py');read,dump,sha=h['read'],h['dump'],h['sha'];ROOT.mkdir(exist_ok=False);torch.set_num_threads(4);threadpool_limits(4)
    plan={'purpose':'Reject out-of-support normal corrections, keep native Laya otherwise','training_data':'RIPA train 360 only; existing head unchanged',
          'normalization':'train-fold feature standardization then row L2 normalization','support_score':'nearest nonclear distance / (nearest normal distance + nearest nonclear distance)',
          'selection':'training 3-fold grouped OOF only; threshold=max nonclear support score +0.01; distance radius=95th percentile of normal OOF nearest normal distances',
          'stress_v1_role':'development diagnosis already inspected; not independent test','deployment_ready':False}
    dump(ROOT/'plan.json',plan)
    x=load_file(str(OLD/'ripa_train-features.safetensors'))['features'].numpy().reshape(-1,9216).astype('float64');cases=read(Path('data/laya_ripa_domain_v1/ripa_train.jsonl'))
    y=np.array([c['expected_status']=='CLEAR' for c in cases]);groups=np.array([c['group_id'] for c in cases]);scores=np.zeros(len(x));ds=np.zeros(len(x))
    def normalize(s,z):
        z=s.transform(z);return z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12)
    def support(z,train,labels):
        dn=euclidean_distances(z,train[labels]).min(-1);da=euclidean_distances(z,train[~labels]).min(-1)
        return da/np.maximum(da+dn,1e-12),dn
    for tr,va in GroupKFold(3).split(x,y,groups):
        s=StandardScaler().fit(x[tr]);scores[va],ds[va]=support(normalize(s,x[va]),normalize(s,x[tr]),y[tr])
    threshold=min(1.,max(.5,float(scores[~y].max())+.01));radius=float(np.quantile(ds[y],.95));scaler=StandardScaler().fit(x);train=normalize(scaler,x)
    save_file({'mean':torch.tensor(scaler.mean_),'scale':torch.tensor(scaler.scale_),'normal':torch.tensor(train[y]),'nonclear':torch.tensor(train[~y])},str(ROOT/'support.safetensors'))
    for n in ['question-head.safetensors','question-head-lock.json','fp-policy.json']:shutil.copyfile(PREV/n,ROOT/n)
    policy={'policy_version':'f1o-neighborhood-support-v1','threshold':threshold,'normal_radius':radius,'weights_sha256':sha(ROOT/'support.safetensors'),
            'question_head_sha256':sha(ROOT/'question-head.safetensors'),'profile_policy_sha256':sha(ROOT/'fp-policy.json'),'deployment_ready':False}
    dump(ROOT/'support-lock.json',policy);dump(ROOT/'oof-support.json',{'normal_accepted':int(((scores>threshold)&(ds<=radius)&y).sum()),'normal_n':int(y.sum()),'nonclear_accepted':int(((scores>threshold)&(ds<=radius)&~y).sum()),'threshold':threshold,'radius':radius})
    # Obtain frozen native features for the already-inspected stress set; do not fit to it.
    stress=read(PREV/'stress-cases.jsonl');backend=LayaBackend(cpu_threads=4);backend._load();helper=runpy.run_path('scripts/compare_laya_prompts.py');cap=[]
    hook=backend.agent.model.scorer.register_forward_pre_hook(lambda m,a:cap.append(a[0].detach().clone()));pieces=[]
    for i in range(0,len(stress),2):
        states=[]
        for c in stress[i:i+2]:
            req,ctx,binding,ocr=helper['inputs'](c);state=make_state(req,ctx,ocr);check_budget(backend.agent.tok,state,QUESTIONS);states.append(state)
        cap.clear();backend.agent.predict_batch(states,QUESTIONS,batch_size=2,max_len=1024,head_max_len=256,sort_by_length=False)
        assert len(cap)==1;pieces.append(cap[0]);print(json.dumps({'stress_cache':min(i+2,len(stress))}),flush=True)
    hook.remove();features=torch.cat(pieces);save_file({'features':features},str(ROOT/'stress-v1-features.safetensors'))
    summary={'plan':plan,'policy':policy,'evaluations':{}};metrics=runpy.run_path('scripts/evaluate_laya_gold.py')['metrics']
    for name,fpath,rpath in [('ripa_validation',OLD/'ripa_validation-features.safetensors',PREV/'ripa_validation-native.jsonl'),('stress-v1',ROOT/'stress-v1-features.safetensors',PREV/'new-stress-native.jsonl')]:
        fx=load_file(str(fpath))['features'].numpy().reshape(-1,9216);rows=read(rpath);sp,dn=support(normalize(scaler,fx),train,y);out=[]
        for r,v,d in zip(rows,sp,dn):
            accept=r['normal_repair'].get('applied',False) and v>threshold and d<=radius
            status='CLEAR' if accept else 'ATTACK' # Original base was ATTACK on every semantic case in these two sets.
            out.append({**r,'actual_status':status,'status_correct':status==r['expected_status'],'support_score':float(v),'support_distance':float(d),'support_accepted':bool(accept)})
        summary['evaluations'][name]=metrics(out);(ROOT/(name+'-screening.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in out),encoding='utf8')
    dump(ROOT/'screening-summary.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
