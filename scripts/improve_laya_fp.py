"""Train question-specific heads; select regularization using train-only grouped CV."""
import os,json,runpy,time,warnings,hashlib
os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from safetensors.torch import load_file,save_file
from f1.laya.questions_v2 import QUESTIONS,PROMPT_VERSION

ROOT=Path('artifacts/laya/fp-question-head-cpu-v2');OLD=Path('artifacts/laya/ripa-augmented-cpu-v1');DATA=Path('data/laya_ripa_domain_v1')
def read(p):return [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines()]
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fit(x,y,c,scaled):
    models=[];scalers=[]
    for q in range(4):
        scaler=StandardScaler(with_mean=scaled,with_std=scaled);z=scaler.fit_transform(x[:,q])
        model=LogisticRegression(C=c,max_iter=3000,tol=1e-6,class_weight=None,solver='lbfgs')
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always',ConvergenceWarning);model.fit(z,y[:,q])
        models.append(model);scalers.append(scaler)
        model.converged_=not any(issubclass(w.category,ConvergenceWarning) for w in ws)
    return models,scalers
def probabilities(fitted,x):
    out=np.zeros((len(x),4,3))
    for q,(model,scaler) in enumerate(zip(*fitted)):
        out[:,q,model.classes_]=model.predict_proba(scaler.transform(x[:,q]))
    return out
def decisions(probs):
    p=np.round(probs,4);d=p.argmax(-1);d[(p==p.max(-1,keepdims=True)).sum(-1)>1]=2
    return d
def metric(probs,cases):
    d=decisions(probs);out=[]
    for a,c in zip(d,cases):
        actual='ATTACK' if (a[:3]==0).any() else 'CLEAR' if list(a)==[1,1,1,0] else 'UNKNOWN'
        out.append({**c,'actual_status':actual,'status_correct':actual==c['expected_status']})
    metrics=runpy.run_path('scripts/evaluate_laya_gold.py')['metrics']
    return {'all':metrics(out),'legacy':metrics([r for r in out if 'hazard_id' not in r]),'ripa':metrics([r for r in out if 'hazard_id' in r])},out
def selection(m):
    a,b=m['legacy'],m['ripa'];recall=min(a['attack_recall'],b['attack_recall']);leak=max(a['attack_clear_rate'],b['attack_clear_rate'])
    clear=(a['benign_clear_rate']+b['benign_clear_rate'])/2
    eligible=recall>=.9 and leak<=.05
    return (int(eligible),clear if eligible else (recall+clear)/2,-leak,recall)
def main():
    ROOT.mkdir(exist_ok=True);torch.set_num_threads(4);threadpool_limits(4)
    plan={'architecture':'four independent multinomial logistic heads on native Laya post-decision marker features (3 x 768 per question)',
          'arms':['unscaled','standardized'],'C_candidates':[.01,.1,1.,10.],'selection':'3-fold GroupKFold on training groups only. Prefer recall >=90% and attack CLEAR <=5% in BOTH source domains; then maximize macro normal clearance. If none eligible, diagnostic balanced utility only; not deployable.',
          'class_weight':None,'folds':3,'train_cases':418,'train_questions':1672,'validation_used_for_selection':False,'prior_diagnosis_used_for_selection':False,
          'normalization':'fit each question scaler using training fold only; refit on full training only after C selected','gate':'unchanged first-three YES => ATTACK; NO NO NO YES => CLEAR; otherwise UNKNOWN',
          'human_reviewed':False,'deployment_ready':False,'data_lock_sha256':sha(DATA/'data-lock.json')}
    if (ROOT/'plan.json').exists():assert json.loads((ROOT/'plan.json').read_text())==plan
    else:dump(ROOT/'plan.json',plan)
    data={}
    for name,p,cp in [('legacy_train',Path('artifacts/laya/deepset-scorer-cpu-v1/train-features.safetensors'),Path('data/laya_deepset_domain_v1/train.jsonl')),
        ('legacy_validation',Path('artifacts/laya/deepset-scorer-cpu-v1/validation-features.safetensors'),DATA/'legacy_validation.jsonl'),
        ('ripa_train',OLD/'ripa_train-features.safetensors',DATA/'ripa_train.jsonl'),('ripa_validation',OLD/'ripa_validation-features.safetensors',DATA/'ripa_validation.jsonl')]:
        tensors=load_file(str(p));cases=read(cp);x=tensors['features'].numpy().reshape(-1,4,2304).astype('float64');y=tensors['labels'].numpy().reshape(-1,4)
        assert np.array_equal(y,np.array([[['YES','NO','UNKNOWN'].index(c['expected_questions'][q]) for q in QUESTIONS] for c in cases]))
        data[name]=(x,y,cases)
    x=np.concatenate([data['legacy_train'][0],data['ripa_train'][0]]);y=np.concatenate([data['legacy_train'][1],data['ripa_train'][1]])
    cases=data['legacy_train'][2]+data['ripa_train'][2];groups=np.array([c['group_id'] for c in cases]);folds=list(GroupKFold(3).split(x,y,groups))
    dump(ROOT/'folds.json',[{'train_ids':[cases[i]['case_id'] for i in tr],'validation_ids':[cases[i]['case_id'] for i in va]} for tr,va in folds])
    summary={'plan':plan,'arms':{}};start=time.perf_counter()
    for arm,scaled in [('unscaled',False),('standardized',True)]:
        folder=ROOT/arm;folder.mkdir(exist_ok=True);trials=json.loads((folder/'cv.json').read_text()) if (folder/'cv.json').exists() else [];best=max(((tuple(t['key']),t['C']) for t in trials),default=None)
        for c in plan['C_candidates']:
            if any(t['C']==c for t in trials):continue
            oof=np.zeros((len(x),4,3));converged=[]
            for fold,(tr,va) in enumerate(folds):
                assert not set(groups[tr])&set(groups[va]);fitted=fit(x[tr],y[tr],c,scaled);oof[va]=probabilities(fitted,x[va])
                converged.extend(m.converged_ for m in fitted[0])
                print(json.dumps({'arm':arm,'C':c,'fold':fold+1,'seconds':time.perf_counter()-start}),flush=True)
            m,_=metric(oof,cases);key=selection(m);trials.append({'C':c,'oof':m,'key':key,'all_fits_converged':all(converged)})
            if best is None or key>best[0]:best=(key,c)
            dump(folder/'cv.json',trials)
        fitted=fit(x,y,best[1],scaled)
        weights=[];bias=[];means=[];scales=[]
        for model,scaler in zip(*fitted):
            w=np.zeros((3,2304));b=np.full(3,-1e6)
            if len(model.classes_)==2:
                w[model.classes_[0]]=-model.coef_[0]/2;w[model.classes_[1]]=model.coef_[0]/2
                b[model.classes_[0]]=-model.intercept_[0]/2;b[model.classes_[1]]=model.intercept_[0]/2
            else:w[model.classes_]=model.coef_;b[model.classes_]=model.intercept_
            weights.append(w);bias.append(b);means.append(scaler.mean_ if scaled else np.zeros(2304));scales.append(scaler.scale_ if scaled else np.ones(2304))
        ts={k:torch.tensor(np.stack(v),dtype=torch.float64) for k,v in [('weight',weights),('bias',bias),('mean',means),('scale',scales)]}
        save_file(ts,str(folder/'question-head.safetensors'))
        lock={'base_model_lock_sha256':json.loads((OLD/'cache-audit.json').read_text())['base_model_lock_sha256'],'weights_sha256':sha(folder/'question-head.safetensors'),
              'prompt_version':PROMPT_VERSION,'questions':list(QUESTIONS),'classes':['YES','NO','UNKNOWN'],'architecture':'question-specific-linear-v1','C':best[1],
              'features':'original frozen native scorer input, 3 x 768 flattened per question','dtype':'float64','training_data_lock_sha256':plan['data_lock_sha256'],'deployment_ready':False}
        dump(folder/'question-head-lock.json',lock)
        result={'selected_C':best[1],'oof_eligible':bool(best[0][0]),'full_fit_converged':all(m.converged_ for m in fitted[0]),'evaluations':{}}
        for name,(xx,yy,cc) in {**data,'combined_train':(x,y,cases)}.items():
            probs=probabilities(fitted,xx);_,rows=metric(probs,cc)
            # Verify exported parameters reproduce sklearn probabilities.
            tx=torch.tensor(xx);logits=torch.einsum('nqd,qcd->nqc',(tx-ts['mean'])/ts['scale'],ts['weight'])+ts['bias']
            assert np.max(np.abs(logits.softmax(-1).numpy()-probs))<1e-7
            for r,p,a in zip(rows,probs,decisions(probs)):
                r['answers']={q:{'decision':['YES','NO','UNKNOWN'][int(a[j])],'probabilities':dict(zip(['yes','no','unknown'],p[j].tolist()))} for j,q in enumerate(QUESTIONS)}
                r['question_errors']=[q for q in QUESTIONS if r['answers'][q]['decision']!=r['expected_questions'][q]]
            (folder/(name+'-predictions.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf8')
            result['evaluations'][name]=runpy.run_path('scripts/evaluate_laya_gold.py')['metrics'](rows)
        summary['arms'][arm]=result;dump(ROOT/'summary.json',summary)
    summary['completed_training']=True;summary['seconds']=time.perf_counter()-start;dump(ROOT/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
