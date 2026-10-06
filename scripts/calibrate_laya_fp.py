"""Train-only out-of-fold calibration for conservative normal clearance repair."""
import json,runpy,time,shutil
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from sklearn.model_selection import GroupKFold
from safetensors.torch import load_file,save_file

ROOT=Path('artifacts/laya/fp-conservative-cpu-v1');HEAD=Path('artifacts/laya/fp-question-head-cpu-v2/standardized')
OLD=Path('artifacts/laya/ripa-augmented-cpu-v1');DATA=Path('data/laya_ripa_domain_v1')
def main():
    ROOT.mkdir(exist_ok=False);torch.set_num_threads(4);threadpool_limits(4)
    h=runpy.run_path('scripts/improve_laya_fp.py');read,dump,sha=h['read'],h['dump'],h['sha']
    training=json.loads((HEAD.parent/'summary.json').read_text());assert training['completed_training']
    cv=json.loads((HEAD/'cv.json').read_text());eligible=[t for t in cv if t['all_fits_converged']]
    chosen={'selected_C':max(eligible,key=lambda t:t['key'])['C']}
    plan={'readout':'standardized four question heads','C':chosen['selected_C'],'calibration':'3-fold GroupKFold on the same 418 training cases; scaler and readout refit inside each fold',
          'score':'minimum of the four probabilities for NO, NO, NO, YES; not calibrated safety probability',
          'threshold':'max out-of-fold score over all non-CLEAR training cases + 0.01, capped at 1; strict greater-than',
          'route':'only replace native answers when readout decisions are NO NO NO YES and score exceeds threshold; otherwise preserve original native Laya answers',
          'validation_used_for_calibration':False,'production_default_changed':False,'human_reviewed':False,'deployment_ready':False}
    dump(ROOT/'plan.json',plan)
    data={}
    for name,p,cp in [('legacy_train',Path('artifacts/laya/deepset-scorer-cpu-v1/train-features.safetensors'),Path('data/laya_deepset_domain_v1/train.jsonl')),
        ('legacy_validation',Path('artifacts/laya/deepset-scorer-cpu-v1/validation-features.safetensors'),DATA/'legacy_validation.jsonl'),
        ('ripa_train',OLD/'ripa_train-features.safetensors',DATA/'ripa_train.jsonl'),('ripa_validation',OLD/'ripa_validation-features.safetensors',DATA/'ripa_validation.jsonl')]:
        t=load_file(str(p));data[name]=(t['features'].numpy().reshape(-1,4,2304).astype('float64'),t['labels'].numpy().reshape(-1,4),read(cp))
    x=np.concatenate([data['legacy_train'][0],data['ripa_train'][0]]);y=np.concatenate([data['legacy_train'][1],data['ripa_train'][1]])
    cases=data['legacy_train'][2]+data['ripa_train'][2];groups=np.array([c['group_id'] for c in cases]);oof=np.zeros((len(x),4,3))
    for i,(tr,va) in enumerate(GroupKFold(3).split(x,y,groups)):
        fitted=h['fit'](x[tr],y[tr],chosen['selected_C'],True);assert all(m.converged_ for m in fitted[0])
        oof[va]=h['probabilities'](fitted,x[va]);print(json.dumps({'calibration_fold':i+1}),flush=True)
    score=oof[:,np.arange(4),[1,1,1,0]].min(-1);negative=np.array([c['expected_status']!='CLEAR' for c in cases]);maximum=float(score[negative].max());threshold=min(1.,maximum+.01)
    accepted=(score>threshold)&(h['decisions'](oof)==[1,1,1,0]).all(-1)
    assert not accepted[negative].any()
    fitted=h['fit'](x,y,chosen['selected_C'],True);assert all(m.converged_ for m in fitted[0])
    ts={k:[] for k in ['weight','bias','mean','scale']}
    for model,scaler in zip(*fitted):
        w=np.zeros((3,2304));b=np.full(3,-1e6)
        if len(model.classes_)==2:
            w[model.classes_[0]]=-model.coef_[0]/2;w[model.classes_[1]]=model.coef_[0]/2
            b[model.classes_[0]]=-model.intercept_[0]/2;b[model.classes_[1]]=model.intercept_[0]/2
        else:w[model.classes_]=model.coef_;b[model.classes_]=model.intercept_
        for key,value in [('weight',w),('bias',b),('mean',scaler.mean_),('scale',scaler.scale_)]:ts[key].append(value)
    tensors={k:torch.tensor(np.stack(v),dtype=torch.float64) for k,v in ts.items()}
    save_file(tensors,str(ROOT/'question-head.safetensors'))
    lock=json.loads((HEAD/'question-head-lock.json').read_text());lock.update(C=chosen['selected_C'],weights_sha256=sha(ROOT/'question-head.safetensors'),selection='best training grouped CV among numerically converged candidates; no validation used')
    dump(ROOT/'question-head-lock.json',lock)
    policy={'policy_version':'f1o-conservative-normal-repair-v1','threshold':threshold,'threshold_source':'training OOF only','max_nonclear_oof_score':maximum,
            'question_head_sha256':lock['weights_sha256'],'base_model_lock_sha256':lock['base_model_lock_sha256'],'prompt_version':lock['prompt_version'],
            'training_data_lock_sha256':sha(DATA/'data-lock.json'),'deployment_ready':False}
    dump(ROOT/'fp-policy.json',policy);dump(ROOT/'oof-calibration.json',{'threshold':threshold,'nonclear_false_clear':0,'normal_accepted':int(accepted[~negative].sum()),'normal_n':int((~negative).sum()),'cases':[{'case_id':c['case_id'],'expected_status':c['expected_status'],'score':float(z),'accepted':bool(a)} for c,z,a in zip(cases,score,accepted)]})
    from f1.laya.question_head import QuestionReadout
    head=QuestionReadout(ROOT,json.loads(Path('artifacts/laya/model/model-lock.json').read_text()))
    assert np.max(np.abs(head.logits(torch.tensor(x,dtype=torch.float32).reshape(-1,3,768)).softmax(-1).numpy()-h['probabilities'](fitted,x)))<1e-7
    metrics=runpy.run_path('scripts/evaluate_laya_gold.py')['metrics'];summary={'plan':plan,'policy':policy,'evaluations':{}}
    for name,(xx,yy,cc) in data.items():
        features=torch.tensor(xx,dtype=torch.float32).reshape(-1,3,768)
        p=head.logits(features).softmax(-1).numpy();d=h['decisions'](p);s=p[:,np.arange(4),[1,1,1,0]].min(-1)
        baseline=read(OLD/'legacy_only'/(name+'-predictions.jsonl'));assert [r['case_id'] for r in baseline]==[r['case_id'] for r in cc]
        rows=[]
        for c,b,prob,dec,z in zip(cc,baseline,p,d,s):
            accept=bool(z>threshold and list(dec)==[1,1,1,0]);actual='CLEAR' if accept else b['actual_status']
            a={q:{'decision':['YES','NO','UNKNOWN'][int(dec[j])],'probabilities':dict(zip(['yes','no','unknown'],prob[j].tolist()))} for j,q in enumerate(h['QUESTIONS'])} if accept else b['answers']
            rows.append({**c,'actual_status':actual,'status_correct':actual==c['expected_status'],'answers':a,'question_errors':[q for q in h['QUESTIONS'] if a[q]['decision']!=c['expected_questions'][q]],'normal_repair_applied':accept,'normal_score':float(z),'baseline_status':b['actual_status']})
        (ROOT/(name+'-predictions.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf8')
        summary['evaluations'][name]={'before':metrics(baseline),'after':metrics(rows)}
    dump(ROOT/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
