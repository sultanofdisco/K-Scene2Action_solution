"""Calibrate only trusted task profiles with sufficient train-only OOF support."""
import json,runpy,hashlib,shutil
from collections import defaultdict
from pathlib import Path
import numpy as np
from f1.laya.question_head import context_profile,QuestionReadout
from f1.laya.pipeline import parse_answer
from safetensors.torch import load_file

ROOT=Path('artifacts/laya/fp-profile-cpu-v1');PREV=Path('artifacts/laya/fp-conservative-cpu-v1');OLD=Path('artifacts/laya/ripa-augmented-cpu-v1')
def main():
    h=runpy.run_path('scripts/improve_laya_fp.py');read,dump,sha=h['read'],h['dump'],h['sha'];ROOT.mkdir(exist_ok=False)
    plan={'mode':'normal-clearance repair restricted to trusted task profiles with training support','support_minimum':{'normal':20,'nonclear':20,'groups':6},
          'profile':'sha256 of canonical trusted_context only: command, target, explicit_goal, constraints; never use OCR, case ID, origin or label at inference',
          'threshold':'profile max non-CLEAR training OOF normal score +0.01, capped at1, strict greater-than',
          'unknown_profile':'unchanged original native Laya prediction','validation_used_for_thresholds':False,'production_default_changed':False,'deployment_ready':False}
    dump(ROOT/'plan.json',plan)
    cases=read(Path('data/laya_ripa_domain_v1/train.jsonl'));oof=json.loads((PREV/'oof-calibration.json').read_text())['cases'];byid={r['case_id']:r for r in oof}
    groups=defaultdict(list)
    for c in cases:
        ctx={k:c[k] for k in ['command','target','explicit_goal','constraints']};groups[context_profile(ctx)].append(c)
    profiles={};audit=[]
    for key,rs in groups.items():
        normal=[r for r in rs if r['expected_status']=='CLEAR'];nonclear=[r for r in rs if r['expected_status']!='CLEAR'];ng=len(set(r['group_id'] for r in rs))
        support={'normal':len(normal),'nonclear':len(nonclear),'groups':ng};active=len(normal)>=20 and len(nonclear)>=20 and ng>=6
        entry={'profile':key,'support':support,'active':active,'command':rs[0]['command']}
        if active:
            mx=max(byid[r['case_id']]['score'] for r in nonclear);threshold=min(1.,mx+.01)
            profiles[key]={'threshold':threshold,'support':support,'max_nonclear_oof_score':mx}
            entry.update(threshold=threshold,oof_normal_accepted=sum(byid[r['case_id']]['score']>threshold for r in normal),oof_nonclear_accepted=sum(byid[r['case_id']]['score']>threshold for r in nonclear))
        audit.append(entry)
    lock=json.loads((PREV/'question-head-lock.json').read_text())
    for name in ['question-head.safetensors','question-head-lock.json']:shutil.copyfile(PREV/name,ROOT/name)
    policy={'policy_version':'f1o-profile-normal-repair-v1','profiles':profiles,'question_head_sha256':lock['weights_sha256'],'base_model_lock_sha256':lock['base_model_lock_sha256'],
            'prompt_version':lock['prompt_version'],'training_data_lock_sha256':sha(Path('data/laya_ripa_domain_v1/data-lock.json')),'deployment_ready':False}
    dump(ROOT/'fp-policy.json',policy);dump(ROOT/'profile-calibration.json',audit)
    head=QuestionReadout(ROOT,json.loads(Path('artifacts/laya/model/model-lock.json').read_text()))
    metrics=runpy.run_path('scripts/evaluate_laya_gold.py')['metrics'];summary={'plan':plan,'policy':policy,'evaluations':{}}
    for name in ['legacy_train','ripa_train','legacy_validation','ripa_validation']:
        rows=read(PREV/(name+'-predictions.jsonl'));base=read(OLD/'legacy_only'/(name+'-predictions.jsonl'));out=[]
        feature_path=(OLD/(name+'-features.safetensors') if name.startswith('ripa') else Path('artifacts/laya/deepset-scorer-cpu-v1')/(('train' if name=='legacy_train' else 'validation')+'-features.safetensors'))
        features=load_file(str(feature_path))['features']
        for i,(r,b) in enumerate(zip(rows,base)):
            key=context_profile({k:r[k] for k in ['command','target','explicit_goal','constraints']});p=profiles.get(key)
            candidate={q:parse_answer(a).model_dump(mode='json') for q,a in head.answers(features[i*4:i*4+4]).items()}
            accept=bool(p and r['normal_score']>p['threshold'] and [candidate[q]['decision'] for q in h['QUESTIONS']]==['NO','NO','NO','YES'])
            actual='CLEAR' if accept else b['actual_status']
            answers=candidate if accept else b['answers']
            out.append({**r,'actual_status':actual,'status_correct':actual==r['expected_status'],'answers':answers,'question_errors':[q for q in h['QUESTIONS'] if answers[q]['decision']!=r['expected_questions'][q]],'normal_repair_applied':accept,'profile_supported':p is not None,'profile':key,'baseline_status':b['actual_status']})
        (ROOT/(name+'-predictions.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in out),encoding='utf8')
        summary['evaluations'][name]={'before':metrics(base),'after':metrics(out)}
    dump(ROOT/'preliminary-summary.json',summary);print(json.dumps({'profiles':audit,'evaluations':summary['evaluations']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
