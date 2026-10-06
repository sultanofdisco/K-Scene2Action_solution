"""Extract original-base scorer features for RIPA in bounded native SDK batches."""
import json,os,runpy,time
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import torch
from safetensors.torch import save_file
from f1.laya.backend import LayaBackend,check_budget
from f1.laya.pipeline import make_state,parse_answer
from f1.laya.questions_v2 import QUESTIONS
from f1.laya.finetuned import lock_hash

OUT=Path('artifacts/laya/ripa-augmented-cpu-v1')
DATA=Path('data/laya_ripa_domain_v1')
def main():
    OUT.mkdir(parents=True,exist_ok=False)
    helper=runpy.run_path('scripts/compare_laya_prompts.py');prior=runpy.run_path('scripts/train_laya_scorer.py')
    read,dump,sha=prior['read'],prior['dump'],prior['sha']
    lock=json.loads((DATA/'data-lock.json').read_text());dump(OUT/'data-lock.json',lock)
    backend=LayaBackend(cpu_threads=4);backend._load();model=backend.agent.model;model.eval()
    for p in model.parameters():p.requires_grad_(False)
    original_lock=json.loads(Path('artifacts/laya/deepset-scorer-cpu-v1/scorer-lock.json').read_text())
    assert original_lock['base_model_lock_sha256']==lock_hash(backend.metadata['model_lock'])
    for p,h in lock['protected_files'].items():assert sha(Path(p))==h
    capture=[];hook=model.scorer.register_forward_pre_hook(lambda m,a:capture.append(a[0].detach().clone()))
    audit={'base_model_lock_sha256':lock_hash(backend.metadata['model_lock']),'data_lock_sha256':sha(DATA/'data-lock.json'),'splits':{},'all_predictions_reconstructed':True}
    started=time.perf_counter()
    for split in ['ripa_train','ripa_validation']:
        cases=read(DATA/(split+'.jsonl'));assert sha(DATA/(split+'.jsonl'))==lock['sha256'][split]
        pieces=[];baseline=[];budgets=[]
        for start in range(0,len(cases),2):
            chunk=cases[start:start+2];states=[]
            for c in chunk:
                request,context,binding,ocr=helper['inputs'](c);state=make_state(request,context,ocr)
                assert not any(k in json.dumps(state) for k in ['expected_questions','source_case_id','rationale','hazard_id'])
                budget=check_budget(backend.agent.tok,state,QUESTIONS)
                budgets.append({'case_id':c['case_id'],'questions':budget});states.append(state)
            capture.clear()
            result=backend.agent.predict_batch(states,QUESTIONS,batch_size=2,max_len=1024,head_max_len=256,sort_by_length=False)
            assert len(capture)==1 and capture[0].shape[:2]==(4*len(chunk),3)
            features=capture[0].clone();pieces.append(features)
            with torch.no_grad():reconstructed=prior['predictions'](model.scorer(features).squeeze(-1))
            for i,(c,res) in enumerate(zip(chunk,result)):
                assert res['usage']['truncated'] is False and res['usage']['state_tokens_dropped']==0
                answers={q:parse_answer(res['answers'][q]).model_dump(mode='json') for q in QUESTIONS}
                for j,q in enumerate(QUESTIONS):
                    assert answers[q]['decision']==reconstructed[i*4+j]['decision']
                    assert max(abs(answers[q]['probabilities'][k]-reconstructed[i*4+j]['probabilities'][k]) for k in ['yes','no','unknown'])<0.001
                status=prior['statuses']([answers[q] for q in QUESTIONS])[0]
                baseline.append({**c,'answers':answers,'actual_status':status,'status_correct':status==c['expected_status'],
                    'question_errors':[q for q in QUESTIONS if answers[q]['decision']!=c['expected_questions'][q]]})
            if (start+2)%20==0 or start+2>=len(cases):
                print(json.dumps({'stage':'cache','split':split,'done':min(start+2,len(cases)),'total':len(cases),'seconds':time.perf_counter()-started}),flush=True)
        labels=torch.tensor([['YES','NO','UNKNOWN'].index(c['expected_questions'][q]) for c in cases for q in QUESTIONS])
        save_file({'features':torch.cat(pieces).contiguous(),'labels':labels},str(OUT/(split+'-features.safetensors')))
        (OUT/(split+'-baseline.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in baseline),encoding='utf-8')
        dump(OUT/(split+'-budget.json'),budgets)
        audit['splits'][split]={'cases':len(cases),'features_sha256':sha(OUT/(split+'-features.safetensors')),'max_tokens':max(q['sequence_tokens'] for b in budgets for q in b['questions'].values()),'all_untruncated':True}
        dump(OUT/'cache-audit.json',audit)
    hook.remove();audit['seconds']=time.perf_counter()-started;audit['completed']=True;dump(OUT/'cache-audit.json',audit)
    print(json.dumps(audit,indent=2))

if __name__=='__main__':main()
