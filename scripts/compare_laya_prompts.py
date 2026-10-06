"""Compare frozen V1/V2 choice wording with unchanged weights and gate rules."""
import copy,hashlib,json,os,runpy,time
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
from f1.bindings import bind_input,command_sha256
from f1.schemas import ImageRef,Request,TaskContext,Target,OCRRegion,OCRResult
from f1.laya.backend import LayaBackend
from f1.laya.pipeline import assess
from f1.laya.questions import QUESTIONS as V1,PROMPT_VERSION as P1
from f1.laya.questions_v2 import QUESTIONS as V2,PROMPT_VERSION as P2

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,x):path.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def read(path):return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]

class CandidateBackend:
    def __init__(self,base,questions):self.base,self.questions=base,questions
    @property
    def version(self):return self.base.version
    @property
    def metadata(self):return self.base.metadata
    def predict(self,state,questions):
        assert list(questions)==list(self.questions)
        return self.base.predict(state,self.questions)

def inputs(case):
    image=ImageRef(asset_id='synthetic-transcript:'+case['case_id'],sha256=hashlib.sha256(case['case_id'].encode()).hexdigest(),width=1000,height=1000)
    request=Request(request_id=case['case_id'],command=case['command'],image=image,environment_version='laya-gold-v1')
    context=TaskContext(context_id=case['case_id']+'-context',command_sha256=command_sha256(case['command']),environment_version=request.environment_version,context_status='READY',target=Target(description=case['target']),explicit_goal=case['explicit_goal'],constraints=tuple(case['constraints']))
    regions=tuple(OCRRegion(region_id=f'r{i}',polygon=((0.,0.),(900.,0.),(900.,100.)),raw_text=text,recognition_score=1.,group_id=f'g{i}',reading_order=i) for i,text in enumerate(case['ocr_texts']))
    ocr=OCRResult(image=image,status=case['ocr_status'],regions=regions,reason_codes=tuple(case['ocr_reasons']),model_version='synthetic-oracle-transcript-v1')
    return request,context,bind_input(request,image,'synthetic-transcript-no-image-v1'),ocr

def compare(before,after,metric):
    before={r['case_id']:r for r in before if r['suite']=='semantic'}
    after={r['case_id']:r for r in after if r['suite']=='semantic'}
    assert before.keys()==after.keys()
    return {'v1':metric(list(before.values())),'v2':metric(list(after.values())),
        'improved_status':[k for k in before if not before[k]['status_correct'] and after[k]['status_correct']],
        'regressed_status':[k for k in before if before[k]['status_correct'] and not after[k]['status_correct']],
        'changed_status':[k for k in before if before[k]['actual_status']!=after[k]['actual_status']]}

def main():
    out=Path('artifacts/laya/prompt-comparison-cpu-v2');out.mkdir(parents=True,exist_ok=False)
    data=Path('data/laya_validation_v2')
    lock=json.loads((data/'comparison-lock.json').read_text(encoding='utf-8'))
    assert sha(data/'cases.jsonl')==lock['sha256']
    assert sha(Path('src/f1/laya/questions_v2.py'))==lock['candidate_source_sha256']
    assert lock['questions']=={'v1':V1,'v2':V2}
    assert sha(Path('data/laya_gold_v1/cases.jsonl'))==lock['development_gold_sha256']
    (out/'validation-cases.jsonl').write_bytes((data/'cases.jsonl').read_bytes())
    save(out/'comparison-lock.json',lock)
    development=read(Path('data/laya_gold_v1/cases.jsonl'))
    validation=read(data/'cases.jsonl')
    old=read(Path('artifacts/laya/gold-eval-cpu-v1/predictions.jsonl'))
    old_summary=json.loads(Path('artifacts/laya/gold-eval-cpu-v1/summary.json').read_text(encoding='utf-8'))
    assert old_summary['gold_sha256']==lock['development_gold_sha256'] and old_summary['prompt_version']==P1
    helpers=runpy.run_path('scripts/evaluate_laya_gold.py')
    backend=LayaBackend(cpu_threads=4)
    adapters={'v1':CandidateBackend(backend,V1),'v2':CandidateBackend(backend,V2)}
    outputs={key:[] for key in ['development-v2','validation-v1','validation-v2']}
    for key in outputs:(out/key).mkdir()
    jobs=[('development-v2',c,'v2') for c in development]
    jobs += [('validation-'+v,c,v) for c in validation for v in ['v1','v2']]
    start=datetime.now(timezone.utc).isoformat()
    for key,case,variant in jobs:
        trace={};t=time.perf_counter()
        result=assess(*inputs(case),adapters[variant],trace)
        seconds=time.perf_counter()-t
        trace['prompt_version']={'v1':P1,'v2':P2}[variant]
        trace['comparison_adapter']=True
        meta=trace.pop('backend',{})
        trace['token_budget']=meta.get('token_budget')
        trace['backend_ref']='../model-metadata.json'
        answers=result.questions.model_dump(mode='json')
        errors=[q for q,gold in (case['expected_questions'] or {}).items() if not answers[q] or answers[q]['decision']!=gold]
        row={**case,'actual_status':str(result.status),'status_correct':str(result.status)==case['expected_status'],
            'answers':answers,'question_errors':errors,'reason_codes':list(result.reason_codes),
            'seconds':seconds,'laya_executed':trace['laya_executed'],'analysis_complete':result.analysis_complete,'prompt_version':trace['prompt_version']}
        if 'expected_laya_executed' in case:row['guard_execution_correct']=row['laya_executed']==case['expected_laya_executed']
        outputs[key].append(row)
        save(out/key/(case['case_id']+'-assessment.json'),result.model_dump(mode='json'))
        save(out/key/(case['case_id']+'-trace.json'),trace)
        with (out/key/'predictions.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        print(json.dumps(dict(run=key,id=case['case_id'],expected=case['expected_status'],actual=row['actual_status'],question_errors=errors),ensure_ascii=False),flush=True)
    model=copy.deepcopy(backend.metadata);model.pop('token_budget',None)
    save(out/'model-metadata.json',model)
    summary={'started_utc':start,'completed_utc':datetime.now(timezone.utc).isoformat(),'human_reviewed':False,
        'candidate_selected_before_validation':True,'production_changed':False,'weights_updated':False,
        'development_baseline_predictions_sha256':sha(Path('artifacts/laya/gold-eval-cpu-v1/predictions.jsonl')),
        'development':compare(old,outputs['development-v2'],helpers['metrics']),
        'validation':compare(outputs['validation-v1'],outputs['validation-v2'],helpers['metrics']),
        'runs':{}}
    for key,rows in outputs.items():
        semantic=[r for r in rows if r['suite']=='semantic']
        run={'suites':{s:helpers['metrics']([r for r in rows if r['suite']==s]) for s in ['semantic','ambiguous','guard']},
            'languages':{lang:helpers['metrics']([r for r in semantic if r['language']==lang]) for lang in ['en','ko']},
            'questions':{q:{'correct':sum(q not in r['question_errors'] for r in semantic),'n':len(semantic),'confusion':dict(Counter(r['expected_questions'][q]+' -> '+(r['answers'][q]['decision'] if r['answers'][q] else 'MISSING') for r in semantic))} for q in V1},
            'four_questions_correct':sum(not r['question_errors'] for r in semantic),
            'guard_execution_checks_passed':all(r.get('guard_execution_correct',True) for r in rows)}
        summary['runs'][key]=run
        save(out/key/'summary.json',run)
        helpers['render_review'](out/key,rows)
        viewer=out/key/'error-review.html'
        viewer.write_text(viewer.read_text(encoding='utf-8').replace('Laya 소규모 정답 세트 평가',key+' Laya 비교 평가'),encoding='utf-8')
    assert sha(data/'cases.jsonl')==lock['sha256']
    assert sha(Path('src/f1/laya/questions_v2.py'))==lock['candidate_source_sha256']
    save(out/'summary.json',summary)
    print(json.dumps({k:summary[k] for k in ['development','validation']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
