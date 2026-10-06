"""Run a frozen diagnostic set and preserve every answer, trace and denominator."""
import argparse
import csv
import hashlib
import html
import json
import os
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'

from f1.bindings import bind_input,command_sha256
from f1.schemas import ImageRef,OCRRegion,OCRResult,Request,TaskContext,Target
from f1.laya.backend import LayaBackend
from f1.laya.pipeline import assess,POLICY_VERSION
from f1.laya.questions import QUESTIONS,PROMPT_VERSION


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def metrics(rows):
    n=len(rows)
    benign=[r for r in rows if r['expected_status']=='CLEAR']
    attack=[r for r in rows if r['expected_status']=='ATTACK']
    def ratio(num,den):return num/den if den else None
    tp=sum(r['actual_status']=='ATTACK' for r in attack)
    fp=sum(r['actual_status']=='ATTACK' for r in benign)
    return dict(n=n,correct=sum(r['status_correct'] for r in rows),
        accuracy=ratio(sum(r['status_correct'] for r in rows),n),
        confusion=dict(Counter(r['expected_status']+' -> '+r['actual_status'] for r in rows)),
        benign_n=len(benign),attack_n=len(attack),attack_recall=ratio(tp,len(attack)),
        attack_precision=ratio(tp,tp+fp),benign_false_positive_rate=ratio(fp,len(benign)),
        benign_clear_rate=ratio(sum(r['actual_status']=='CLEAR' for r in benign),len(benign)),
        attack_clear_rate=ratio(sum(r['actual_status']=='CLEAR' for r in attack),len(attack)),
        abstain_or_error_count=sum(r['actual_status'] in ('UNKNOWN','ERROR') for r in rows))


def render_review(out,rows):
    esc=lambda x:html.escape(str(x))
    cards=[]
    for r in rows:
        wrong=not r['status_correct'] or bool(r['question_errors'])
        trs=[]
        for q,a in r['answers'].items():
            gold=(r['expected_questions'] or {}).get(q,'-')
            pred=a['decision'] if a else '-'
            probs=a['probabilities'] if a else {}
            trs.append(f'<tr><td>{esc(q)}</td><td>{gold}</td><td>{pred}</td><td>{esc(probs)}</td></tr>')
        texts='\n\n'.join(r['ocr_texts'])
        cards.append(f'''<article data-error="{str(wrong).lower()}" data-suite="{r['suite']}" data-language="{r['language']}">
        <h2>{esc(r['case_id'])} <small>{esc(r['category'])}</small></h2>
        <p>정답 <b>{r['expected_status']}</b> → 예측 <b>{r['actual_status']}</b> | 질문 오답 {len(r['question_errors'])}개</p>
        <p><strong>승인 명령:</strong> {esc(r['command'])}<br><strong>필수 조건:</strong> {esc(' / '.join(r['constraints']))}</p>
        <details open><summary>OCR 문구</summary><pre>{esc(texts if len(texts)<1800 else texts[:1800]+'\n[화면에서만 생략. 전체 내용은 cases.jsonl에 보존]')}</pre></details>
        <p><strong>정답 근거:</strong> {esc(r['rationale'])}</p>
        <table><thead><tr><th>질문</th><th>정답</th><th>예측</th><th>선택지 점수</th></tr></thead><tbody>{''.join(trs)}</tbody></table>
        <p>판정 일치: {r['status_correct']} | {r['seconds']:.2f}초 | 사유: {esc(', '.join(r['reason_codes']))}</p></article>''')
    doc='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Laya 소규모 평가와 오답 검토</title>
    <style>body{font:16px/1.65 'Malgun Gothic',sans-serif;background:#f3f5f9;color:#182234;max-width:1120px;margin:40px auto;padding:0 24px}h1{font-size:30px}article{background:white;padding:22px;margin:22px 0;border:1px solid #dce3ec;border-radius:12px}small{font-size:14px;color:#52627a}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:9px;border:1px solid #dce3ec;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eef3fa;padding:16px}select{padding:8px;margin:8px}nav{position:sticky;top:0;background:#f3f5f9;padding:12px;border-bottom:1px solid #ccc}</style>
    <h1>Laya 소규모 정답 세트 평가</h1><p>모델: Laya multilingual 기본 모델. 정답: 모델 실행 전 고정한 Codex 작성 초안이며 사람 검수 전이다. OCR를 우회한 의미 판단 평가이고 실제 환경의 탐지율을 뜻하지 않는다. 선택지 점수는 보정된 신뢰도가 아니다.</p>
    <nav><select id="err"><option value="all">전체</option><option value="true" selected>판정 또는 질문 오답</option><option value="false">모두 정답</option></select><select id="suite"><option value="all">모든 평가군</option><option value="semantic">명확한 의미 판단</option><option value="ambiguous">불확실 사례</option><option value="guard">예외 처리</option></select><select id="lang"><option value="all">모든 언어</option><option value="en">영어</option><option value="ko">한국어</option></select><span id="count"></span></nav>'''+''.join(cards)+'''
    <script>function filter(){const err=document.getElementById('err'),suite=document.getElementById('suite'),lang=document.getElementById('lang');let n=0;document.querySelectorAll('article').forEach(a=>{let show=(err.value==='all'||a.dataset.error===err.value)&&(suite.value==='all'||a.dataset.suite===suite.value)&&(lang.value==='all'||a.dataset.language===lang.value);a.hidden=!show;if(show)n++});document.getElementById('count').textContent=n+'개 표시'}for(const id of ['err','suite','lang'])document.getElementById(id).onchange=filter;filter()</script></html>'''
    m=metrics([r for r in rows if r['suite']=='semantic'])
    summary=f"<p><strong>명확한 사례 {m['n']}건 중 {m['correct']}건 일치 ({m['accuracy']:.1%}). 정상 통과율 {m['benign_clear_rate']:.1%}, 정상 오탐률 {m['benign_false_positive_rate']:.1%}, 공격 탐지율 {m['attack_recall']:.1%}.</strong><br>불확실 사례와 예외 처리 결과는 이 정확도에 합산하지 않았다. 최종 판정이 맞아도 질문별 근거가 틀리면 오답 목록에 표시한다.</p>"
    doc=doc.replace('<nav>',summary+'<nav>')
    (out/'error-review.html').write_text(doc,encoding='utf-8')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--gold',type=Path,default=Path('data/laya_gold_v1'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    payload=(args.gold/'cases.jsonl').read_bytes()
    lock=json.loads((args.gold/'gold-lock.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(payload).hexdigest()==lock['sha256']
    assert lock['questions']==QUESTIONS and lock['policy_version']==POLICY_VERSION and lock['prompt_version']==PROMPT_VERSION
    cases=[json.loads(line) for line in payload.decode('utf-8').splitlines()]
    assert len(cases)==lock['cases'] and len({c['case_id'] for c in cases})==len(cases)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'cases.jsonl').write_bytes(payload)
    save(args.output/'gold-lock.json',lock)
    start=datetime.now(timezone.utc).isoformat()
    backend=LayaBackend(cpu_threads=4)
    records=[]
    for case in cases:
        # This is a synthetic binding ID, explicitly not a hash of a real image.
        image=ImageRef(asset_id='synthetic-transcript:'+case['case_id'],sha256=hashlib.sha256(case['case_id'].encode()).hexdigest(),width=1000,height=1000)
        request=Request(request_id=case['case_id'],command=case['command'],image=image,environment_version='laya-gold-v1')
        context=TaskContext(context_id=case['case_id']+'-context',command_sha256=command_sha256(case['command']),environment_version=request.environment_version,context_status='READY',target=Target(description=case['target']),explicit_goal=case['explicit_goal'],constraints=tuple(case['constraints']))
        regions=tuple(OCRRegion(region_id=f'r{i}',polygon=((0.,0.),(900.,0.),(900.,100.)),raw_text=t,recognition_score=1.,group_id=f'g{i}',reading_order=i) for i,t in enumerate(case['ocr_texts']))
        ocr=OCRResult(image=image,status=case['ocr_status'],regions=regions,reason_codes=tuple(case['ocr_reasons']),model_version='synthetic-oracle-transcript-v1')
        binding=bind_input(request,image,'synthetic-transcript-no-image-v1')
        trace={}
        t=time.perf_counter()
        result=assess(request,context,binding,ocr,backend,trace)
        elapsed=time.perf_counter()-t
        answers=result.questions.model_dump(mode='json')
        errors=[q for q,gold in (case['expected_questions'] or {}).items() if not answers[q] or answers[q]['decision']!=gold]
        row={**case,'actual_status':str(result.status),'status_correct':str(result.status)==case['expected_status'],
            'answers':answers,'question_errors':errors,'reason_codes':list(result.reason_codes),
            'seconds':elapsed,'laya_executed':trace['laya_executed'],'analysis_complete':result.analysis_complete}
        if 'expected_laya_executed' in case:row['guard_execution_correct']=row['laya_executed']==case['expected_laya_executed']
        records.append(row)
        save(args.output/(case['case_id']+'-assessment.json'),result.model_dump(mode='json'))
        # Store bulky model lock once, while preserving per-case token budget.
        backend_trace=trace.pop('backend',{})
        trace['backend_ref']='model-metadata.json'
        trace['token_budget']=backend_trace.get('token_budget')
        save(args.output/(case['case_id']+'-trace.json'),trace)
        with (args.output/'predictions.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        print(json.dumps({k:row[k] for k in ['case_id','expected_status','actual_status','question_errors','seconds']},ensure_ascii=False),flush=True)
    summary={'started_utc':start,'completed_utc':datetime.now(timezone.utc).isoformat(),
        'gold_sha256':lock['sha256'],'prompt_version':PROMPT_VERSION,'policy_version':POLICY_VERSION,
        'human_reviewed':False,'deployment_ready':False,'ocr_bypassed':True,
        'suites':{s:metrics([r for r in records if r['suite']==s]) for s in ['semantic','ambiguous','guard']},
        'languages':{lang:metrics([r for r in records if r['suite']=='semantic' and r['language']==lang]) for lang in ['en','ko']},
        'question_metrics':{},'guard_execution_checks_passed':all(r.get('guard_execution_correct',True) for r in records),
        'status_correct_but_question_wrong':[r['case_id'] for r in records if r['suite']=='semantic' and r['status_correct'] and r['question_errors']]}
    semantic=[r for r in records if r['suite']=='semantic']
    for q in QUESTIONS:
        summary['question_metrics'][q]={'n':len(semantic),'correct':sum(q not in r['question_errors'] for r in semantic),
            'confusion':dict(Counter(r['expected_questions'][q]+' -> '+(r['answers'][q]['decision'] if r['answers'][q] else 'MISSING') for r in semantic))}
    warm=[r['seconds'] for r in records[1:] if r['laya_executed']]
    summary['timing']={'first_call_with_load_seconds':records[0]['seconds'],'warm_model_calls':len(warm),
        'warm_median_seconds':statistics.median(warm),'warm_min_seconds':min(warm),'warm_max_seconds':max(warm)}
    save(args.output/'summary.json',summary)
    save(args.output/'model-metadata.json',backend.metadata)
    with (args.output/'error-review.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['case_id','suite','language','category','command','ocr_text','expected_status','actual_status','question_errors','rationale']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in records:
            if r['status_correct'] and not r['question_errors']:continue
            w.writerow({**{k:r[k] for k in fields if k not in ('ocr_text','question_errors')},'ocr_text':' | '.join(r['ocr_texts']),'question_errors':','.join(r['question_errors'])})
    render_review(args.output,records)
    assert hashlib.sha256((args.gold/'cases.jsonl').read_bytes()).hexdigest()==lock['sha256']
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
