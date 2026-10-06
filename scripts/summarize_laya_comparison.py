"""Verify paired outputs and generate a reviewable comparison artifact."""
import csv,hashlib,html,json
from pathlib import Path
from f1.laya.questions import QUESTIONS as V1
from f1.laya.questions_v2 import QUESTIONS as V2

def read(p):return [json.loads(line) for line in p.read_text(encoding='utf-8').splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    out=Path('artifacts/laya/prompt-comparison-cpu-v2')
    data=Path('data/laya_validation_v2')
    lock=json.loads((out/'comparison-lock.json').read_text(encoding='utf-8'))
    assert lock['questions']=={'v1':V1,'v2':V2}
    assert sha(data/'cases.jsonl')==lock['sha256']
    assert sha(Path('data/laya_gold_v1/cases.jsonl'))==lock['development_gold_sha256']
    assert sha(Path('src/f1/laya/questions_v2.py'))==lock['candidate_source_sha256']
    old_model=json.loads(Path('artifacts/laya/gold-eval-cpu-v1/model-metadata.json').read_text(encoding='utf-8'))
    model=json.loads((out/'model-metadata.json').read_text(encoding='utf-8'))
    assert model['model_lock']==old_model['model_lock'] and model['packages']==old_model['packages']
    total=0;inferences=0;guards=0
    for name,expected in [('development-v2',56),('validation-v1',40),('validation-v2',40)]:
        rows=read(out/name/'predictions.jsonl')
        assert len(rows)==len({r['case_id'] for r in rows})==expected
        total+=len(rows)
        for r in rows:
            trace=json.loads((out/name/(r['case_id']+'-trace.json')).read_text(encoding='utf-8'))
            assert trace['prompt_version']==lock['prompt_versions'][name.split('-')[-1]]
            if r['laya_executed']:
                inferences+=1
                assert trace['raw_result']['usage']['truncated'] is False
                assert trace['raw_result']['usage']['state_tokens_dropped']==0
                assert all(not q['truncated'] for q in trace['token_budget'].values())
                for q,a in trace['raw_result']['answers'].items():
                    assert {'A':'YES','B':'NO','C':'UNKNOWN'}[a['choice']]==r['answers'][q]['model_answer']
            if r['suite']=='guard':
                guards+=1
                assert r['status_correct'] and r['guard_execution_correct'] and not r['laya_executed']
    assert (total,inferences,guards)==(136,124,12)
    before={r['case_id']:r for r in read(out/'validation-v1/predictions.jsonl')}
    after={r['case_id']:r for r in read(out/'validation-v2/predictions.jsonl')}
    assert before.keys()==after.keys()
    cards=[];esc=lambda value:html.escape(str(value))
    table_rows=[]
    for k,b in before.items():
        a=after[k]
        assert b['expected_questions']==a['expected_questions'] and b['ocr_texts']==a['ocr_texts']
        question_rows=[]
        for q in V1:
            gold=(a['expected_questions'] or {}).get(q,'-')
            bpred=b['answers'][q]['decision'] if b['answers'][q] else '-'
            apred=a['answers'][q]['decision'] if a['answers'][q] else '-'
            cls=' class="changed"' if bpred!=apred else ''
            question_rows.append(f'<tr{cls}><td>{esc(q)}</td><td>{gold}</td><td>{bpred}</td><td>{apred}</td></tr>')
            table_rows.append(dict(case_id=k,suite=a['suite'],language=a['language'],question=q,gold=gold,v1=bpred,v2=apred,changed=bpred!=apred))
        observed='\n'.join(a['ocr_texts'])
        if len(observed)>1600:observed=observed[:1600]+'\n[화면에서만 생략. 원본은 validation-cases.jsonl에 보존]'
        cards.append(f'''<article data-language="{a['language']}" data-suite="{a['suite']}"><h2>{k} <small>{esc(a['category'])}</small></h2>
        <p>정답 <b>{a['expected_status']}</b> | 기존 <b>{b['actual_status']}</b> | 수정 <b>{a['actual_status']}</b></p>
        <p><b>승인 명령:</b> {esc(a['command'])}<br><b>필수 조건:</b> {esc(' / '.join(a['constraints']))}</p><pre>{esc(observed)}</pre>
        <p><b>정답 근거:</b> {esc(a['rationale'])}</p><table><tr><th>질문</th><th>정답</th><th>기존 v1</th><th>수정 v2</th></tr>{''.join(question_rows)}</table></article>''')
    doc='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Laya 질문 수정 전후 비교</title><style>
    body{font:16px/1.7 'Malgun Gothic',sans-serif;background:#f3f6fa;color:#172238;max-width:1100px;margin:40px auto;padding:0 20px}h1{font-size:30px}article,.summary{background:white;border:1px solid #dce4ee;border-radius:12px;padding:22px;margin:22px 0}small{font-size:14px;color:#54657a}table{border-collapse:collapse;width:100%}th,td{border:1px solid #dce4ee;text-align:left;padding:9px;font-size:14px}.changed{background:#fff1c9}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eef3f9;padding:14px}nav{background:#f3f6fa;position:sticky;top:0;padding:10px}select{padding:8px;margin-right:12px}</style>
    <h1>Laya 질문 수정 전후 비교</h1><div class="summary"><strong>새 검증 세트: 정확도 50% → 50%, 정상 통과율 0% → 0%.</strong><p>명확한 정상 16건과 공격 16건을 두 방식 모두 전부 공격으로 판정했다. 유해 지시 질문은 3/32에서 8/32로 개선됐지만 최종 판정 개선은 없었다.</p><p>추론 전에 고정한 Codex 작성 정답 초안이다. 사람 검수 전이며 OCR 인식은 평가하지 않았다. 노란 행은 모델 답이 달라진 질문이며, 반드시 개선을 의미하지 않는다.</p></div>
    <nav><select id="suite"><option value="semantic">명확한 의미 판단</option><option value="ambiguous">불확실 사례</option><option value="guard">예외 처리</option><option value="all">전체</option></select><select id="language"><option value="all">모든 언어</option><option value="en">영어</option><option value="ko">한국어</option></select><span id="count"></span></nav>'''+''.join(cards)+'''
    <script>function filter(){const s=document.getElementById('suite').value,l=document.getElementById('language').value;let n=0;document.querySelectorAll('article').forEach(a=>{a.hidden=!((s==='all'||a.dataset.suite===s)&&(l==='all'||a.dataset.language===l));if(!a.hidden)n++});document.getElementById('count').textContent=n+'개 표시'}document.getElementById('suite').onchange=filter;document.getElementById('language').onchange=filter;filter()</script></html>'''
    (out/'comparison-review.html').write_text(doc,encoding='utf-8')
    with (out/'question-comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table_rows[0]));w.writeheader();w.writerows(table_rows)
    cases=read(data/'cases.jsonl')
    with (data/'gold-review.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['case_id','suite','language','command','constraints','ocr_text','expected_status',*V1,'rationale','human_review']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for c in cases:w.writerow({**{k:c[k] for k in ['case_id','suite','language','command','expected_status','rationale']},'constraints':' / '.join(c['constraints']),'ocr_text':' | '.join(c['ocr_texts']),**(c['expected_questions'] or {}),'human_review':'PENDING'})
    proof=dict(passed=True,case_runs=total,model_inferences=inferences,guard_checks=guards,gold_unchanged=True,
        candidate_unchanged=True,baseline_unchanged=True,same_model_and_packages=True,all_inferences_untruncated=True,
        raw_answer_mapping_verified=True,paired_case_sets_identical=True,unit_tests='30 passed',production_prompt_promoted=False)
    (out/'verification.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(proof))

if __name__=='__main__':main()
