"""Summarize the frozen RIPA experiment without changing model selection."""
import html,json,runpy
from pathlib import Path
from collections import Counter

ROOT=Path('artifacts/laya/ripa-augmented-cpu-v1')
DATA=Path('data/laya_ripa_domain_v1')
def read(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
def esc(x):return html.escape(str(x))
def pct(x):return '-' if x is None else f'{x:.1%}'
STYLE="""body{font:16px/1.7 'Malgun Gothic',sans-serif;background:#eef2f7;color:#172339;max-width:1200px;margin:32px auto;padding:0 24px}h1{font-size:30px}h2{margin-top:36px}table{border-collapse:collapse;width:100%;background:white}th,td{border:1px solid #d5dfea;padding:10px;text-align:left}article,aside{background:white;border:1px solid #d5dfea;border-radius:12px;padding:18px;margin:14px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f6fb;padding:14px;font-size:14px}small{color:#54647a}select{padding:9px;margin:8px}nav{position:sticky;top:0;background:#eef2f7;padding:8px}summary{cursor:pointer}a{color:#185fc3}.bad{color:#9e301b}.good{color:#18663b}"""
def page(title,body):return f'<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(title)}</title><style>{STYLE}</style>{body}</html>'
def table(headers,rows):return '<table><thead><tr>'+''.join(f'<th>{esc(h)}</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{esc(c)}</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table>'
def render_data():
    rows=read(DATA/'ripa_train.jsonl')+read(DATA/'ripa_validation.jsonl');cards=[]
    for r in rows:
        cards.append(f'<article data-split="{r["split"]}" data-label="{r["expected_status"]}"><details><summary>{esc(r["case_id"])} | {r["split"]} | {r["expected_status"]} | {esc(r["origin"])}</summary><p>위험 목표: {esc(r["hazard_id"])} / 유형: {esc(r["attack_subtype"])}</p><p>승인 명령: {esc(r["command"])}</p><pre>{esc(chr(10).join(r["ocr_texts"]))}</pre><p>정답: {esc(r["expected_questions"])}</p><p>근거: {esc(r["rationale"])}</p><p>출처: {esc(r["source_record"])}</p></details></article>')
    body='<h1>RIPA 학습 자료 검토</h1><aside>원본 공격 225건과 합성 정상 225건. 학습 360건, 검증 90건이다. 기존 학습 58건은 별도 결합한다. 정답은 Codex 작성 초안으로 사람 검수 전이다. OCR를 실행한 자료가 아닌 외부 관찰 텍스트 실험이다. SA03, SA08, SA14는 검증에만 배정했다.</aside><nav><select id="split"><option value="all">전체 분할</option><option>ripa_train</option><option>ripa_validation</option></select><select id="label"><option value="all">전체 정답</option><option>ATTACK</option><option>CLEAR</option></select><span id="count"></span></nav>'+''.join(cards)
    body+='''<script>function filter(){let n=0;document.querySelectorAll('article').forEach(a=>{let ok=['split','label'].every(k=>document.getElementById(k).value==='all'||a.dataset[k]===document.getElementById(k).value);a.hidden=!ok;if(ok)n++});document.getElementById('count').textContent=n+'건'}for(let id of ['split','label'])document.getElementById(id).onchange=filter;filter()</script>'''
    (DATA/'review.html').write_text(page('RIPA 데이터 검토',body),encoding='utf-8')

def main():
    render_data()
    s=json.loads((ROOT/'summary.json').read_text(encoding='utf-8'));assert s['completed_utc']
    metric=runpy.run_path('scripts/evaluate_laya_gold.py')['metrics']
    names={'legacy_only':'기존 58건 대조군','ripa_augmented':'RIPA 결합 418건'}
    sets={'legacy_validation':'기존 검증 22건','ripa_validation':'RIPA 검증 90건'}
    selected_ripa=s['arms']['ripa_augmented']['selected']['ripa_validation']['metrics']
    conclusion=(f'RIPA 결합 모델의 선택 체크포인트는 RIPA 검증에서 공격 탐지 {pct(selected_ripa["attack_recall"])}, '
                f'정상 통과 {pct(selected_ripa["benign_clear_rate"])}, 정확도 {pct(selected_ripa["accuracy"])}였다. ')
    if selected_ripa['benign_clear_rate']==0:
        conclusion+='정상 통과 문제가 해결되지 않아 실용적인 성능 개선을 확인하지 못했다. 공격 탐지만으로 학습 성공이라고 평가하지 않는다.'
    table_rows=[]
    def add(name,split,m):
        attack_unknown=m['confusion'].get('ATTACK -> UNKNOWN',0)
        table_rows.append([name,split,pct(m['accuracy']),f'{pct(m["attack_recall"])} ({m["confusion"].get("ATTACK -> ATTACK",0)}/{m["attack_n"]})',f'{pct(m["benign_clear_rate"])} ({m["confusion"].get("CLEAR -> CLEAR",0)}/{m["benign_n"]})',f'{m["confusion"].get("ATTACK -> CLEAR",0)}/{m["attack_n"]}',f'{attack_unknown}/{m["attack_n"]}'])
    for split,label in sets.items():
        add('미학습 기본 모델',label,s['baseline'][split]['metrics'])
        for arm in names:add(names[arm],label,s['arms'][arm]['selected'][split]['metrics'])
    headers=['모델','평가 자료','판정 정확도','공격 탐지','정상 통과','공격 CLEAR','공격 UNKNOWN']
    selected_rows=list(table_rows)
    table_rows=[]
    for arm in names:
        for split,label in sets.items():add(names[arm]+' 마지막 6,000회',label,s['arms'][arm]['last'][split]['metrics'])
    last_rows=table_rows
    table_rows=selected_rows
    fit_rows=[]
    for arm,label in names.items():
        trainset='legacy_train' if arm=='legacy_only' else 'augmented_train'
        for checkpoint,title in [('selected','선택 체크포인트'),('last','마지막 6,000회')]:
            m=s['arms'][arm][checkpoint][trainset]
            fit_rows.append([label,title,pct(m['metrics']['accuracy']),pct(m['question_accuracy'])])
    prior_rows=[]
    for arm,label in names.items():
        for key,title in [('prior-gold','기존 진단 48건'),('prior-validation','기존 진단 32건')]:
            m=s['arms'][arm]['prior_diagnosis'][key]['metrics']
            prior_rows.append([label,title,pct(m['accuracy']),pct(m['attack_recall']),pct(m['benign_clear_rate']),str(m['confusion'].get('ATTACK -> CLEAR',0)),str(m['confusion'].get('ATTACK -> UNKNOWN',0))])
    body='<h1>RIPA 추가 학습 결과</h1><aside><strong>'+esc(conclusion)+'</strong></aside><p>두 실험 모두 원래 Laya multilingual에서 시작해 기존 scorer 592,897개 파라미터만 학습했다. 6,000회 업데이트, CPU 4스레드, 질문 행 256개 미니배치, 질문별 정답 클래스 가중치를 사용했다. 아래 표는 검증 기준으로 선택한 체크포인트 결과다. 기본 모델도 선택 후보에 포함했다.</p>'
    body+='<p>'+esc(' / '.join(f'{names[a]} 선택: {v["selected_step"]}회' for a,v in s['arms'].items()))+'</p><h2>모델 선택에 사용한 검증</h2>'+table(headers,table_rows)
    body+='<h2>마지막 업데이트와 학습 적합도</h2><p>선택 시점이 0회이면 학습은 완료했지만 선택 기준을 충족한 학습 모델이 없어 초기 가중치가 남았다는 뜻이다. 아래 마지막 가중치 결과도 함께 확인해야 추가 학습 효과를 판단할 수 있다.</p>'+table(headers,last_rows)+table(['모델','시점','학습 판정 정확도','학습 질문 정확도'],fit_rows)
    body+='<h2>선택 완료 후 기존 진단</h2>'+table(headers,prior_rows)+'<p>기존 진단은 명확한 의미 판단 80건이다. 두 세트의 불확실 사례와 예외 처리 각 8건은 위 정확도에서 제외했다. 이 진단 세트는 과거에 여러 번 확인한 합성 자료이므로 독립 테스트라고 부르지 않는다.</p>'
    body+='<h2>해석 범위</h2><p>RIPA 공격 180건과 합성 정상 180건을 기존 58건에 추가했다. RIPA 검증은 세 위험 목표 전체를 분리한 90건이다. 문구 템플릿과 역할 표현은 학습과 공유하고, 정답은 사람 검수 전이다. 실제 이미지 OCR를 거친 성능이나 새로운 환경에 대한 성능을 뜻하지 않는다. 모델은 실험용 체크포인트로 보관하며 서비스 기본 모델을 교체하지 않았다.</p>'
    body+='<h2>RIPA 검증 세부 결과</h2>'
    detail=[];diagnosis={}
    for arm,label in names.items():
        rows=read(ROOT/arm/'ripa_validation-predictions.jsonl');diagnosis[arm]={}
        for field in ['origin','hazard_id','attack_subtype']:
            groups={v:metric([r for r in rows if r[field]==v]) for v in sorted({r[field] for r in rows})};diagnosis[arm][field]=groups
            for v,m in groups.items():detail.append([label,field,v,m['n'],pct(m['accuracy']),pct(m['attack_recall']),pct(m['benign_clear_rate'])])
    body+=table(['모델','기준','그룹','건수','정확도','공격 탐지','정상 통과'],detail)
    normal=[r for r in read(ROOT/'ripa_augmented/ripa_validation-predictions.jsonl') if r['expected_status']=='CLEAR']
    question_counts={q:dict(Counter(r['answers'][q]['decision'] for r in normal)) for q in normal[0]['answers']}
    qrows=[[q,len(normal),v.get('YES',0),v.get('NO',0),v.get('UNKNOWN',0)] for q,v in question_counts.items()]
    diagnosis['ripa_augmented']['normal_question_counts']=question_counts
    body+='<h2>정상 문구가 막히는 위치</h2>'+table(['질문','정상 건수','YES','NO','UNKNOWN'],qrows)+'<p>앞의 세 질문 중 하나라도 YES면 최종 ATTACK이다. 네 번째 질문에서 정상 관찰을 인식해도 앞의 오탐을 취소하지 않는다. 질문별 오탐과 학습 적합도를 먼저 확인해야 하며, 이 결과만으로 데이터가 쓸모없거나 Laya 자체가 불가능하다고 단정할 수 없다.</p>'
    body+='<h2>선택한 모델의 개별 검토</h2><nav><select id="arm"><option value="ripa_augmented">RIPA 결합</option><option value="legacy_only">기존 대조군</option></select><select id="split"><option value="all">모든 검증 및 진단</option>'+''.join(f'<option value="{k}">{v}</option>' for k,v in {**sets,'prior-gold':'기존 진단 48건','prior-validation':'기존 진단 32건'}.items())+'</select><select id="errors"><option value="wrong">판정 또는 질문 오답</option><option value="all">전체</option></select><span id="count"></span></nav>'
    for arm in names:
        for split in [*sets,'prior-gold','prior-validation']:
            for r in read(ROOT/arm/(split+'-predictions.jsonl')):
                if r['suite']=='guard':continue
                bad=not r['status_correct'] or bool(r['question_errors'])
                qs=[[q,(r.get('expected_questions') or {}).get(q,'-'),a['decision'] if a else '-'] for q,a in r['answers'].items()]
                body+=f'<article data-arm="{arm}" data-split="{split}" data-wrong="{str(bad).lower()}"><details><summary>{esc(r["case_id"])} | {r["expected_status"]} → {r["actual_status"]} | 질문 오답 {len(r["question_errors"])}</summary><p>승인 명령: {esc(r["command"])}</p><pre>{esc(chr(10).join(r["ocr_texts"]))}</pre>'+table(['질문','정답','예측'],qs)+f'<p>{esc(r["rationale"])}</p></details></article>'
    body+='''<script>function filter(){let n=0;document.querySelectorAll('article').forEach(a=>{let ok=a.dataset.arm===document.getElementById('arm').value&&(document.getElementById('split').value==='all'||a.dataset.split===document.getElementById('split').value)&&(document.getElementById('errors').value==='all'||a.dataset.wrong==='true');a.hidden=!ok;if(ok)n++});document.getElementById('count').textContent=n+'건'}for(let id of ['arm','split','errors'])document.getElementById(id).onchange=filter;filter()</script>'''
    (ROOT/'comparison.html').write_text(page('Laya RIPA 추가 학습 비교',body),encoding='utf-8')
    (ROOT/'subgroup-diagnosis.json').write_text(json.dumps(diagnosis,ensure_ascii=False,indent=2),encoding='utf-8')
    md='# Laya RIPA 추가 학습 결과\n\n'+conclusion+'\n\nRIPA 공격을 기존 F1-O 학습 데이터에 결합하고, 같은 학습 조건의 기존 데이터 대조군과 비교했다. F1-T와 운영 모델은 바꾸지 않았다.\n\n'
    md+='## 학습 내용\n\n- 기존 58건 + RIPA 공격 180건 + 합성 정상 180건 = 418건.\n- 검증은 기존 22건과 RIPA 90건으로 분리했다.\n- RIPA 225개 원본 공격 중 SA03, SA08, SA14의 45건은 검증 전용이다. SA16과 SA17은 제외했다.\n- 정상은 안전 안내, 공격 인용, 점검 기록으로 생성한 합성 자료다. 정답은 사람 검수 전이다.\n- 원래 Laya multilingual의 scorer 592,897개 파라미터만 학습했다. 인코더와 decision head는 동결했다.\n- AdamW, 6,000회 업데이트, 질문 행 256개 미니배치, seed 42, CPU 4스레드. 학습률은 0.001에서 0.00001까지 코사인 감소한다.\n- 질문별 클래스 빈도 가중치를 각 실험의 학습 데이터에서 계산했다. 기존 데이터 대조군에도 동일한 알고리즘을 적용했다.\n\n'
    md+='## 모델 선택\n\n100회마다 두 검증 집합을 각각 평가했다. 두 집합 중 낮은 공격 탐지율을 먼저 높이고, 그 값이 같으면 공격 CLEAR 비율을 낮추고 정상 통과율의 집합별 평균을 높이는 순서로 선택했다. UNKNOWN은 공격 탐지 성공이 아니다. 초기 기본 모델도 후보에 포함했고, 기존 진단 결과로 다시 선택하지 않았다.\n\n'
    for a,v in s['arms'].items():md+=f'- {names[a]}: {v["selected_step"]}회 체크포인트 선택. 동결 가중치 유지 {v["frozen_parameters_unchanged"]}, 전체 파이프라인 재실행 대조 {v["native_replay_checks_passed"]}건 통과.\n'
    def mdtable(head,rows):return '| '+' | '.join(head)+' |\n| '+' | '.join(['---']*len(head))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)
    md+='\n## 검증 결과\n\n'+mdtable(headers,table_rows)+'\n## 선택 후 기존 진단\n\n'+mdtable(headers,prior_rows)
    md+='\n## 마지막 업데이트와 학습 적합도\n\n선택 시점이 0회이면 6,000회 학습을 수행했어도 선택 기준을 충족한 학습 가중치가 없어서 초기 모델을 선택했다는 뜻이다. 마지막 가중치는 아래와 같다. 학습 적합도는 일반화 성능이 아니다.\n\n'+mdtable(headers,last_rows)+'\n'+mdtable(['모델','시점','학습 판정 정확도','학습 질문 정확도'],fit_rows)
    md+='\n## 정상 문구가 막히는 위치\n\n선택된 RIPA 모델이 정상 검증 45건에 답한 분포다. 앞의 세 질문 중 하나라도 YES면 공격이 된다. 네 번째 질문에서 정상 문구를 인식하더라도 앞의 오탐을 취소하지 않는다.\n\n'+mdtable(['질문','정상 건수','YES','NO','UNKNOWN'],qrows)+'\n현재 실험만으로 실패 원인을 데이터 양 하나로 단정할 수 없다. 다음 진단은 데이터를 더 모으기 전에 작업 변경 및 권한 사칭 질문에서 정상 안내와 인용문의 정답을 학습할 수 있는지, 동결 특징과 scorer 학습 방식에 문제가 없는지 확인하는 것이다. 이번 결과를 보고 정답이나 검증 분할을 바꾸거나 후속 학습을 추가하지 않았다.\n'
    md+='\n## 해석의 한계\n\nRIPA 검증은 위험 목표를 분리했지만 문구 템플릿과 출처 역할은 공유한다. 새 환경의 독립 평가가 아니다. 이전 F1-T 학습에 관련 로컬 위험 목표가 포함된 적도 있다. 모든 수치는 원본 외부 관찰 문구를 직접 주입한 Laya 의미 판별 결과이며 OCR 성능이 포함되지 않았다. 기존 진단은 이전에도 사용한 합성 자료로, 모델 선택 후 회귀 확인에만 썼다. 검증에 반복적으로 선택한 결과이므로 별도 실환경 평가가 필요하다.\n\n서비스 기본 모델은 교체하지 않았다. 두 체크포인트와 마지막 업데이트 가중치를 모두 보관했다. 마지막 가중치 결과는 `summary.json`의 `last`, 선택한 가중치 결과는 `selected`에 있다.\n\n## 파일\n\n- 데이터: `data/laya_ripa_domain_v1/`\n- 전체 비교와 오답: `artifacts/laya/ripa-augmented-cpu-v1/comparison.html`\n- 선택된 RIPA 모델: `artifacts/laya/ripa-augmented-cpu-v1/ripa_augmented/scorer.safetensors`\n- 선택 모델 잠금: 같은 폴더의 `scorer-lock.json`\n- 수치 및 학습 계획: 실험 폴더의 `summary.json`, `training-plan.json`\n- 재현: `prepare_laya_ripa.py` → `cache_laya_ripa.py` → `train_laya_ripa.py` → `report_laya_ripa.py`. 기존 결과 덮어쓰기는 막혀 있다.\n'
    Path('docs/laya-ripa-training.md').write_text(md,encoding='utf-8')
    print(json.dumps({'report':str(ROOT/'comparison.html'),'documentation':'docs/laya-ripa-training.md','selected_steps':{a:v['selected_step'] for a,v in s['arms'].items()}},ensure_ascii=False))

if __name__=='__main__':main()
