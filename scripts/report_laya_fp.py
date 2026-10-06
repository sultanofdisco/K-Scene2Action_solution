"""Report native results and repair preliminary cached answer records."""
import json,html,runpy,hashlib
from pathlib import Path
from safetensors.torch import load_file
from f1.laya.question_head import QuestionReadout
from f1.laya.pipeline import parse_answer
from f1.laya.questions_v2 import QUESTIONS

ROOT=Path('artifacts/laya/fp-profile-cpu-v1');OLD=Path('artifacts/laya/ripa-augmented-cpu-v1')
def read(p):return [json.loads(l) for l in p.read_text(encoding='utf8').splitlines()]
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def pct(x):return '-' if x is None else f'{x:.1%}'
def esc(x):return html.escape(str(x))
def mdtable(h,rs):return '| '+' | '.join(h)+' |\n| '+' | '.join(['---']*len(h))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rs)
def table(h,rs):return '<table><tr>'+''.join('<th>'+esc(x)+'</th>' for x in h)+'</tr>'+''.join('<tr>'+''.join('<td>'+esc(c)+'</td>' for c in r)+'</tr>' for r in rs)+'</table>'
def main():
    summary=json.loads((ROOT/'native-summary.json').read_text());assert summary['completed'];policy=json.loads((ROOT/'fp-policy.json').read_text())
    # Preliminary status calculations were correct; now ensure every answer record matches that status.
    head=QuestionReadout(ROOT,json.loads(Path('artifacts/laya/model/model-lock.json').read_text()))
    for name in ['legacy_train','ripa_train','legacy_validation','ripa_validation']:
        p=ROOT/(name+'-predictions.jsonl');rows=read(p)
        features=load_file(str(OLD/(name+'-features.safetensors') if name.startswith('ripa') else Path('artifacts/laya/deepset-scorer-cpu-v1')/(('train' if name=='legacy_train' else 'validation')+'-features.safetensors')))['features']
        for i,r in enumerate(rows):
            if r['normal_repair_applied']:
                r['answers']={q:parse_answer(a).model_dump(mode='json') for q,a in head.answers(features[i*4:i*4+4]).items()}
                r['question_errors']=[q for q in QUESTIONS if r['answers'][q]['decision']!=r['expected_questions'][q]]
            d=[r['answers'][q]['decision'] for q in QUESTIONS];actual='ATTACK' if 'YES' in d[:3] else 'CLEAR' if d==['NO','NO','NO','YES'] else 'UNKNOWN'
            assert actual==r['actual_status']
        p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf8')
    headers=['자료','모델','정상 오탐 FP','정상 통과','공격 탐지','공격 CLEAR','정확도'];trs=[]
    prelim=json.loads((ROOT/'preliminary-summary.json').read_text())
    for name,title in [('ripa_validation','RIPA 검증'),('legacy_validation','기존 검증')]:
        for label,m in [('기존 native Laya',prelim['evaluations'][name]['before']),('작업 맥락 보정',summary['sets'][name]['metrics'])]:
            conf=m['confusion'];trs.append([title,label,f'{pct(m["benign_false_positive_rate"])} ({conf.get("CLEAR -> ATTACK",0)}/{m["benign_n"]})',f'{conf.get("CLEAR -> CLEAR",0)}/{m["benign_n"]}',f'{conf.get("ATTACK -> ATTACK",0)}/{m["attack_n"]}',f'{conf.get("ATTACK -> CLEAR",0)}/{m["attack_n"]}',pct(m['accuracy'])])
    stress=summary['sets']['new-stress']['metrics'];diag=[]
    for name,title in [('new-stress','동결 후 새 합성 진단'),('prior-gold','기존 진단 48건'),('prior-validation','기존 진단 32건')]:
        m=summary['sets'][name]['metrics'];cf=m['confusion'];diag.append([title,pct(m['benign_false_positive_rate']),f'{cf.get("CLEAR -> CLEAR",0)}/{m["benign_n"]}',f'{cf.get("ATTACK -> ATTACK",0)}/{m["attack_n"]}',f'{cf.get("ATTACK -> CLEAR",0)}/{m["attack_n"]}',pct(m['accuracy'])])
    threshold=next(iter(policy['profiles'].values()))['threshold']
    intro='현재 공 정렬 작업의 RIPA 검증에서 정상 오탐을 45/45에서 13/45로 줄였다. 정상 32/45를 통과시키며 공격 45/45 탐지를 유지했다. 다른 작업 맥락의 정상 오탐은 아직 해결하지 못했다.'
    process='Laya가 승인 작업과 OCR 텍스트를 읽음 → 질문별 특징 정규화 → 네 개의 작은 분류기가 각각 YES/NO/UNKNOWN 판단 → 지원하는 작업 맥락에서 네 답이 NO/NO/NO/YES이고 점수 기준도 충족하면 정상으로 보정 → 나머지는 원래 Laya의 답을 유지'
    md='# F1-O 정상 오탐 개선 실험\n\n'+intro+'\n\n## 변경 내용\n\n'+process+'\n\nLaya 인코더와 기존 판단층을 유지했다. 같은 418건을 사용해 질문별 선형 분류기 네 개를 학습했다. 학습한 추가 파라미터는 27,660개이며 별도 LLM 호출은 없다. 질문별 세 선택지 특징을 연결해 2,304차원 입력으로 사용한다. 정규화 통계는 학습 자료에서만 계산한다.\n\n단순히 분류기로 기존 모델을 교체하는 대안은 정상 통과가 늘었지만 기존 공격을 정상으로 통과시키기도 했다. 전체 작업에 공통으로 높은 기준을 적용한 대안은 정상 통과 개선이 없어, 자료가 충분한 승인 작업 맥락에 한정한 보정을 후보로 구현했다. 운영 기본 모델은 바꾸지 않았다.\n\n## 결과\n\n'+mdtable(headers,trs)
    md+='\n정상 오탐 FP는 정상 자료를 ATTACK으로 판정한 비율이다. UNKNOWN으로 옮기는 것과 실제 CLEAR 통과를 구분한다. 위 최종 결과에서는 정상의 UNKNOWN은 없었다. 공격 CLEAR와 공격 UNKNOWN은 탐지 성공으로 세지 않는다.\n\n## 새 진단과 기존 기능 확인\n\n'+mdtable(['자료','정상 FP','정상 통과','공격 탐지','공격 CLEAR','정확도'],diag)
    md+='\n새 진단은 모델과 기준값을 동결한 뒤 만든 영어 합성 24건이다. 안전 안내 뒤에 유해 지시를 추가한 혼합 공격, 권한 사칭, 목적지 변경, 정상 인용 및 안내를 포함한다. 이 결과를 보고 모델이나 기준을 재조정하지 않았다. 사람이 확인한 외부 독립 테스트는 아니다. 기존 진단의 OCR 예외 처리 8건도 실제 경로에서 확인했다.\n\n## 적용 범위\n\n승인된 명령, 대상, 목표, 제약 전체를 정규화한 JSON 해시로 작업 맥락을 구분한다. OCR 문구나 정답, 자료 출처, 사례 ID는 이 선택에 사용하지 않는다. 작업 맥락이 같다는 이유만으로 통과시키지 않으며, 네 질문의 정상 답과 점수 기준을 모두 충족해야 한다.\n\n현재 지원하는 맥락은 빨간 공을 Zone 1, 파란 공을 Zone 2로 옮기고 칼과 렌치를 건드리지 않는 작업이다. 같은 명령이라도 제약이나 목표가 바뀌면 기존 Laya로 돌아간다. 다른 명령의 오탐은 여전히 남아 있으므로 일반적인 F1-O 오탐 문제가 해결됐다고 발표하면 안 된다.\n\n## 학습과 기준 결정\n\n- 학습 418건과 기존 분할을 유지했다. 정답을 바꾸거나 새 검증 사례를 학습에 넣지 않았다.\n- 학습 그룹만 이용한 3분할 교차검증으로 C를 비교했다. 정규화도 각 학습 fold에서만 계산했다. 수렴한 후보 중 C=0.1을 사용했다.\n- Laya 원래 scorer 재학습과 달리 네 질문의 분류기를 분리했다. 최적화 방법과 손실 설정도 달라졌으므로 개선 원인을 분리 하나에만 귀속하지 않는다.\n- 수렴하지 않은 후보 C=10과 정규화 없는 후보도 결과를 보관했다. 정상 통과가 높아 보여도 운영 후보로 채택하지 않았다.\n- 보정 대상 맥락은 학습 정상 20건 이상, 비정상 20건 이상, 그룹 6개 이상을 요구했다. 실제 활성 맥락에는 정상 180건, 공격 180건, 위험 목표 12개가 있다.\n'
    md+=f'- 학습 fold 밖에서 얻은 각 사례의 네 정상 선택지 점수 중 최솟값을 계산했다. 해당 맥락의 공격 및 UNKNOWN 최대값에 0.01을 더해 기준 {threshold:.6f}을 고정했다. 이 수치는 보정된 안전 확률이 아니다.\n'
    md+='- C와 기준값의 수치 계산에는 기존 검증 112건을 쓰지 않았다. 다만 이 검증을 여러 실험에서 반복 확인했으므로 개발 검증 결과로 해석한다.\n\n## 한계와 다음 작업\n\nRIPA의 정상 45건은 합성 대조 자료이며 공격과 공유하는 템플릿이 있다. 모든 정답은 사람 검수 전이다. OCR를 직접 실행한 평가가 아니라 전사 텍스트를 입력한 의미 판별 평가다. 공격을 이번 표에서 놓치지 않았다는 것이 미래 공격을 놓치지 않는다는 보장은 아니다. 다음은 실제 환경의 정상 표지와 혼합 공격을 별도로 모아 이 후보를 동결한 상태로 확인하는 것이다. 다른 작업은 별도 자료와 검증으로 지원 범위를 넓혀야 한다.\n\n## 사용 및 파일\n\n새 후보는 명시적으로 선택하는 실험용 backend로 연결했다. 기존 요청, 작업 맥락, OCR 결과, 무결성 바인딩을 그대로 전달한다.\n\n```python\nfrom f1.laya.question_head import ProfileRepairLayaBackend, assess_profile_repair\n\nbackend = ProfileRepairLayaBackend(\n    "artifacts/laya/fp-profile-cpu-v1", cpu_threads=4\n)\nresult = assess_profile_repair(request, context, binding, ocr, backend)\n```\n\n- 모델과 정책: `artifacts/laya/fp-profile-cpu-v1/`\n- 비교 및 오답: 같은 폴더의 `comparison.html`\n- 실제 경로 결과: `native-summary.json`, 각 자료의 `*-native.jsonl`\n- 신규 진단과 동결 기록: `stress-cases.jsonl`, `stress-lock.json`, `pre-stress-freeze.json`\n- 구현: `src/f1/laya/question_head.py`\n- 연결 및 정책 검사: `tests/test_laya_question_head.py`\n\n단위 및 기존 Laya 테스트 37건이 통과했다. 실제 검증 112건의 최종 판정은 캐시 계산과 전부 일치했다. 모델과 정책 해시는 신규 진단 전후 동일했다.\n'
    Path('docs/laya-fp-improvement.md').write_text(md,encoding='utf8')
    style=runpy.run_path('scripts/report_laya_ripa.py')['STYLE']
    body='<h1>F1-O 정상 오탐 개선</h1><aside><strong>'+esc(intro)+'</strong></aside><p>'+esc(process)+'</p><h2>실제 Laya 경로 검증</h2>'+table(headers,trs)+'<h2>추가 진단</h2>'+table(['자료','정상 FP','정상 통과','공격 탐지','공격 CLEAR','정확도'],diag)
    body+='<aside>현재 공 정렬 작업 맥락에만 보정을 적용한다. 다른 명령이나 제약이면 기존 Laya 판단을 유지한다. 위 수치는 사람이 검수하지 않은 합성 정상 자료와 원본 공격을 사용한 텍스트 판단 실험이다. 실제 OCR 성능이나 모든 작업의 정확도를 뜻하지 않는다. 운영 기본 모델은 교체하지 않았다.</aside>'
    body+='<h2>개별 결과</h2><nav><select id="set"><option value="ripa_validation">RIPA 검증</option><option value="legacy_validation">기존 검증</option><option value="new-stress">새 진단</option><option value="prior-gold">기존 진단 48건</option><option value="prior-validation">기존 진단 32건</option></select><select id="error"><option value="all">전체</option><option value="true">판정 오답</option></select><span id="count"></span></nav>'
    for name in summary['sets']:
        for r in read(ROOT/(name+'-native.jsonl')):
            if r['suite']=='guard':continue
            body+=f'<article data-set="{name}" data-error="{str(not r["status_correct"]).lower()}"><details><summary>{esc(r["case_id"])} | {r["expected_status"]} → {r["actual_status"]} | 정상 보정 {r["normal_repair"].get("applied",False)}</summary><p>승인 명령: {esc(r["command"])}</p><pre>{esc(chr(10).join(r["ocr_texts"]))}</pre>'+table(['질문','정답','실제 답'],[[q,(r.get('expected_questions') or {}).get(q,'-'),a['decision'] if a else '-'] for q,a in r['answers'].items()])+f'<p>보정 정보: {esc(r["normal_repair"])}</p></details></article>'
    body+='''<script>function filter(){let n=0;document.querySelectorAll('article').forEach(a=>{const ok=a.dataset.set===document.getElementById('set').value&&(document.getElementById('error').value==='all'||a.dataset.error==='true');a.hidden=!ok;if(ok)n++});document.getElementById('count').textContent=n+'건'}for(let k of ['set','error'])document.getElementById(k).onchange=filter;filter()</script>'''
    (ROOT/'comparison.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>Laya FP 개선</title><style>'+style+'</style>'+body+'</html>',encoding='utf8')
    print(json.dumps({'report':str(ROOT/'comparison.html'),'docs':'docs/laya-fp-improvement.md','stress':stress},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
