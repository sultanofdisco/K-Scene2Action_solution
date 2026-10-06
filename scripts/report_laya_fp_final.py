"""Report every FP/attack tradeoff, including rejected candidates."""
import json,html,runpy
from pathlib import Path
ROOT=Path('artifacts/laya/fp-support-cpu-v1');PROFILE=Path('artifacts/laya/fp-profile-cpu-v1')
def read(p):return [json.loads(l) for l in p.read_text(encoding='utf8').splitlines()]
def pct(x):return '-' if x is None else f'{x:.1%}'
def esc(x):return html.escape(str(x))
def table(h,rs):return '<table><tr>'+''.join('<th>'+esc(x)+'</th>' for x in h)+'</tr>'+''.join('<tr>'+''.join('<td>'+esc(c)+'</td>' for c in r)+'</tr>' for r in rs)+'</table>'
def mtable(h,rs):return '| '+' | '.join(h)+' |\n| '+' | '.join(['---']*len(h))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rs)
def main():
    s=json.loads((ROOT/'native-summary.json').read_text());assert s['completed'];p=json.loads((PROFILE/'native-summary.json').read_text())
    bad=sum(s['sets'][n]['after']['confusion'].get('ATTACK -> CLEAR',0) for n in ['ripa_validation','legacy_validation','stress-v1','stress-v2'])
    ripa=s['sets']['ripa_validation']['after'];before=s['sets']['ripa_validation']['before'];clear=ripa['confusion'].get('CLEAR -> CLEAR',0)
    if bad:
        verdict=f'정상 오탐은 일부 줄었지만, 최종 후보도 검사한 공격 {bad}건을 정상 통과시켰다. 공격 탐지를 유지하는 실용적인 개선안은 아직 확보하지 못했다. 기본 모델에 적용하지 않았다.'
    else:
        verdict=f'RIPA 검증의 정상 {ripa["benign_n"]}건 중 {clear}건을 통과시키는 보수적 후보를 만들었다. FP는 {pct(before["benign_false_positive_rate"])}에서 {pct(ripa["benign_false_positive_rate"])}로 줄었다. 이번 공격 검증에서 추가 정상 통과는 없었지만, 정상 오탐은 여전히 높아 실용적인 해결로 보기 어렵다. 기본 모델은 교체하지 않았다.'
    headers=['평가 자료','방식','정상 FP','정상 통과','공격 탐지','공격 CLEAR','정확도'];rows=[]
    def add(title,label,m):
        c=m['confusion'];rows.append([title,label,f'{pct(m["benign_false_positive_rate"])} ({c.get("CLEAR -> ATTACK",0)}/{m["benign_n"]})',f'{c.get("CLEAR -> CLEAR",0)}/{m["benign_n"]}',f'{c.get("ATTACK -> ATTACK",0)}/{m["attack_n"]}',f'{c.get("ATTACK -> CLEAR",0)}/{m["attack_n"]}',pct(m['accuracy'])])
    for key,title in [('ripa_validation','RIPA 검증'),('legacy_validation','기존 검증'),('stress-v1','새 공격 진단 1'),('stress-v2','경계 공격 진단 2')]:
        add(title,'기존 native Laya',s['sets'][key]['before'])
        if key in ['ripa_validation','legacy_validation','stress-v1']:add(title,'정상 보정만 적용: 탈락',p['sets']['new-stress' if key=='stress-v1' else key]['metrics'])
        add(title,'학습 범위 검사 포함',s['sets'][key]['after'])
    body='<h1>F1-O FP 개선 실험 최종 결과</h1><aside><strong>'+esc(verdict)+'</strong></aside>'+table(headers,rows)
    explanation='Laya가 네 질문의 특징을 계산한다. 추가한 질문별 분류기가 모두 정상 답을 내고 점수 기준을 넘는지 확인한다. 학습 자료가 충분한 승인 작업 맥락인지, 특징이 학습 정상 사례에 가깝고 공격 사례와 구분되는지도 확인한다. 모든 조건을 만족한 경우에만 정상으로 보정한다. 그 외에는 원래 Laya의 답을 유지한다.'
    body+='<h2>바꾼 구조</h2><p>'+esc(explanation)+'</p><h2>큰 개선 수치를 채택하지 않은 이유</h2><p>정상 보정만 적용한 후보는 RIPA 정상 32/45를 통과시켰지만, 동결 후 작성한 다른 공격 12건 중 8건도 정상 통과시켰다. 이 후보는 탈락 처리했다. 학습 범위 검사로 보정을 줄이면 공격 누락을 줄일 수 있지만 정상 통과도 크게 줄었다.</p>'
    body+='<aside>기존 검증은 반복 확인한 개발 자료다. 새 진단 1은 추가 장치의 개발 계기가 되어 최종 후보의 독립 평가가 아니다. 진단 2는 최종 후보 동결 후 작성했지만 검증 정상 문구를 변형한 합성 자료라 외부 독립 벤치마크도 아니다. 실제 OCR 인식 성능은 포함되지 않았고 정답은 사람 검수 전이다.</aside><h2>개별 실제 추론 결과</h2><nav><select id="set"><option value="ripa_validation">RIPA 검증</option><option value="legacy_validation">기존 검증</option><option value="stress-v1">새 공격 진단 1</option><option value="stress-v2">경계 공격 진단 2</option></select><select id="kind"><option value="all">전체</option><option value="wrong">판정 오답</option><option value="repair">정상 보정 적용</option></select><span id="count"></span></nav>'
    for key in ['ripa_validation','legacy_validation','stress-v1','stress-v2']:
        for r in read(ROOT/(key+'-native.jsonl')):
            route=r['normal_repair'];body+=f'<article data-set="{key}" data-wrong="{str(not r["status_correct"]).lower()}" data-repair="{str(route.get("applied",False)).lower()}"><details><summary>{esc(r["case_id"])} | 정답 {r["expected_status"]} / 기존 {route.get("native_status","-")} / 최종 {r["actual_status"]}</summary><p>명령: {esc(r["command"])}</p><pre>{esc(chr(10).join(r["ocr_texts"]))}</pre>'+table(['질문','정답','판정'],[[q,(r.get('expected_questions') or {}).get(q,'-'),a['decision'] if a else '-'] for q,a in r['answers'].items()])+f'<pre>{esc(json.dumps(route,ensure_ascii=False,indent=2))}</pre></details></article>'
    body+='''<script>function filter(){let n=0;document.querySelectorAll('article').forEach(a=>{const k=document.getElementById('kind').value;const ok=a.dataset.set===document.getElementById('set').value&&(k==='all'||a.dataset[k]==='true');a.hidden=!ok;if(ok)n++});document.getElementById('count').textContent=n+'건'}for(let id of ['set','kind'])document.getElementById(id).onchange=filter;filter()</script>'''
    style=runpy.run_path('scripts/report_laya_ripa.py')['STYLE'];(ROOT/'comparison.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>F1-O FP 개선 최종 검토</title><style>'+style+'</style>'+body+'</html>',encoding='utf8')
    md='# F1-O FP 개선 결과와 남은 문제\n\n'+verdict+'\n\n## 실제 추론 비교\n\n'+mtable(headers,rows)
    md+='\nFP는 정상 자료를 ATTACK으로 판정한 비율이다. 정상 통과는 실제 CLEAR만 세며 UNKNOWN으로 옮긴 결과와 구분한다. 공격 CLEAR는 공격을 정상으로 놓친 경우다. 모든 최종 수치는 캐시 추정이 아니라 실제 Laya 파이프라인 실행 결과다.\n\n## 시도와 판단\n\n1. 기존 네 질문 공용 scorer 대신 질문별 선형 분류기 네 개를 학습했다. Laya 인코더와 판단층은 유지하고 같은 418건을 사용했다. 추가 학습 파라미터는 27,660개다.\n2. 학습 자료 내부 그룹 교차검증으로 정규화와 규제를 비교했다. 수렴한 후보에서 C=0.1을 고르고 학습 fold 안에서만 정규화 통계를 계산했다. 새 분류기를 직접 쓰면 기존 공격을 놓쳐 교체하지 않았다.\n3. 네 질문이 모두 정상이고 점수 기준을 넘을 때만 보정하도록 했다. 전체 맥락에 하나의 기준을 쓰면 정상 통과 개선이 없었다.\n4. 자료가 충분한 승인 작업 맥락에서 보정하니 RIPA FP가 28.9%로 낮아졌다. 그러나 새 공격 진단 12건 중 8건을 놓쳤다. 이 후보는 채택 불가다.\n5. 학습 정상 사례와 공격 사례에 대한 특징 거리를 추가로 검사해 근거가 부족한 정상 보정을 거부했다. 위 표의 마지막 후보가 이 방식이다. 공격 탐지와 FP를 함께 보고하며 큰 개선 수치만 채택하지 않았다.\n\n## 최종 후보 작동\n\n'+explanation+'\n\n승인 명령, 대상, 목표, 제약만으로 작업 맥락을 구분한다. OCR 내용, 사례 ID, 출처 또는 정답을 맥락 선택에 사용하지 않는다. 현재 지원 맥락은 빨간 공을 Zone 1, 파란 공을 Zone 2로 옮기며 칼과 렌치를 건드리지 않는 작업이다. 다른 명령이나 제약은 원래 Laya 판단을 유지하므로 다른 작업의 오탐은 해결하지 못했다.\n\n학습 범위 검사는 특징을 학습 통계로 표준화하고 L2 정규화한 뒤 가장 가까운 정상 및 공격 사례의 거리를 비교한다. 지지 점수 기준과 정상 거리 상한은 RIPA 학습 360건의 3분할 그룹 교차검증에서만 계산했다. 최종 검증 수치를 보고 임계값을 낮추지 않았다. 이 거리는 안전 확률이나 보장이 아니다.\n\n## 검증과 한계\n\n- 단위 및 기존 Laya 회귀 테스트 44건 통과. OCR 오류, 입력 및 모델 검사, 지원하지 않는 작업의 기존 판단 유지, 추가 장치의 보정 거부를 확인했다.\n- 실제 경로로 기존 검증 112건, 공격 진단 24건, 경계 진단 16건, OCR 예외 처리 8건을 실행했다.\n- 후보의 모델, 정책, 지원 특징과 구현 코드 해시를 마지막 진단 전 동결하고 종료 후 동일함을 확인했다.\n- 진단 1은 첫 후보 동결 후 만들었지만 그 실패를 보고 추가 장치를 설계했으므로 최종 후보의 독립 평가가 아니다. 진단 2는 최종 후보 동결 후 작성했고, 그 결과로 재학습하거나 기준을 바꾸지 않았다. 일부는 검증 정상 문구에서 파생되어 외부 독립 자료는 아니다.\n- RIPA 정상은 합성 대조 자료이고 템플릿을 공유한다. 정답은 사람 검수 전이며 실제 OCR 인식은 우회했다. 전체 서비스나 모든 작업에서 같은 FP 개선을 기대할 근거는 없다.\n\n## 다음에 필요한 변화\n\n현재 데이터와 동결된 특징에서는 정상 안내와 그 뒤에 붙은 실제 공격을 충분히 구분하지 못했다. 공용 scorer만 학습했을 때의 낮은 학습 적합도와 별개로, 질문별 분류기는 학습 자료를 맞혀도 새로운 공격에 일반화하지 못했다. 이를 데이터 양 하나의 문제로 단정할 수는 없다.\n\n다음 학습 자료는 정상 안내, 같은 안내에 공격 한 문장을 덧붙인 경우, 실제 인용만 있는 경우를 같은 가족으로 묶어 만들어야 한다. 사람 검수 후 가족 전체를 학습과 평가로 분리해야 한다. 추가로 Laya 판단층 또는 인코더 일부까지 학습할지 비교할 필요가 있다. 이번에는 진단 결과를 학습 자료로 옮기거나 테스트에 맞춰 기준을 조정하지 않았다.\n\n## 구현과 산출물\n\n운영 기본 backend와 F1-T는 바꾸지 않았다. 새로운 연결은 실험용으로 명시적으로 선택한다.\n\n```python\nfrom f1.laya.question_head import SupportRepairLayaBackend, assess_profile_repair\n\nbackend = SupportRepairLayaBackend(\n    "artifacts/laya/fp-support-cpu-v1", cpu_threads=4\n)\nresult = assess_profile_repair(request, context, binding, ocr, backend)\n```\n\n- 최종 비교 및 오답: `artifacts/laya/fp-support-cpu-v1/comparison.html`\n- 최종 모델과 기준: 같은 폴더의 `question-head.safetensors`, `fp-policy.json`, `support.safetensors`, `support-lock.json`\n- 실제 추론 기록: `native-summary.json`, 각 자료의 `*-native.jsonl`\n- 탈락한 정상 보정 후보: `artifacts/laya/fp-profile-cpu-v1/`\n- 분류기 대조 실험: `artifacts/laya/fp-question-head-cpu-v2/`\n- 구현: `src/f1/laya/question_head.py`\n- 테스트: `tests/test_laya_question_head.py`\n'
    Path('docs/laya-fp-improvement.md').write_text(md,encoding='utf8')
    # Clearly mark the prior optimistic candidate as rejected, preserving its raw predictions.
    prior_html=PROFILE/'comparison.html';text=prior_html.read_text(encoding='utf8')
    warning='<aside><strong>채택 불가: RIPA FP는 줄었지만 새 공격 12건 중 8건을 정상 통과시켰다. 후속 결과는 fp-support-cpu-v1/comparison.html에 있다.</strong></aside>'
    if '채택 불가: RIPA FP' not in text:prior_html.write_text(text.replace('<h1>',warning+'<h1>',1),encoding='utf8')
    print(json.dumps({'verdict':verdict,'report':str(ROOT/'comparison.html'),'docs':'docs/laya-fp-improvement.md','extra_attack_clear_count':bad},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
