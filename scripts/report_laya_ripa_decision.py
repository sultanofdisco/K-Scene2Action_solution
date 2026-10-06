"""Build evidence-based report after native evaluation completes."""
import json, html
from pathlib import Path

ROOT = Path('artifacts/laya/ripa-decision-cpu-v1')
def read(name): return json.loads((ROOT/name).read_text(encoding='utf8'))
def cell(m, source, target): return m['confusion'].get(f'{source} -> {target}', 0)
def values(m):
    b,a=m['benign_n'],m['attack_n']
    return [f"{cell(m,'CLEAR','CLEAR')}/{b}",f"{cell(m,'CLEAR','ATTACK')}/{b}",f"{cell(m,'CLEAR','UNKNOWN')}/{b}",f"{cell(m,'ATTACK','ATTACK')}/{a}",f"{cell(m,'ATTACK','CLEAR')}/{a}",f"{cell(m,'ATTACK','UNKNOWN')}/{a}"]
def main():
    train=read('training-summary.json'); native=read('native-summary.json'); history=read('history.json'); assert native['completed']
    old=json.loads(Path('artifacts/laya/ripa-augmented-cpu-v1/summary.json').read_text())['arms']['ripa_augmented']
    heads=['모델과 평가 집합','정상 통과','정상 오탐','정상 보류','공격 탐지','공격 통과','공격 보류']
    rows=[]
    for name,label in [('ripa_validation','RIPA 검증'),('legacy_validation','기존 검증')]:
        rows.append([f'이전 scorer 단독: {label}',*values(old['selected'][name]['metrics'])])
        rows.append([f'이번 판단층 + scorer: {label}',*values(native['sets'][name]['metrics'])])
    for name in ['stress-v1','stress-v2']:
        rows.append([f'이번 모델: {name}',*values(native['sets'][name]['metrics'])])
    table='| '+' | '.join(heads)+' |\n| '+ ' | '.join(['---']*len(heads))+' |\n'+'\n'.join('| '+' | '.join(row)+' |' for row in rows)
    epochs='| Epoch | 업데이트 | 학습 상태 정확도 | RIPA 상태 정확도 | 기존 검증 상태 정확도 | 평균 검증 CE |\n| --- | --- | --- | --- | --- | --- |\n'
    for h in history:
        epochs+=f"| {h['epoch']} | {h['updates']} | {h['train']['metrics']['accuracy']:.1%} | {h['ripa_validation']['metrics']['accuracy']:.1%} | {h['legacy_validation']['metrics']['accuracy']:.1%} | {h['macro_validation_ce']:.4f} |\n"
    qtable='| 집합 | 질문 | 정답 / 전체 |\n| --- | --- | --- |\n'
    for name in ['legacy_validation','ripa_validation']:
        cases=[json.loads(l) for l in (ROOT/(name+'-native.jsonl')).read_text(encoding='utf8').splitlines()]
        for q in cases[0]['expected_questions']:
            correct=sum(r['answers'][q]['decision']==r['expected_questions'][q] for r in cases)
            qtable+=f'| {name} | {q} | {correct}/{len(cases)} |\n'
    source=train['plan']['source']; commit=source.get('commit',source.get('commit_sha','8a6e1328cce2460a0e5aa348ad465bb1b5821cd2'))
    url=f'https://github.com/NandhaKishorM/laya/blob/{commit}/notebooks/laya_finetune_typed_decisions_mps.py'
    rm=native['sets']['ripa_validation']['metrics']; lm=native['sets']['legacy_validation']['metrics']
    leaks=sum(cell(native['sets'][s]['metrics'],'ATTACK','CLEAR') for s in ['ripa_validation','legacy_validation','stress-v1','stress-v2'])
    conclusion=f"선택된 모델은 epoch {train['selected_epoch']}이다. RIPA 검증의 정상 {rm['benign_n']}건 중 {cell(rm,'CLEAR','CLEAR')}건을 통과시키고 {cell(rm,'CLEAR','ATTACK')}건을 공격으로 오탐했다. 공격 {rm['attack_n']}건 중 {cell(rm,'ATTACK','ATTACK')}건을 탐지하고 {cell(rm,'ATTACK','CLEAR')}건을 통과시켰다. 기존 검증의 정상 통과는 {cell(lm,'CLEAR','CLEAR')}/{lm['benign_n']}건이다."
    warning=f"검증과 진단 집합에서 공격을 CLEAR로 통과시킨 결과가 합계 {leaks}건 있다. 이 합계는 서로 다른 합성 집합을 합친 참고 수치이며 독립 표본의 성능 추정치가 아니다." if leaks else '이번 검증과 진단 집합에서 공격의 CLEAR 통과는 관측되지 않았다. 이는 미지의 공격에 대한 안전성 보장을 의미하지 않는다.'
    md=f'''# RIPA 포함 Laya 판단층 학습 결과

{conclusion}

{warning} 운영 기본값으로 승격하지 않은 실험 모델이다.

## 공식 학습 방식과 이번 적용

[Laya 공식 학습 스크립트]({url})의 RLCD와 cross entropy를 함께 사용하는 방법을 참고했다. RLCD는 선택지 점수에 잡음을 넣어 여러 답 분포를 만들고, 정답에 가까운 분포에 더 높은 보상을 주도록 학습 신호를 계산한다. CE는 정답 선택지의 점수를 높인다. 새로운 별도 분류기를 덧붙이지 않고 Laya의 기존 판단층과 scorer를 갱신했다.

공식 스크립트는 encoder도 학습할 수 있지만 이번에는 요청한 판단층 범위에 맞춰 encoder, type embedding, act_head를 고정했다. 원본 다국어 Laya에서 시작했으며 이전 scorer 학습 결과나 FP 보정기를 이어받지 않았다. 원본 encoder 출력을 float32로 저장하고 동일한 출력으로 판단층을 학습한 뒤, 평가에서는 원본 encoder와 새 판단층을 다시 연결했다.

## 데이터와 학습량

- 학습 418건: 기존 데이터 58건 + RIPA 공격 180건 + 합성 정상 대조군 180건.
- 각 입력에 질문 4개를 적용해 정답 선택지 1,672개를 학습했다. 유해 지시, 작업 변경, 권한 사칭, 정상 관찰 포함 여부를 각각 YES/NO/UNKNOWN으로 판단한다.
- 검증 112건: RIPA 공격 45건과 합성 정상 45건, 기존 검증 22건. 이 데이터는 기울기 계산에 사용하지 않았으나 체크포인트 선택에 사용했다.
- RIPA는 기존 위험 시나리오 단위 분할을 유지했다. SA03, SA08, SA14는 검증에 남겼다. 기존 제외 항목도 그대로 유지했다.
- 입력은 영어로 정리한 OCR 정답 텍스트다. 실제 OCR 오류가 섞인 이미지 기반 종단 평가나 OCR 모델 학습은 아니다.
- 학습 대상: Transformer 판단층 2개와 scorer, 총 14,768,641개 파라미터.
- CPU 4스레드, 4 epochs, 질문 행 batch 8, 총 {train['updates']}회 optimizer 업데이트. 각 epoch에서 모든 학습 행을 한 번씩 사용했다.
- AdamW, learning rate 0.0001에서 0.000001로 cosine 감소, weight decay 0.01, gradient clipping 1.0, seed 42.
- RLCD 잡음 표본 4개, sigma 0.4에서 0.1로 감소. 클래스 가중치는 사용하지 않았다.
- 체크포인트 선택: 두 검증 집합의 질문 단위 CE를 동일 비중으로 평균한 값의 최솟값. 선택 epoch {train['selected_epoch']}. 선택 이후 스트레스 진단으로 재선정하지 않았다.
- 온도 보정은 하지 않았다. 출력 점수를 보정된 안전 확률로 해석하면 안 된다.

## 이전 학습과 결과 비교

{table}

오탐은 정상 입력을 ATTACK으로 판정한 경우만 뜻한다. UNKNOWN 보류도 정상 작업을 진행하지 못하게 하므로 정상 통과와 별도로 함께 보고한다. 공격 탐지율을 높이면서 모든 정상 입력도 막는 모델은 성공으로 보지 않는다.

이전 scorer 모델은 같은 418건으로 6,000회 업데이트한 뒤 300회 체크포인트를 선택한 결과다. 이번에는 학습 범위뿐 아니라 손실, 학습률, 샘플링과 선택 기준도 달라졌다. 결과 차이를 판단층 학습만의 효과로 단정할 수 없다.

## 학습 경과

{epochs}

## 결과 해석과 후속 판단

학습 집합의 최종 상태 정확도는 {train['selected']['train']['metrics']['accuracy']:.1%}, 질문별 정확도는 {train['selected']['train']['question_accuracy']:.1%}이다. 학습 집합에서도 오답이 상당하므로, 검증 집합에 대한 일반화 문제만으로 이번 실패를 설명할 수 없다. 다만 이 한 번의 실행으로 데이터 라벨, 학습 방식, encoder 표현 중 무엇이 원인인지 확정할 수는 없다.

{qtable}

검증 CE를 낮추는 것과 최종 안전 판정을 잘하는 것은 같은 목표가 아니다. 이번에 선택된 체크포인트도 실제 공격 통과 여부를 별도로 봐야 한다. 후속 실험에서는 검수한 작은 대조 세트에서 CE 단독과 RLCD+CE를 동일 조건으로 비교하고, 질문별 오답과 최종 통과 조건의 불일치를 먼저 진단하는 것이 타당하다. 학습 횟수나 데이터를 무조건 늘리는 것으로 해결된다고 가정하지 않는다.

## 검증과 해석의 한계

- 선택 모델의 실제 Laya 경로에서 검증 112건의 모든 질문 판정과 최종 상태가 캐시 기반 평가와 일치했다.
- 원본 encoder 파라미터의 해시가 복원 후에도 일치하고, 저장한 판단층 가중치가 정확히 다시 로드됨을 확인했다. 고정된 type embedding과 act_head도 유지했다.
- 입력 길이 등 guard 사례 8건을 확인했다. 학습된 모델은 입력 검증이나 결과 통합 규칙을 변경하지 않았다.
- 새 학습 손실과 고정 입력의 gradient 관련 테스트 3개, 기존 판단층 로더와 입력 검사 관련 테스트 35개가 통과했다.
- stress-v1 24건과 stress-v2 16건은 이전 실험에서 이미 살펴본 합성 진단이다. 새 독립 시험이나 일반화 성능으로 제시하면 안 된다. 일부 stress-v2는 RIPA 검증 정상 문구에서 파생됐다.
- 합성 정상 데이터는 사람의 최종 검수가 완료되지 않았다. 위험 시나리오를 분리했어도 문구 템플릿이나 승인 작업이 겹칠 수 있다.
- 실제 환경의 정상 표지판과 OCR 오류가 있는 입력, 새로운 공격을 별도 수집해 확인해야 운영 적합성을 판단할 수 있다. 현재 보고서는 이 모델을 자동 배포할 근거가 아니다.

## 산출물

- 선택 모델: `artifacts/laya/ripa-decision-cpu-v1/decision.safetensors`
- 마지막 epoch: `artifacts/laya/ripa-decision-cpu-v1/last-decision.safetensors`
- 학습 설정과 기록: `training-plan.json`, `history.json`, `training-summary.json`
- 실제 모델 경로의 평가: `native-summary.json`, 각 `*-native.jsonl`
- 원본 코드와 고정 버전: `upstream/source-lock.json`
- 실행 코드: `scripts/cache_laya_ripa_decision.py`, `scripts/train_laya_ripa_decision.py`, `scripts/evaluate_laya_ripa_decision.py`
'''
    assert not any(ch in md for ch in '\u00b7\u318d\u30fb')
    Path('docs/laya-ripa-decision-training.md').write_text(md,encoding='utf8')
    th=''.join('<th>'+html.escape(h)+'</th>' for h in heads)
    tr=''.join('<tr>'+''.join('<td>'+html.escape(c)+'</td>' for c in row)+'</tr>' for row in rows)
    details=[]
    for name in ['legacy_validation','ripa_validation','stress-v1','stress-v2']:
        cases=[json.loads(l) for l in (ROOT/(name+'-native.jsonl')).read_text(encoding='utf8').splitlines()]
        errors=[r for r in cases if not r['status_correct']]
        details.append(f'<h2>{name}: 상태 오답 {len(errors)}건</h2>')
        for r in errors:
            details.append('<details><summary>'+html.escape(f"{r['case_id']} | {r['expected_status']} → {r['actual_status']}")+'</summary><pre>'+html.escape(json.dumps(r,ensure_ascii=False,indent=2))+'</pre></details>')
    page='<!doctype html><html lang="ko"><meta charset="utf-8"><title>Laya RIPA 판단층 학습 비교</title><style>body{font:16px/1.65 system-ui;margin:40px auto;max-width:1300px;padding:0 24px;color:#213047;background:#f5f7fb}h1{color:#142c4e}table{border-collapse:collapse;width:100%;background:white}th,td{padding:13px;border:1px solid #d6dfea;text-align:left}th{background:#e7effc}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:16px}summary{cursor:pointer}details{margin:8px 0;padding:10px;background:#eaf0f7}.note{background:#fff0d7;padding:18px}</style><h1>RIPA 포함 Laya 판단층 학습 결과</h1><p>'+html.escape(conclusion)+'</p><p class="note">'+html.escape(warning)+' 운영 기본값은 변경하지 않았습니다.</p><p>학습 418건 / 검증 112건 / 판단층과 scorer 14,768,641개 파라미터 / 4 epochs / '+str(train['updates'])+' updates</p><table><thead><tr>'+th+'</tr></thead><tbody>'+tr+'</tbody></table><p>분수는 해당 정답 유형 전체 건수 대비 결과입니다. UNKNOWN은 통과가 아닙니다. 스트레스 세트는 이전에 검토한 합성 진단이며 독립 테스트가 아닙니다.</p><p><a href="../../../../docs/laya-ripa-decision-training.md">상세 학습 보고서</a> | <a href="'+url+'">공식 학습 스크립트 고정 버전</a></p>'+''.join(details)+'</html>'
    page=page.replace('../../../../docs/laya-ripa-decision-training.md','../../../docs/laya-ripa-decision-training.md')
    (ROOT/'comparison.html').write_text(page,encoding='utf8')
    print(json.dumps({'report':str(ROOT/'comparison.html'),'selected_epoch':train['selected_epoch'],'rows':rows},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
