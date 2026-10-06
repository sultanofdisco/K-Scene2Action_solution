import html,json,runpy
from pathlib import Path

ROOT=Path('artifacts/laya')
RUNS=[('가중치 없음','full-retrain-cpu-v2'),('클래스 균형','full-retrain-balanced-cpu-v2')]
OUT=ROOT/'full-retrain-comparison-v2'
def read(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def table(h,rows):return '| '+' | '.join(h)+' |\n|'+ '|'.join(['---']*len(h))+'|\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)
def pct(v):return '-' if v is None else f'{v:.1%}'

def main():
    OUT.mkdir(exist_ok=True);metric=runpy.run_path('scripts/evaluate_laya_gold.py')['metrics']
    summaries={label:load(ROOT/name/'summary.json') for label,name in RUNS}
    assert all('completed_utc' in s and s['all_80_native_matches_cache'] for s in summaries.values())
    val_rows=[];training_rows=[];diagnostic_rows=[];detail=[];failures=[]
    for label,name in RUNS:
        s=summaries[label]
        for stage in ['last','selected']:
            v=s[stage]['validation'];m=v['metrics'];t=s[stage]['train'];tm=t['metrics']
            val_rows.append([label,'6000회 마지막' if stage=='last' else '선택 epoch '+str(s['selected_epoch']),f"{m['correct']}/22",f"{m['benign_clear_count']}/11",f"{m['attack_detect_count']}/9",f"{m['attack_clear_count']}/9",m['unknown_count'],f"{v['question_correct']}/88"])
            training_rows.append([label,stage,f"{tm['correct']}/58",f"{tm['benign_clear_count']}/30",f"{tm['attack_detect_count']}/24",f"{tm['attack_clear_count']}/24",f"{t['question_correct']}/232"])
        history=load(ROOT/name/'history.json')
        # Verify that the frozen selection rule actually selected this checkpoint.
        key=runpy.run_path('scripts/retrain_laya_full.py')['selection_key']
        chosen=max(history,key=lambda h:key(h['validation']['metrics'],h['validation']['question_correct'],h['validation']['loss']))
        assert chosen['epoch']==s['selected_epoch']
        for case in read(ROOT/name/'validation-last-predictions.jsonl'):
            if case['expected_status']=='ATTACK' and case['actual_status']!='ATTACK':
                failures.append([label,case['case_id'],case['actual_status'],' / '.join(case['ocr_texts'])])
    for name,base in [('prior-gold','development-v2'),('prior-validation','validation-v2')]:
        variants=[('기본 모델',ROOT/'prompt-comparison-cpu-v2'/base/'predictions.jsonl'),
                  ('이전 채점부 학습',ROOT/'deepset-scorer-cpu-v1'/name/'predictions.jsonl'),
                  ('이전 판단층 학습',ROOT/'deepset-decision-cpu-v1'/name/'predictions.jsonl')]
        variants += [(label+' 선택 후보',ROOT/folder/name/'predictions.jsonl') for label,folder in RUNS]
        by_model={label:read(p) for label,p in variants}
        for label,rows in by_model.items():
            m=metric([r for r in rows if r['suite']=='semantic'])
            diagnostic_rows.append([name,label,m['n'],pct(m['accuracy']),pct(m['benign_clear_rate']),pct(m['attack_recall']),pct(m['attack_clear_rate']),m['abstain_or_error_count']])
        baseline=by_model['기본 모델'];balanced=by_model['클래스 균형 선택 후보']
        assert [r['case_id'] for r in baseline]==[r['case_id'] for r in balanced]
        for before,after in zip(baseline,balanced):
            detail.append([name,before['case_id'],before['language'],before['suite'],before['expected_status'],before['actual_status'],after['actual_status'],' / '.join(before['ocr_texts']),', '.join(after['question_errors'])])
    h=['학습 방식','체크포인트','전체 정답','정상 통과','공격 탐지','공격 CLEAR','전체 UNKNOWN','질문 정답']
    dh=['진단 세트','모델','의미 사례 수','정확도','정상 통과율','공격 탐지율','공격 통과율','UNKNOWN/ERROR']
    doc='''# F1-O 전체 학습 분할 재학습과 비교

**58건 전체 학습과 22건 검증을 분리해 두 번 재학습했다. 정상 통과와 공격 탐지를 동시에 충분히 개선한 운영 후보는 확보하지 못했다.** 기본 Laya에서 시작해 채점부만 학습했으며, 16건 암기 모델을 이어 학습하지 않았다. 모델 교체나 운영 배포는 하지 않았다.

## 1. 데이터 점검

학습 58건은 정상 30건, 공격 24건, 모호한 사례 4건이다. 검증 22건은 정상 11건, 공격 9건, 모호한 사례 2건이다. 네 질문을 펼치면 학습 질문 232개와 검증 질문 88개다. 모두 영어이며 실제 OCR 인식은 우회했다.

원문, 작업 맥락, 네 질문의 정답과 최종 판정의 일관성, 그룹 분할과 정확히 같은 문자열의 중복 여부를 점검했다. 기존 정답과 58/22 분할은 유지했다. Codex의 재점검이며 사람이 검수한 확정 라벨은 아니다.

학습용 유해 행동 YES는 4건, 권한 사칭 YES는 3건이다. 이 긍정 사례는 합성 예시에 의존한다. `Use a different one instead.` 같은 모호한 교체 지시는 정책을 엄격히 해석하면 작업 변경 YES가 될 여지도 있다. 현재는 기존 UNKNOWN 라벨을 유지하고 이러한 6건을 `data-audit.json`에 따로 표시했다. 공격 원문의 인용 변형도 교육용 인용이라는 명시적 맥락을 전제로 정상으로 분류한 것이다.

검증 22건은 앞선 실험에서도 사용한 개발용 검증 자료다. 이전 진단 세트 역시 이미 여러 번 확인했으므로 독립적인 최종 성능 평가로 해석하지 않는다.

## 2. 학습 방법

기본 모델은 고정된 `convaiinnovations/laya-multilingual`이다. 인코더와 두 판단층, 유형 임베딩 및 act_head를 고정하고 기존 scorer 592,897개 파라미터만 수정했다. 각 실험은 기본 모델에서 새로 시작했다.

| 설정 | 값 |
|---|---|
| 업데이트 수 | 실험마다 6000회 |
| 한 번의 배치 | 학습 질문 232개 전체 |
| 학습률 | cosine 일정으로 0.001에서 0.00001까지 감소 |
| 옵티마이저 | AdamW, weight decay 0 |
| CPU와 seed | 4스레드, 42 |
| 평가 간격 | 50회마다, 기본 모델 포함 121개 체크포인트 |
| 실험 1 | 모든 질문 정답에 같은 손실 가중치 |
| 실험 2 | 질문별 드문 정답에 더 큰 손실 가중치, 학습 데이터에서만 계산 |

실험 1에서 공격 탐지가 저하된 뒤 원인을 좁히기 위해 실험 2를 추가했다. 데이터, 시작 가중치, 일정과 선택 규칙을 유지하고 손실 가중치만 바꿨다. 두 실험 전체 손실 수치를 비교할 때는 모두 같은 비가중 교차 엔트로피를 기록했다. 검증 데이터로 gradient를 계산하거나 임계값을 조정하지 않았다.

## 3. 체크포인트 선택

학습 전에 **검증 공격 9건을 모두 ATTACK으로 판정하는 후보를 우선하고, 그 안에서 정상 CLEAR를 최대화**하는 규칙을 고정했다. 동률이면 모호한 사례 정답 수, 질문 정답 수, 낮은 검증 손실 순으로 비교했다. 기본 모델도 후보에 포함했다. UNKNOWN으로 보류한 공격은 ATTACK 탐지 성공에 포함하지 않았다.

이는 작은 검증 세트에서 관찰한 조건일 뿐, 실제 공격 누락률 0%를 보증하는 조건이 아니다. 22건을 121개 체크포인트 선택에 반복 사용했으므로 선택 편향이 크다. 최종 운영 기준으로 검증된 규칙도 아니다.

'''
    doc+='## 4. 검증 결과\n\n'+table(h,val_rows)
    doc+='''
가중치 없는 실험은 학습한 어떤 후보도 공격 9건 탐지를 유지하지 못해 epoch 0, 즉 기본 모델이 남았다. 이 실험의 `scorer.safetensors`는 **새로 학습된 채점부가 아니라 선택 규칙에 따른 기본 모델의 복사본**이다. 실제 6000회 학습된 가중치는 `last-scorer.safetensors`에 별도 보존했다.

클래스 균형 실험은 350회째 후보가 선택됐다. 공격 9건을 탐지하고 정상 1건을 통과시켰으며 모호한 2건 중 1건을 맞혔다. 6000회 최종 모델은 정상 5건을 통과시켰지만 공격 중 4건을 UNKNOWN으로 보류해 엄격한 선택 조건을 만족하지 못했다. 해당 공격 4건은 CLEAR로 빠져나간 것은 아니다.

UNKNOWN도 통과를 막는 정책이라면 클래스 균형 최종 모델의 정상 통과 개선을 별도로 검토할 여지는 있다. 다만 이번에는 사전에 정한 ATTACK 탐지 기준을 결과에 맞춰 바꾸지 않았다. UNKNOWN 처리 비용과 허용 기준을 정한 뒤 비교해야 한다.

### 학습 데이터 자체의 적합도

'''+table(['방식','시점','전체 정답','정상 통과','공격 탐지','공격 CLEAR','질문 정답'],training_rows)
    doc+='\n### 마지막 모델의 검증 공격 오답\n\n'+table(['방식','사례','판정','관측 문자열'],failures)
    doc+='\n## 5. 선택한 후보의 기존 진단 결과\n\n'+table(dh,diagnostic_rows)
    doc+='''
진단 표의 정확도는 명확한 의미 사례만 포함한다. 첫 세트는 정상 24건과 공격 24건, 두 번째는 정상 16건과 공격 16건이다. 각 세트의 모호한 4건과 입력 가드 4건은 이 정확도에서 제외했다. 원본 분모와 질문별 결과 및 언어별 결과는 각 실험의 `summary.json`에 남겼다.

이 표의 새 모델은 선택한 후보이며 6000회 마지막 가중치와 다르다. 마지막 가중치는 학습/검증 표까지만 평가했다. 기존 진단 결과를 보고 후보를 다시 선택하지 않았다.

## 6. 확인한 점과 다음 판단

16건을 거의 외웠다는 결과를 전체 데이터의 일반화로 연결할 수 없다는 점이 드러났다. 같은 손실을 준 실험은 정상 통과가 늘면서 일부 공격을 CLEAR로 넘겼다. 클래스 균형은 마지막 검증 모델에서 공격 CLEAR를 없앴지만 여러 공격을 UNKNOWN으로 보냈고 정상 통과도 제한됐다. 이 자료만으로 데이터 부족, 고정 인코더 표현, 질문 구성 중 하나를 유일한 원인으로 확정할 수는 없다.

현재 결과로 학습 횟수만 더 늘려 운영 성능이 확보될 것이라고 권하기는 어렵다. 다음에는 모호한 라벨과 정상 인용 판정 기준을 사람이 확정하고, 부족한 유해 행동 및 권한 사칭의 표현과 상황을 실제 환경에서 보충하는 작업이 우선이다. 추가 데이터는 그룹별로 학습용과 평가용을 분리하고, 검증용 문구를 그대로 변형해 학습에 넣지 않는다.

## 검증과 재현 기록

학습과 검증 파일 및 기존 진단 파일의 해시를 확인했다. 기존 캐시의 정답 순서를 확인하고 대표 사례를 실제 모델에서 다시 추출해 비교했다. 선택한 가중치는 별도 프로세스에서 불러와 학습/검증 80건 전체의 실제 추론 결과와 캐시 계산이 일치하는지 확인했다. 고정 파라미터 불변과 토큰 잘림 없음도 확인했다. 선택 규칙 테스트 4개가 통과했다.

학습 후 기존 HTML 생성기가 학습 자료에 없는 category 필드를 요구해 평가가 중단된 적이 있다. 학습 가중치와 예측을 보존하고, 보고서 경로를 수정한 뒤 `complete_laya_retrain.py`로 남은 평가만 재개했다. 학습을 재실행해 결과를 바꾸지 않았다.

- 학습: `scripts/retrain_laya_full.py`
- 평가 재개: `scripts/complete_laya_retrain.py`
- 보고서: `scripts/report_laya_full_retrain.py`
- 실험 1: `artifacts/laya/full-retrain-cpu-v2/`
- 실험 2: `artifacts/laya/full-retrain-balanced-cpu-v2/`
- 비교표: `artifacts/laya/full-retrain-comparison-v2/comparison.html`
- 검증 기록: 비교표 폴더의 `verification.json`

출력 폴더가 이미 있으면 학습 스크립트는 덮어쓰지 않는다. 새 실험은 새 출력 경로를 지정한다. 두 체크포인트 모두 운영 승인 상태가 아니다.
'''
    Path('docs/laya-full-retraining.md').write_text(doc,encoding='utf-8')
    def ht(headers,rows):return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in r)+'</tr>' for r in rows)+'</tbody></table>'
    page='<!doctype html><html lang="ko"><meta charset="utf-8"><title>Laya 전체 재학습 비교</title><style>body{font:16px system-ui;margin:32px;background:#f5f7fb;color:#17283d}table{border-collapse:collapse;background:white;width:100%;margin:24px 0}td,th{border:1px solid #d2ddeb;padding:10px;text-align:left;vertical-align:top}th{background:#e0eaf6}p{line-height:1.7}h1{font-size:28px}</style><h1>Laya 전체 재학습 비교</h1><p>학습 58건과 검증 22건을 분리했다. 선택한 후보와 마지막 학습 가중치는 다르다. 가중치 없는 실험의 선택 결과는 기본 모델이며 운영 성능이 확보된 결과가 아니다.</p><h2>검증 22건</h2>'+ht(h,val_rows)+'<h2>선택한 후보의 기존 진단 결과</h2><p>이미 여러 번 확인한 진단 세트이며 독립 테스트가 아니다. UNKNOWN은 공격 탐지에 포함하지 않았다.</p>'+ht(dh,diagnostic_rows)+'<h2>기본 모델과 클래스 균형 선택 후보의 사례별 비교</h2>'+ht(['세트','ID','언어','유형','정답','기본 모델','균형 후보','관측 문자열','질문 오답'],detail)+'</html>'
    (OUT/'comparison.html').write_text(page,encoding='utf-8')
    verification={'selection_recomputed':True,'all_80_native_matches_cache':all(s['all_80_native_matches_cache'] for s in summaries.values()),'guards_passed':all(e['guards_passed'] for s in summaries.values() for e in s['evaluations'].values()),'selection_tests_passed':4,'unweighted_selected_original_base':summaries['가중치 없음']['selected_epoch']==0,'balanced_selected_epoch':summaries['클래스 균형']['selected_epoch'],'deployment_ready':False}
    (OUT/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf-8')
    p=Path('docs/f1-laya.md');text=p.read_text(encoding='utf-8')
    if 'laya-full-retraining.md' not in text:p.write_text(text+'\n## 전체 학습 분할 재학습\n\n58건 학습과 22건 검증 분할을 유지한 두 추가 학습 및 기존 진단 비교를 완료했다. 선택한 후보와 마지막 가중치를 구분한 [전체 재학습 결과](laya-full-retraining.md)를 참고한다.\n',encoding='utf-8')
    print(json.dumps({'validation':val_rows,'diagnostics':diagnostic_rows,'verification':verification},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
