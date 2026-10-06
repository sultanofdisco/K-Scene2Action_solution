# SafeAgentBench 학습 연결과 분할

> 2026-10-05 후속 학습 완료: CPU 30 epoch 학습 후 검증 정확도 87.1%, 위험 누락 3/7, 정상 오차단 1/24. 아래 준비 단계 설명은 당시 기록이며, 현재 결과는 [학습 결과 문서](f1t-safeagentbench-training.md)를 기준으로 확인한다.

2026-10-05. 선별된 텍스트 후보 429개를 분할하고 기존 다국어 CLIP 텍스트 보조학습 코드에 연결했다. 이는 이미지 없는 별도 보조 분류기의 준비이며, 이미지+텍스트 F1-T의 가중치를 변경하지 않았다.

## 현재 결과

| 분할 | 정상 | 위험 | 합계 |
|---|---:|---:|---:|
| 학습 | 200 | 166 | 366 |
| 검증 | 24 | 7 | 31 |
| 시험 | 23 | 9 | 32 |
| 합계 | 247 | 182 | 429 |

선별 파일의 원문과 라벨은 그대로 유지했다. 학습에는 SafeAgentBench만 사용하도록 분리했으며 SafeText나 RoboJailBench와 아직 합치지 않았다. 자료를 추가한 효과를 확인하려면 향후 같은 평가셋에서 학습 자료만 바꾼 대조 실험이 필요하다. 이전 SafeText 검증 정확도 50%와 이번 분할의 정확도를 직접 비교해 개선 폭이라고 해석하면 안 된다.

## 분할 기준

같은 원본 장면, 원본이 연결한 정상/위험 비교 명령, 정규화한 동일 문장은 한 묶음으로 관리한다. 여기에 문자 TF-IDF 유사도 0.90 이상의 명령과 직접 검토한 바꿔쓴 문장을 연결했다. 물을 주는 용기만 바뀐 식물 물주기, 감자의 전자레인지 조리, 토마토 썰기 등의 동일 작업도 묶었다.

중복으로 선별에서 빠진 원본도 그 대표 문장의 장면 정보를 보존하기 위해 연결에 포함했다. 보류와 제외 항목은 모델 입력에 쓰지 않으므로 분할 연결 대상에서도 제외했다. 원문을 정규화해서 학습하는 것은 아니며, 정규화는 중복과 그룹 판별에만 사용한다.

이 관계를 모두 연결하면 429개 중 366개가 하나의 큰 묶음이 된다. 관계를 끊어 7:1.5:1.5 비율을 맞추지 않고 이 묶음을 학습에 배정했다. 따라서 최종 비율은 약 85:7:8이다. 이 큰 묶음은 동일한 장면 하나나 동일한 명령 366개라는 뜻이 아니라, 여러 장면과 비교 명령이 연쇄적으로 연결됐다는 뜻이다.

분할은 seed 42를 사용하며, 묶음 크기와 클래스 및 범주 구성만으로 결정한다. 모델 정확도나 오답을 보고 분할을 고르지 않았다. 초기 분할 초안은 `artifacts/f1t/sab-split-draft-*`에 남겨 두었고 실제 학습 입력은 아래의 `splits_v1`만 사용한다. 초안을 만드는 동안 모델 학습이나 검증 점수 계산을 하지 않았다.

## 평가 범위와 한계

이번 분할은 학습에 포함된 위험 종류에서 다른 장면과 명령 묶음으로 일반화하는지 살펴보는 초기 진단용이다. 위험 범주 전체를 학습에서 제외한 별도의 미학습 위험 유형 평가를 새로 만든 것은 아니다.

- 검증의 위험 사례는 충격과 파손 5개, 가열 오용 1개, 미끄러짐 1개뿐이다.
- 시험의 위험 사례는 충격과 파손 6개, 가열 오용 2개, 생활 물품 액체 손상 1개다.
- 전기 장치 금속 접촉, 전자기기 액체 손상, 식물의 알코올 손상, 비식품 섭취는 이번 검증 및 시험에서 평가하지 못한다. 이 유형들의 성능을 주장하려면 독립적인 상황을 추가 확보해야 한다.
- 검증은 항상 정상으로 답해도 **24/31 = 77.4%**다. 정확도 70% 또는 77%만으로 학습이 잘됐다고 결론 내릴 수 없다.
- 학습 보고서에 항상 정상으로 판정한 기준 정확도, 항상 위험으로 판정한 기준 정확도, balanced accuracy를 추가했다. balanced accuracy는 정상과 위험 각각의 정답률을 반씩 반영하며, 상수 판정의 기준값은 50%다.
- 기존의 위험 누락률, 정상 오차단률과 AUROC도 함께 유지한다. 검증 위험 사례가 7개여서 한 건만 달라도 위험 누락률이 약 14.3%p 바뀐다.
- 자동 및 수동으로 확인한 연결에 대해 분리 경계를 검사했다. 모든 의미적 유사 문장을 완벽하게 제거했다는 의미는 아니다. `cross_split_near_text_review.json`에 유사한 경계 사례를 남겼다.

SafeAgentBench를 학습에 재사용하므로 해당 학습 사례를 다시 독립적인 공식 벤치마크 성능으로 보고하지 않는다. 시험 파일은 분할 생성과 입력 무결성 확인 외에 모델 학습이나 성능 계산에 사용하지 않았다. 고정 인코더의 시험 임베딩은 캐시에 함께 저장하지만 학습 함수가 시험 행을 학습 및 검증 점수 계산에서 제외한다.

## 학습 코드의 변경

`src/f1/training/safeagent_aux.py`에 전용 분할 및 입력 검증을 추가했다. `text_aux.py`는 기존 SafeText 형식의 검사를 유지하면서 명시적으로 허용한 SafeAgentBench 형식도 처리한다. 단순히 모든 출처를 허용하도록 검사를 풀지 않았다.

입력은 원본 `instruction` 그대로인 `text` 필드만 사용한다. 범주, 검토 사유, 비교용 위험 명령, 실행 정답 단계, 장면 ID는 인코더 입력에 넣지 않는다. 잘못된 라벨, 보류 항목, 임의 출처, 변경된 명령, 서로 다른 분할의 동일 장면이나 비교 그룹, 분할 파일 해시 변경을 거부한다.

검증 과정에서 기존 SafeText 경로도 함께 확인했다. 합성 소규모 데이터로 학습 호출을 감시하는 테스트에서 시험 행이 학습 또는 점수 계산에 들어가지 않는 것을 확인했다. 프로젝트 전체 테스트 99개가 통과했다. 실제 SafeAgentBench MLP의 학습은 실행하지 않았다.

## 파일과 실행 명령

분할: `data/safeagentbench_f1t_english_v1/splits_v1/`

- `train.jsonl`, `validation.jsonl`, `test.jsonl`: 새 스키마로 연결한 입력.
- `summary.json`: 분할별 개수, 범주, 원본 및 분할 파일 해시.
- `group_assignment.json`: 원본 행과 연결 묶음 및 분할의 대응.
- `near_text_edges.json`: 자동 유사 문장 및 수동 검토 연결.
- `cross_split_near_text_review.json`: 분할 사이의 유사 명령 검토 자료.

캐시: `artifacts/f1t/text-aux-safeagentbench-v1/cache/`. 기존과 동일한 고정 다국어 CLIP 텍스트 인코더를 CPU에서 실행한다. 모델 revision은 `58edf8cada9e398793dca955574a48cbb7f18be2`다.

다음 명령으로 준비된 캐시의 학습을 실행할 수 있다. 출력 경로는 기존 결과를 덮어쓰지 않는 새 디렉터리여야 한다. 이 문서 작성 시에는 실행하지 않았다.

```powershell
uv run --frozen --extra train --extra ocr python -m f1.training.text_aux fit --cache artifacts/f1t/text-aux-safeagentbench-v1/cache --config configs/f1t-train.json --output artifacts/f1t/text-aux-safeagentbench-v1/fit-001
```

분할 및 캐시의 재현 명령:

```powershell
uv run --frozen --extra train --extra ocr python -m f1.training.text_aux prepare-sab --data data/safeagentbench_f1t_english_v1 --output artifacts/f1t/sab-split-reproduction
uv run --frozen --extra train --extra ocr python -m f1.training.text_aux cache --data data/safeagentbench_f1t_english_v1/splits_v1 --output artifacts/f1t/text-aux-safeagentbench-v1/cache-reproduction
```

선별 단계의 `selected.jsonl`은 감사용 후보 풀로 남겨 두었으므로 그 안의 `split=UNASSIGNED`는 변경하지 않았다. 선별 단계 `summary.json`의 준비 상태도 당시의 기록이며, 현재 학습 연결 상태는 `splits_v1/summary.json`과 이 문서를 기준으로 확인한다.
