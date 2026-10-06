# F1-T 다국어 CLIP 학습 준비

사용자 지시와 장면 이미지를 함께 보고 위험도를 학습하는 **F1-T**의 오프라인 학습 경로다. 이 PC의 CPU에서 실행한다. F1-O의 OCR 판별과 F2는 이번 구현에 포함하지 않는다. F1-T는 Contract를 입력으로 받지 않는다.

## 1. 무엇을 학습하는가

```text
영어 지시 ── 다국어 CLIP 텍스트 인코더 ── 정규화된 512차원 벡터 ─┐
                                                              ├─ 연결: 1,024차원
장면 이미지 ─ CLIP 이미지 인코더 ─────── 정규화된 512차원 벡터 ─┘
                                                                  ↓
                                                    MLP: 1,024 → 128 → 1
                                                                  ↓
                                                sigmoid → 0~1 범위의 위험 점수
```

CLIP은 이미지와 텍스트를 숫자 벡터로 바꾸는 인코더다. 유해 여부를 직접 분류하지 않으므로 뒤에 MLP를 붙인다. MLP는 연결한 벡터를 받아 정상은 0, 유해는 1에 가까운 점수를 내도록 학습하는 작은 신경망이다.

- 텍스트: `sentence-transformers/clip-ViT-B-32-multilingual-v1`
- 이미지: `sentence-transformers/clip-ViT-B-32`
- 인코더는 고정한다. **학습하는 것은 MLP뿐**이다.
- 같은 명령과 이미지는 한 번만 인코딩해 저장한다. 이후 MLP 학습에서는 CLIP을 다시 실행하지 않는다.

다국어 모델은 원래 CLIP의 임베딩 공간에 맞춘 텍스트 인코더다. 이미지는 별도 원본 CLIP 인코더가 처리한다. 두 벡터를 같은 의미 공간에서 얻을 수 있어 지시와 장면을 함께 사용하는 초기 모델로 선택했다. 다만 벡터를 연결한다고 물체 지칭이나 위험한 행동 관계를 반드시 이해하는 것은 아니다. [공식 모델 설명](https://huggingface.co/sentence-transformers/clip-ViT-B-32-multilingual-v1)

현재 학습 데이터는 영어다. 인코더가 다국어를 지원해도 학습한 분류기의 한국어 성능까지 검증된 것은 아니다.

## 2. 사용하는 데이터

`data/robojailbench_f1t_english_v1/selected.jsonl`을 사용한다. 데이터 버전은 `rjb-f1t-en-v1.2`다.

| 항목 | 개수 |
|---|---:|
| 정상 지시와 유해 지시의 쌍 | 102 |
| 정상 샘플 | 102 |
| 유해 샘플 | 102 |
| 전체 샘플 | 204 |
| 중복 유입 방지용 그룹 | 4 |

Ethical Social과 Permission Related는 제외했다. 선별 상태, 영어 여부, 이미지와 명령의 해시, 이미지 크기, 쌍의 완전성, 그룹 간 중복을 실행 전에 검사한다. 원본 명령은 바꾸지 않는다. 현재 라벨은 선별 검토 결과이며 별도의 사람 검증을 마친 정답 집합으로 보지 않는다.

텍스트는 실제 토크나이저로 길이를 검사한다. 특수 토큰을 포함해 128토큰을 넘으면 조용히 자르지 않고 임베딩 추출을 중단한다. 이미지는 방향 정보를 반영하고 RGB로 바꾼 뒤, 비율을 유지해 224×224 영역에 맞추고 여백을 채운다. 여러 카메라 장면이 가로로 이어진 이미지의 양끝이 중앙 크롭으로 사라지지 않게 한다. 다만 긴 이미지를 축소하면 작은 물체가 덜 보일 수 있으므로 실제 지칭 판단 성능은 별도로 평가해야 한다.

## 3. CPU 실행 순서

프로젝트 루트에서 실행한다. 최초 실행 시 공개 모델 가중치를 내려받으므로 인터넷 연결과 저장 공간이 필요하다. 가중치는 Hugging Face 캐시에, 임베딩과 결과는 Git에서 제외한 `artifacts/`에 저장한다.

```powershell
cd C:\Users\ohsuj\K-Scene2Action_solution
uv sync --frozen --extra train
uv run --extra train pytest -q
```

### 3.1 데이터 검사와 그룹 분리

```powershell
uv run --extra train python -m f1.training prepare --manifest data/robojailbench_f1t_english_v1/selected.jsonl --output artifacts/f1t/prepare-run2
```

`data-audit.json`에 검사 결과를, `folds.json`에 학습과 검증 샘플 ID를 저장한다. 텍스트 토큰 길이는 다음 임베딩 추출 단계에서 확인한다.

### 3.2 임베딩 추출

```powershell
uv run --extra train python -m f1.training cache --manifest data/robojailbench_f1t_english_v1/selected.jsonl --output artifacts/f1t/cache-run2 --device cpu
```

`features.npy`에는 샘플별 1,024차원 벡터를 저장한다. `metadata.json`에는 샘플 순서, 데이터 해시, 모델 커밋 SHA, 전처리 설정과 패키지 버전을 저장한다. 모델의 `main`을 실제 커밋 SHA로 해석한 뒤 그 버전을 불러온다. 같은 모델을 다시 추출하려면 기록된 SHA를 설정의 `text_revision`, `image_revision`에 넣는다.

`metadata.json`은 추출이 끝난 뒤에만 생성한다. 없는 디렉터리는 완료된 캐시가 아니다. 후속 단계는 벡터 파일 해시, 차원과 정규화를 다시 검사한다.

### 3.3 그룹별 초기 검증

```powershell
uv run --extra train python -m f1.training cv --cache artifacts/f1t/cache-run2 --output artifacts/f1t/cv-both-run2 --features both --device cpu
uv run --extra train python -m f1.training cv --cache artifacts/f1t/cache-run2 --output artifacts/f1t/cv-text-run2 --features text --device cpu
uv run --extra train python -m f1.training cv --cache artifacts/f1t/cache-run2 --output artifacts/f1t/cv-image-run2 --features image --device cpu
```

4개 그룹 중 하나를 검증용으로 빼고 나머지로 학습하는 과정을 4번 반복한다. 같은 쌍, 동일 이미지와 중복 명령이 학습과 검증에 걸치지 않게 한다. 각 학습은 30 epoch로 고정한다. 검증 그룹 점수를 보고 epoch를 고르거나 조기 종료하지 않는다.

`report.json`에는 그룹별 결과, 그룹 평균, 전체 검증 예측을 모은 결과와 학습 손실을 저장한다. `predictions.json`에는 각 샘플이 학습에서 제외되었을 때의 점수를 저장한다. 정확도, 유해 샘플 누락률, 정상 샘플 오차단률과 AUROC를 함께 본다. **0.5는 초기 비교를 위한 임시 기준이며 운영용 차단 임계값이 아니다.**

`both`와 `text`를 비교하면 이미지 추가가 현재 데이터에서 도움이 되는지 확인할 수 있다. `image`는 정상과 유해가 같은 이미지를 공유하는 데이터의 기준 실험이다. 이 비교만으로 장면 이해를 입증할 수는 없다. 추후 동일 지시에 안전한 이미지와 위험한 이미지를 교체한 별도 평가가 필요하다.

### 3.4 연구용 MLP 저장

```powershell
uv run --extra train python -m f1.training fit --cache artifacts/f1t/cache-run2 --output artifacts/f1t/fit-both-run2 --features both --device cpu
```

204개 전체로 MLP를 학습해 `model.safetensors`와 `report.json`을 저장한다. 전체로 학습한 모델에는 이 데이터에서의 독립 평가 점수를 부여하지 않는다. 검증은 앞의 그룹별 실험 결과로 구분한다.

각 명령은 새 출력 디렉터리를 요구한다. 재실행할 때는 `run2`를 `run3`처럼 바꾼다. 기존 결과나 중단된 캐시를 자동으로 덮어쓰지 않는다.

## 4. 초기 학습 설정

설정 파일은 `configs/f1t-train.json`이다. 아래 값은 검증으로 최적화한 결과가 아니라 첫 실험의 출발점이다.

| 설정 | 값 |
|---|---|
| MLP | Linear(1024, 128), ReLU, Dropout(0.1), Linear(128, 1) |
| 학습 대상 | MLP 131,329개 파라미터 |
| 손실 함수 | BCEWithLogitsLoss |
| 최적화 | AdamW, 학습률 0.001, weight decay 0.01 |
| epoch / batch size | 30 / 32 |
| CPU 스레드 | 4 |
| seed | 42 |

CPU 스레드는 작은 MLP에서 과도한 스레드 사용을 줄이기 위한 초기값이다. 최적값이라는 뜻은 아니며 `cpu_threads`로 조정한다. 모델이나 전처리를 바꾸면 임베딩을 다시 추출한다. MLP 설정만 바꾸는 실험은 기존 캐시를 사용할 수 있다.

## 5. 이번 단계의 완료 기준과 다음 작업

이번 구현은 데이터 검사, 고정 CLIP 임베딩 추출, 그룹별 MLP 검증, 연구용 가중치 저장까지다. 합성 임베딩을 사용하는 자동 테스트는 학습 코드와 저장 경로가 작동하는지 확인한다. 실제 공격 탐지 성능의 증거는 아니다.

현재 4개 그룹만으로는 학습용, 보정용, 최종 시험용을 충분히 분리하기 어렵다. 따라서 결과에 `deployment_ready: false`를 기록한다. 저장된 점수는 보정된 확률이 아니며 기존 `TAssessment`의 PASS, BLOCK, DEFER로 아직 연결하지 않는다.

다음에는 독립 데이터를 확보하고, 장면에 따라 정답이 바뀌는 사례와 정상 지시의 오차단을 평가한다. 이후 운영 임계값과 DEFER 조건을 정하고 입출력 인터페이스에 연결한다. 다국어 평가와 긴 입력 처리도 별도 검증 대상이다.

## 6. 이 PC에서 실행한 초기 결과

2026-10-05에 CPU로 실제 모델 로딩, 임베딩 추출, 3종 비교 검증과 전체 데이터 MLP 저장을 완료했다. 중복을 제외한 명령은 203개, 이미지는 102개이며 최대 입력 길이는 특수 토큰을 포함해 36토큰이었다. 204개 샘플의 임베딩은 `artifacts/f1t/cache-cpu-v1.2/`에 있다.

| 입력 | 정확도 | 유해 누락률 | 정상 오차단률 | AUROC |
|---|---:|---:|---:|---:|
| 텍스트 + 이미지 | 76.5% | 16.7% | 30.4% | 0.881 |
| 텍스트만 | 76.5% | 20.6% | 26.5% | 0.874 |
| 이미지만 | 50.0% | 71.6% | 28.4% | 0.500 |

위 표는 4개 그룹을 한 번씩 제외해 얻은 전체 검증 예측을 모은 결과다. AUROC를 제외한 수치는 임시 임계값 0.5 기준이다. 단일 seed의 작은 데이터 실험이고 독립 시험 결과가 아니므로 이미지 결합의 우위를 확정할 수 없다. 특히 현재 정상 오차단률은 높다. 초기 모델을 바로 운영에 연결할 근거로 사용하지 않는다.

텍스트 단독과 결합 모델의 정확도가 같고 이미지 단독은 무작위 수준이다. 정상과 유해가 같은 이미지를 공유하는 데이터 구조에서 예상할 수 있는 결과다. **이 데이터만으로 '같은 지시라도 장면이 위험하면 잡아낸다'는 핵심 기능을 입증할 수 없다는 점**이 다음 데이터 준비에서 가장 중요하다.

검증 결과는 `artifacts/f1t/cv-both-cpu-v1.2/`, `cv-text-cpu-v1.2/`, `cv-image-cpu-v1.2/`에 있다. 연구용 결합 모델은 `artifacts/f1t/fit-both-cpu-v1.2/model.safetensors`에 저장했다. 기존 캐시를 재사용하려면 앞의 학습 명령에서 `--cache`를 `artifacts/f1t/cache-cpu-v1.2`로 지정하면 된다.

## 7. 오답 검토표

[오답 집계와 전체 목록](../artifacts/f1t/error-review-cpu-v1.2/README.md), [이미지를 포함한 검토 화면](../artifacts/f1t/error-review-cpu-v1.2/error-review.html)을 만들었다. 임시 임계값 0.5에서 정상 오차단 31개와 유해 누락 17개, 총 48개다.

HTML 파일을 브라우저에서 열면 원본 장면, 원문 지시, 데이터 라벨, 결합 점수와 텍스트 단독 점수를 볼 수 있다. 같은 이미지의 반대 라벨 지시도 펼쳐볼 수 있다. 라벨 검토, 주요 원인 가설과 메모를 기록한 뒤 JSON으로 내보낸다. 기존 데이터 라벨과 모델은 자동으로 바뀌지 않는다.

원본 이미지를 HTML 안에 포함했으므로 파일 하나로 열 수 있다. 브라우저 임시 저장은 보조 기능이며 검토 결과는 JSON으로 내보내 보관한다. 아직 사람의 오답 원인 검토를 완료한 자료는 아니다.

같은 검증 결과에서 새 검토표를 생성하려면 다음을 실행한다. 출력 디렉터리는 새 이름을 사용한다.

```powershell
uv run --frozen --extra train python scripts/build_f1t_error_review.py --output artifacts/f1t/error-review-run2
```
