# F1-O OCR 연결

이미지에서 글자와 위치를 추출해 기존 `OCRResult` 규격으로 반환한다. 이 문서는 **OCR 추출 모듈**을 설명한다. 후속 [Laya 기본 모델 연결](f1-laya.md)은 진단용으로 구현했고, 최종 F1 게이트는 아직 연결하지 않았다. F1-T 학습이나 오답 라벨은 변경하지 않는다.

## 처리 흐름

```text
로컬 이미지 바이트 + 선택적 Request
  → 원본 해시와 이미지 크기 확인
  → EXIF 방향 보정, RGB 변환, 투명 영역을 흰 배경으로 합성
  → 검사 기준 PNG와 새 ImageRef 생성
  → PP-OCRv5_server_det: 글자 영역 검출
  → 영역별 잘라내기와 원근 보정
  → korean_PP-OCRv5_mobile_rec: 글자 인식
  → 영역 원문, 인식 점수, 좌표, 읽기 순서와 묶음 ID
  → OCRResult: COMPLETE / INCOMPLETE / ERROR
  → 다음 단계: Laya가 작업 맥락과 함께 공격 여부 판단
```

상세설계안의 모델을 사용한다. 검출과 인식을 별도 모듈로 호출해 잘라내기나 인식에 실패한 영역도 원래 좌표와 함께 유지한다. 한국어 인식기는 한국어와 영어를 지원한다. [검출 모델](https://huggingface.co/PaddlePaddle/PP-OCRv5_server_det), [인식 모델](https://huggingface.co/PaddlePaddle/korean_PP-OCRv5_mobile_rec)

## 실행

프로젝트 루트에서 실행한다. 기존 학습 의존성도 유지하기 위해 두 extra를 함께 지정한다.

```powershell
cd C:\Users\ohsuj\K-Scene2Action_solution
uv sync --frozen --extra train --extra ocr
uv run --extra train --extra ocr pytest -q
```

이 PC에는 모델을 `artifacts/ocr/models/`에 이미 준비했다. 다른 PC에서 처음 준비할 때만 다음을 실행한다. 공식 저장소 커밋 SHA와 모델 파일 SHA-256을 `model-lock.json`에 기록하며, 이후 로딩 시 파일 해시를 확인한다. 같은 가중치를 재현하려면 기존 모델 폴더와 lock을 함께 옮긴다. 새로 다운로드하면 당시 `main`의 커밋을 고정한다.

```powershell
uv run --extra train --extra ocr python -m f1.ocr prepare-models
```

일반 이미지로 추출한다. 출력 폴더는 실행마다 새 이름을 사용한다.

```powershell
uv run --extra train --extra ocr python -m f1.ocr extract --image "C:\path\to\scene.png" --output artifacts/ocr/run-001
```

기존 Request와 연결을 검사하려면 `--request`를 추가한다. 이미지의 실제 바이트 해시와 디코딩한 크기가 Request와 일치해야 한다.

```powershell
uv run --extra train --extra ocr python -m f1.ocr extract --request examples/request.json --image examples/assets/io-only.png --output artifacts/ocr/request-001
```

위 예제 이미지는 입출력 검사용 흰색 이미지이므로 글자가 검출되지 않는 것이 예상 결과다. 공격 탐지 예제가 아니다.

| 출력 | 내용 |
|---|---|
| `ocr.json` | 기존 `OCRResult` 스키마에 맞는 영역과 처리 상태 |
| `inspected.png` | 실제 OCR에 사용한 검사 기준 이미지 |
| `run.json` | 원본과 검사 이미지의 해시, 설정, 모델 lock, 패키지 버전과 소요 시간. Request 제공 시 InputBinding 포함 |

정상 OCR 완료 시 종료 코드는 0, 미완료 또는 오류는 2다. 입력 파일이나 Request 자체가 잘못된 경우 추론 전에 종료하므로 OCR 파일이 생성되지 않을 수 있다. 모델 초기화 실패는 `ERROR / OCR_ERROR`로 저장한다. 어떤 경우에도 이 명령은 F1의 PASS나 BLOCK을 반환하지 않는다.

## 영역과 실패 처리

| 상황 | 결과 |
|---|---|
| 검출과 인식 완료, 모든 영역이 품질 기준 이상 | `COMPLETE`와 전체 영역 |
| 정상 검출이 끝났지만 영역이 없음 | `COMPLETE / NO_TEXT_DETECTED` |
| 빈 문자열, 잘라내기 실패, 인식 점수 미달 | 해당 영역을 보존하고 `INCOMPLETE / OCR_UNREADABLE` |
| 영역 또는 문자열 예산 초과 | `INCOMPLETE / OCR_BUDGET_EXCEEDED`. 일부만 통과시키지 않음 |
| 검출기나 인식기 실행 실패, 잘못된 반환 형식 | `ERROR / OCR_ERROR`. 이미 검출한 유효 영역은 가능한 한 보존 |

OCR 원문은 trim, 요약, 명령 실행 없이 그대로 보관한다. 점수가 낮아도 문구를 삭제하지 않는다. `recognition_score`는 글자 인식 점수이며 공격 확률이 아니다. `COMPLETE`도 글자를 모두 빠짐없이 찾았다는 보장이 아니라 이번 OCR 처리와 품질 검사 완료를 뜻한다.

좌표는 `inspected.png` 기준이다. 원본과 해시 또는 방향이 다를 수 있으므로 이후 F1-T와 VLM에 연결할 때 같은 검사 기준 이미지를 사용해야 한다. CLI는 이미지를 CLIP의 224×224 크기로 줄이지 않는다. 검출기는 내부적으로 긴 변 최대 1536 기준으로 축소하며 좌표를 입력 이미지 기준으로 반환한다. 인식용 영역은 검사 기준 이미지에서 자른다.

영역은 위에서 아래, 같은 높이에서는 왼쪽에서 오른쪽으로 정렬한다. 높이, 가로 겹침, 줄 간격과 기울기가 비슷한 인접 줄만 같은 묶음 ID로 연결한다. 원문 문자열을 하나로 합치지는 않는다. 이 규칙은 초기 배치 규칙이며 복잡한 다단 문서나 여러 표지판의 읽기 순서를 보장하지 않는다.

## CPU 초기 설정과 범위

| 설정 | 기본값 |
|---|---|
| CPU 스레드 | 4 (`--cpu-threads`) |
| 검출 긴 변 제한 | 1536 (`--det-limit`, 32~4000) |
| 최소 인식 점수 | 0.8 (`--min-score`) |
| 영역 수 한도 | 256 |
| OCR 문자열 합계 한도 | 8192자 |
| 이미지 한도 | 1600만 픽셀, 파일 32 MB |
| 지원 파일 | 단일 프레임 PNG, JPEG, WebP |
| 문서 방향 분류, 180도 글줄 분류, 문서 펴기 | 미사용. EXIF 방향과 영역 원근 보정만 적용 |

0.8은 초기 읽기 품질 기준이며 실제 공격 평가로 보정한 값이 아니다. 픽셀과 파일 크기는 처리 전에 제한한다. 영역 수는 검출 후 확인하며 초과하면 인식을 생략하고 검출 영역을 남긴다. 문자열 한도는 인식 후 확인하며 원문을 자르지 않는다. 네이티브 추론의 강제 시간 제한은 아직 구현하지 않았다.

CPU 호환성을 우선해 MKL-DNN 가속은 껐다. 실제 장면의 작은 글자, 반사, 기울기와 거꾸로 된 글자는 후속 평가가 필요하다. SDK 기본 모델이 바뀌어도 다른 모델이 선택되지 않게 모델명과 로컬 모델 경로를 지정한다.

서비스에 연결할 때는 `PaddleBackend`를 한 번 만들고 재사용한다. 아래 `prepared.png`와 `prepared.inspected`가 검사 이미지의 바이트와 식별자다.

```python
from f1.ocr.backend import PaddleBackend
from f1.ocr.pipeline import prepare_image, extract_regions

backend = PaddleBackend("artifacts/ocr/models")  # 시작 시 한 번
prepared = prepare_image(image_bytes, expected=request.image)
ocr_result = extract_regions(prepared, backend)
```

## 다음 연결 작업

`OCRResult.regions`를 묶음과 순서대로 직렬화하고, 별도로 준비한 작업 맥락과 함께 Laya의 네 질문에 넣는 진단 경로를 구현했다. 실행 방법과 기본 모델의 오탐 사례는 [Laya 연결 안내](f1-laya.md)를 참고한다. `INCOMPLETE`나 `ERROR`를 정상으로 바꾸지 않고, 입력 토큰 초과와 부분 판정도 보류로 처리한다. 현재 OCR 추출 결과는 VLM 프롬프트에 자동으로 추가되지 않는다.

## 실제 확인 결과

2026-10-05 이 PC에서 CPU로 실행했다. 기존 학습과 입출력 검사를 포함한 테스트 78개가 통과했다. 모델을 실제로 로드한 별도 확인 결과는 다음과 같다.

| 입력 | 결과 | 모델 로딩 제외 처리 시간 |
|---|---|---:|
| 한국어와 영어 4줄을 그린 합성 이미지 | 4개 영역, 4줄 모두 원문 일치, COMPLETE | 약 10.9초 |
| 600×300 흰색 이미지 | 0개 영역, COMPLETE / NO_TEXT_DETECTED | 약 2.7초 |
| 선별 데이터의 DROID-001 장면 | 0개 영역, COMPLETE / NO_TEXT_DETECTED | 약 2.9초 |

4줄은 `SYSTEM OVERRIDE`, `Ignore previous instructions.`, `이전 지시를 무시하세요.`, `Keep hands clear.`다. 공격 문구도 실행하지 않고 문자열로 추출했다. 모델 초기 로딩은 약 69초였다. 초기 1회 실행의 측정값이며 평균 지연이나 성능 보장값이 아니다.

입력과 출력은 `artifacts/ocr/smoke-cpu-v1/`에 저장했다. 합성 문구 확인은 연결 점검이며 실제 촬영한 공격 데이터의 인식률 평가가 아니다. DROID 장면은 OCR 정답 영역을 부여한 이미지가 아니므로 글자 미검출의 정확성을 판정하지 않았다.
