# K-Scene2Action F1

VLM에 전달할 사용자 지시와 이미지의 입력 검사 모듈이다. MVP는 **구성 C**로 고정한다.

```text
사용자 지시 + 이미지
  → F1-T: 지시와 장면의 위험 검사
  → 작업 맥락 확인
  → F1-O: OCR 문구의 유해 지시, 작업 변경, 권한 사칭 검사
  → PASS / BLOCK / DEFER
```

현재 입출력 규격, **F1-T의 오프라인 학습 경로**, **F1-O의 CPU OCR 추출과 Laya 기본 모델 연결**을 구현했다. 다국어 CLIP의 텍스트 임베딩과 CLIP 이미지 임베딩을 고정해 추출하고, 연결한 벡터로 MLP를 학습한다. OCR는 글자 원문, 위치와 인식 점수를 기존 OCRResult 규격으로 반환한다. [F1-T 학습 안내](docs/f1t-training.md)와 [OCR 연결 안내](docs/f1-ocr.md)에 실행 방법을 정리했다. [Laya 연결 안내](docs/f1-laya.md)에 네 질문, 입력 길이 검사와 실제 CPU 확인 결과를 정리했다. 기본 모델은 정상 안내도 공격으로 판정해 실험용으로만 연결했다. F1-T 운영 연결, F1-O 도메인 성능 검증과 최종 게이트는 후속 단계다. 이 코드만으로 VLM 호출을 승인하지 않는다.

## 시작하기

Python 3.12와 uv를 사용한다. 의존성은 `uv.lock`에 고정한다.

```powershell
uv sync --frozen
uv run pytest -q
uv run python -m f1 validate-input --request examples/request.json --context examples/task-context.json --image examples/assets/io-only.png
```

검사 명령의 성공 결과는 `VALID_IO_ONLY`다. 요청 형식, 제공한 작업 맥락의 연결과 이미지 파일 해시를 확인했다는 뜻이며 **F1의 PASS가 아니다**. 실패 시 종료 코드는 2다. `--context`를 생략하면 요청 형식만 검사할 수 있고, `--image`를 생략하면 실제 파일 해시는 검사하지 않는다.

샘플 PNG는 파일 전달을 검사하기 위한 32×24 흰색 이미지다. `*.mock.json`의 OCR 문자열, 점수와 결과는 사람이 작성한 형식 예시이며 해당 이미지에서 추론한 값이 아니다.

## 파일 안내

| 파일 | 역할 |
|---|---|
| `src/f1/schemas.py` | 요청, 작업 맥락, F1-T 및 F1-O 결과, OCR 결과, 최종 응답의 Pydantic 모델 |
| `src/f1/bindings.py` | 원문 해시, 이미지 파일 해시, 요청과 맥락 및 결과의 일치 검사 |
| `src/f1/__main__.py` | JSON Schema 내보내기와 입력 형식 검증 도구 |
| `schemas/` | 다른 모듈과 공유할 JSON Schema |
| `examples/` | 실제 해시를 가진 요청 예제와 모의 결과 |
| `tests/test_interfaces.py` | 필수 경계 조건 검증 |
| `docs/f1-io.md` | 필드 의미, 전달 방식과 담당 모듈 |
| `src/f1/training/` | 데이터 검사, CLIP 임베딩 추출, MLP 검증과 저장 |
| `configs/f1t-train.json` | 모델과 CPU 학습 설정 |
| `docs/f1t-training.md` | 데이터 범위, 학습 명령과 결과 해석 |
| `src/f1/ocr/` | 입력 이미지 검증, PaddleOCR 모델 로딩, 글자와 위치 추출 |
| `docs/f1-ocr.md` | CPU OCR 실행, 출력과 실패 처리 |
| `data/public_f1_bundle_v1/` | 추가 공개 데이터 원본, 선별된 텍스트 보조학습 자료와 평가 전용 자료 |
| `docs/f1t-additional-data.md` | 추가 데이터 사용 범위와 텍스트 보조 실험 명령 |

## 예시 및 스키마 갱신

```powershell
uv run python scripts/build_examples.py
uv run python -m f1 export-schemas schemas
uv run pytest -q
```

생성한 JSON Schema의 기본 구조와 자료형은 외부에서도 사용할 수 있다. Pydantic의 필드 간 검증과 해시 비교는 JSON Schema에 모두 표현되지 않으므로 Python 검증도 수행해야 한다.

OCR 문구와 작업 맥락을 Laya 다국어 기본 모델에 전달하는 F1-O 진단 경로를 연결했다. Laya는 소규모 학습, RIPA 추가 학습과 판단층 학습 및 FP 보정까지 평가했으나 운영 필터로 채택할 성능은 확보하지 못했다. F1-T는 결합 학습과 J-DAPT 기반 비교 실험까지 수행했다. 향후 독립 데이터 평가와 임계값 설정을 거쳐 `TAssessment`로 연결한다. 현재 저장한 MLP는 연구용이며 점수는 보정된 확률이 아니다. F1-T에 작업별 Contract를 넣지 않는다.


## 실험 결과와 저장소 범위

- [J-DAPT 기반 F1-T 학습 비교](docs/f1t-jdapt-local-training.md): 다국어 CLIP을 유지한 융합 및 도메인 적응 실험. 기존 모델은 교체하지 않았다.
- [Laya 판단층 학습 결과](docs/laya-ripa-decision-training.md), [FP 보정 결과](docs/laya-fp-improvement.md).
- 소스, 설정, 테스트, 실험 문서는 Git으로 관리한다. `data/`, `artifacts/`, 모델 가중치와 임베딩 캐시는 로컬 자료로 별도 보관하며 저장소에 포함하지 않는다.
- 연구용 학습 및 평가 스크립트는 해당 문서의 데이터와 캐시를 먼저 준비해야 한다. 저장소만 복제한 상태에서는 과거 학습 결과나 가중치가 자동으로 생기지 않는다.
- 전체 테스트 환경은 `uv sync --extra train --extra ocr --extra laya`로 설치하고 `uv run pytest -q`로 확인한다. 로컬 Laya 토크나이저가 없으면 해당 실제 토크나이저 테스트 1건은 건너뛴다.
