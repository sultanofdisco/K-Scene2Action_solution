# K-Scene2Action F1

VLM에 전달할 사용자 지시와 이미지의 입력 검사 모듈이다. MVP는 **구성 C**로 고정한다.

```text
사용자 지시 + 이미지
  → F1-T: 지시와 장면의 위험 검사
  → 작업 맥락 확인
  → F1-O: OCR 문구의 유해 지시, 작업 변경, 권한 사칭 검사
  → PASS / BLOCK / DEFER
```

현재 완료 범위는 상세설계안 9.4절의 **1단계: 입출력 확정**이다. 요청, 작업 맥락, 모델 입력과 결과, 사유 코드, 입력 식별 검증을 구현했다. 모델 추론, 학습, OCR 실행과 최종 게이트는 후속 단계다. 이 코드만으로 공격을 탐지하거나 VLM 호출을 승인하지 않는다.

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

## 예시 및 스키마 갱신

```powershell
uv run python scripts/build_examples.py
uv run python -m f1 export-schemas schemas
uv run pytest -q
```

생성한 JSON Schema의 기본 구조와 자료형은 외부에서도 사용할 수 있다. Pydantic의 필드 간 검증과 해시 비교는 JSON Schema에 모두 표현되지 않으므로 Python 검증도 수행해야 한다.

다음 단계는 F1-T 연결 및 학습이다. `TInput`의 지시와 검사 기준 이미지를 받아 `TAssessment`를 반환하도록 연결한다. F1-T에 작업별 Contract를 넣지 않는다.
