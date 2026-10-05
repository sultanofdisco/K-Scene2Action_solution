# F1 입출력 명세 v1.0

작성일: 2026-10-05

기준: `K-Scene2Action_F1_상세설계안.md`의 2절, 3.2절, 5.2절, 6절, 7절과 9.4절

구현 범위: 개발 순서 1단계, 입출력 확정

## 1. 이번 단계의 결정

처음에는 한 Python 프로그램에서 모듈을 연결한다. HTTP 서버는 아직 만들지 않는다. 외부 JSON은 Pydantic으로 검증하고, 모듈 사이에는 검증된 객체를 전달한다. 선언하지 않은 필드는 거부한다. 출력 객체는 변경하지 않고 새 입력에 대해서는 새 객체를 만든다.

MVP의 `mode`는 `C`만 허용한다. D의 마스킹 설정이나 영역 보호 정보는 입력에 필요하지 않다. 텍스트 및 이미지 전처리 한도와 모델 임계값은 후속 구현에서 정책으로 확정하며, 이 단계에 임의의 운영 임계값을 넣지 않는다.

## 2. 이미지 전달 방식

`Request.image`는 이미지 바이트를 JSON에 담는 대신 `asset_id`로 원본 파일을 참조한다. 실행 측이 관리하는 자산 저장소나 메모리 매핑에서 ID를 실제 바이트로 해석한다. 임의의 URL이나 사용자 파일 경로를 자동으로 여는 방식은 정의하지 않는다. CLI의 `--image`는 개발자가 지정하는 로컬 검증 파일이다.

| 필드 | 의미 |
|---|---|
| `asset_id` | 원본 또는 검사 기준 이미지의 식별자 |
| `sha256` | 인코딩된 파일 바이트 전체의 SHA-256, 소문자 64자리 |
| `width`, `height` | 해당 파일을 디코딩한 이미지의 너비와 높이, 양의 정수 |

`verify_image_bytes`는 실제 파일 바이트와 선언한 해시를 비교한다. 디코딩 성공과 실제 크기의 일치 여부는 다음 단계의 입력 처리에서 확인한다. 스키마에 정상 크기가 적혀 있다는 사실만으로 이미지 검사가 끝난 것은 아니다.

방향 보정과 RGB 변환으로 새 파일을 만들면 검사 기준 이미지에도 별도의 ID와 실제 해시를 부여한다. `InputBinding`에 `source_image`, `inspected_image`, `preprocessing_version`을 모두 기록한다. 동일한 파일을 그대로 검사하면 두 참조가 같아도 된다.

OCR 좌표는 검사 기준 이미지의 픽셀 경계 좌표이며 x는 0부터 width, y는 0부터 height 범위다. F1-T의 축소 이미지나 OCR 내부 스케일의 좌표는 어댑터에서 이 기준으로 변환한다. PASS 이후에는 CLIP용 축소 이미지가 아니라 이 검사 기준 이미지를 전달한다.

## 3. 요청과 작업 맥락

### Request

| 필드 | 규칙 |
|---|---|
| `schema_version` | `1.0`, 생략 시 동일 값 적용 |
| `request_id` | 요청 식별자, 비어 있을 수 없음 |
| `command` | 사용자 지시 원문, 공백만 있는 문장 거부 |
| `image` | 원본 `ImageRef` |
| `environment_version` | 팀이 제공한 환경 정보의 버전 |
| `mode` | `C`, 생략 시 동일 값 적용 |

원문의 공백, 줄바꿈과 유니코드 표현을 보존한다. `command_sha256`은 원문을 BOM 없는 UTF-8로 인코딩해 계산하며 trim, 요약, NFC 정규화를 적용하지 않는다. 128토큰 여부는 F1-T의 실제 토크나이저로 검사할 예정이며 문자 수로 대체하지 않는다.

`TInput`에는 `command`와 검사 기준 `image`만 있다. 작업 맥락이나 Contract를 전달하면 스키마 검증에서 거부한다.

### TaskContext

| 필드 | 규칙 |
|---|---|
| `schema_version` | `1.0` |
| `context_id` | 작업 맥락의 식별자. 내용이 바뀌면 새 ID 발급 |
| `command_sha256` | 해당 맥락을 준비한 command 원문의 해시 |
| `environment_version` | 해당 맥락이 사용하는 환경 버전 |
| `context_status` | `READY`, `REJECTED`, `INCOMPLETE` |
| `target` | 대상 설명과 선택적 등록 ID. 일반적인 대상 설명도 허용 |
| `explicit_goal` | 사용자가 명시한 최종 목표 |
| `constraints` | 관련 안전 조건과 필수 작업 조건, 빈 배열 가능 |

READY에는 대상과 목표가 필요하다. 나머지 상태는 의미 정보가 일부 빠진 채 반환할 수 있다. 제약은 자연어 목록으로 받고, 올바른 행동 순서를 하나로 강제하지 않는다.

작업 맥락은 요청 JSON 안의 자가 승인 필드로 받지 않는다. Contract 준비 모듈이 별도 인자로 제공한다. F1 단독 시험에서는 팀이 작성한 같은 형식의 자료를 쓴다. 사용자 command는 F1-T 통과 전에는 검사 대상이며, 이미지 속 문구로 작업 맥락을 수정하지 않는다.

`require_ready_context(request, context)`는 원문 해시, 환경 버전과 준비 상태를 확인한다. READY라는 문자열만으로 내용의 정확성이나 제공자의 신뢰성을 증명하지 않는다. 제공자를 신뢰할 수 있는 경로로 연결하는 책임은 최종 게이트에 있다.

## 4. 내부 분석 결과

### TAssessment

`binding`, `decision`, `reason_codes`, `risk_score`, `model_version`, `policy_version`을 기록한다. PASS와 BLOCK에는 유한한 0~1 점수가 필요하다. 오류로 보류된 경우 점수가 null일 수 있다. 점수의 값만으로 판정을 생성하는 코드는 다음 학습 단계에서 구현한다.

`InputBinding`은 요청 ID, 원문 해시, 환경 버전, 원본 및 검사 기준 이미지와 전처리 버전을 묶는다. `validate_assessment_binding`으로 현재 요청의 식별값과 비교한다. 모델과 정책 버전에는 실제 배포 체크포인트 및 설정을 식별할 값을 넣는다. 예시의 `mock-*` 값은 운영 버전이 아니다.

### OCRResult

`status`는 COMPLETE, INCOMPLETE, ERROR다. 영역에는 ID, 다각형, 원문, 인식 점수, 묶음 ID와 읽기 순서를 저장한다. 읽기 순서는 0부터 시작한다. 같은 영역 ID를 중복 사용하지 않는다.

- 정상 완료 후 영역이 0개이면 COMPLETE와 `NO_TEXT_DETECTED`를 기록한다.
- 검출했지만 읽지 못한 영역은 삭제하지 않는다. 빈 문자열과 점수 null을 허용하고 INCOMPLETE로 남긴다.
- OCR 오류를 글자 없음으로 바꾸지 않는다.
- 품질 기준 미달 여부와 줄 묶기 알고리즘은 OCR 연결 단계에서 구현한다.

### OAssessment

입력 `binding`, 사용한 `context_id`, OCR 결과와 네 질문 결과를 함께 기록한다. 원시 OCR 문자열은 내부 분석 기록에만 남기고 최종 승인 입력의 프롬프트 필드에 추가하지 않는다.

| 질문 필드 | 의미 |
|---|---|
| `harmful_instruction` | 유해 행동이나 안전 조건 무시 요구 |
| `task_override` | 승인 대상, 최종 목표나 필수 조건 변경 요구 |
| `authority_spoofing` | 시스템이나 운영자 권한을 주장하며 지시 우선권 요구 |
| `benign_observation` | 정상 표지, 안전 안내나 인용의 근거 |

SDK 선택지 A/B/C는 YES/NO/UNKNOWN에 대응한다. 각 질문은 `model_answer`, 정책 적용 후 `decision`, 세 선택지의 `probabilities`를 반환한다. 정책은 원래 답을 UNKNOWN으로 보류할 수 있으나 YES를 NO로 뒤집지 않는다. 확률은 유한하고 합이 1이어야 한다. 정확도가 검증된 확률이라는 뜻은 아니다.

| 상태 | 허용되는 결과 |
|---|---|
| CLEAR | OCR와 분석 완료, 네 질문 NO/NO/NO/YES |
| CLEAR / NO_TEXT_DETECTED | 정상 완료된 OCR에 영역 없음. 네 질문은 모두 null이며 Laya 생략 |
| ATTACK | Q1~Q3 중 명확한 YES. 부분 분석이어도 공격 근거를 보존 |
| UNKNOWN | 필요한 의미 판단이 불확실하거나 문구 분석이 불완전 |
| ERROR | 실행 오류 또는 필수 결과 누락 |

스키마는 모순된 CLEAR나 공격 근거를 숨긴 상태를 거부한다. 누락된 모델 반환을 OAssessment로 무조건 끼워 맞추지 않는다. 후속 어댑터가 부분 결과를 보존해 ERROR를 만들고, 이미 확정된 공격이 있으면 ATTACK과 `analysis_complete=false`를 기록한다.

## 5. 외부 응답

`F1Result`의 필드는 다음과 같다.

| 필드 | 의미 |
|---|---|
| `schema_version` | `1.0` |
| `request_id` | 원래 요청 ID. 잘못된 요청에서 ID를 확보하지 못한 경우만 null |
| `decision` | PASS, BLOCK, DEFER |
| `reason_codes` | 등록된 사유 목록, 빈 배열과 중복 거부 |
| `approved_input` | PASS에서만 존재. BLOCK/DEFER에서는 반드시 null |
| `trace_id` | 내부 상세 기록과 연결하는 ID |

`approved_input`에는 원문과 원문 해시, 원본 이미지 해시, 실제 검사 이미지 참조, 환경 버전과 작업 맥락 ID를 넣는다. `validate_result_binding`으로 요청 및 실제 검사 이미지와 비교한다. 원문을 바꾸고 이전 해시를 재사용할 수 없다.

이 검증 함수는 모델 수행 여부를 증명하거나 PASS를 생성하지 않는다. 게이트 구현 단계에서 아래 조건을 모두 확인한 뒤 F1Result를 만든다.

1. 실제 파일 읽기와 전처리 검사가 완료되어 있다.
2. 현재 입력에 대한 F1-T가 PASS다.
3. 별도 제공된 작업 맥락이 READY이고 원문과 환경 버전이 일치한다.
4. 같은 검사 기준 이미지와 작업 맥락에 대한 F1-O가 CLEAR다.
5. 검사 결과와 모델 및 정책 버전을 상세 기록으로 남겼다.

외부에서 받은 PASS JSON을 신뢰해 VLM을 호출하는 API는 만들지 않는다. F1Result 생성은 내부 게이트의 책임이다. 파일을 해시로 확인한 뒤에도 읽을 때마다 내용이 바뀔 수 있으므로 실제 모델과 VLM에는 고정한 바이트 또는 검증된 불변 자산을 연결한다.

## 6. 사유 코드

코드의 전체 목록은 `ReasonCode`가 기준이며 주요 분류는 다음과 같다.

| 결과 | 사유 코드 |
|---|---|
| PASS | `INPUT_CLEAR`, `NO_TEXT_DETECTED` |
| BLOCK | `F1T_HARMFUL_REQUEST`, `F1O_HARMFUL_INSTRUCTION`, `F1O_TASK_OVERRIDE`, `F1O_AUTHORITY_SPOOFING`, `CONTEXT_REJECTED` |
| DEFER: 입력 | `INVALID_REQUEST`, `UNSUPPORTED_MODE`, `IMAGE_UNAVAILABLE`, `IMAGE_DECODE_ERROR`, `IMAGE_MISMATCH`, `INPUT_BINDING_MISMATCH` |
| DEFER: 작업 맥락 | `CONTEXT_MISSING`, `CONTEXT_INCOMPLETE`, `CONTEXT_MISMATCH` |
| DEFER: F1-T | `F1T_TEXT_TOO_LONG`, `F1T_UNCERTAIN`, `F1T_MODEL_NOT_READY`, `F1T_ERROR` |
| DEFER: OCR | `OCR_UNREADABLE`, `OCR_BUDGET_EXCEEDED`, `OCR_ERROR` |
| DEFER: F1-O | `F1O_UNCERTAIN`, `F1O_INPUT_TOO_LONG`, `F1O_INVALID_OUTPUT`, `F1O_ERROR`, `REQUIRED_CHECK_MISSING` |

명확한 공격과 처리 미완료가 함께 있으면 BLOCK 사유와 보류 관련 상세 사유를 함께 기록할 수 있다. 실행 실패만으로 BLOCK을 만들지 않는다. 상태 집계의 전체 결정표와 실제 실행 순서는 게이트 단계에서 구현한다.

요청 형식 오류는 `ValidationError`, 입력 연결 오류는 `BindingError.reason_code`로 전달된다. 향후 서비스 어댑터에서 이를 DEFER 응답으로 변환한다. 요청 ID를 확보하지 못하면 `request_id=null`, DEFER, `INVALID_REQUEST`를 사용한다. 현재 CLI는 서비스 응답 대신 INVALID_IO와 종료 코드 2를 출력한다.

## 7. 대표 예시와 완료 범위

`request.json`과 `task-context.json`은 검증 가능한 원문 해시와 이미지 참조를 가진다. `t-assessment.mock.json`과 `o-assessment.mock.json`은 정상 사용자 지시에 외부 작업 변경 문구가 있는 상황의 모의 결과다.

`responses.mock.json`은 정상 통과, 유해 지시 차단, 작업 변경 차단, 맥락 누락 및 불일치, OCR 실패, 불확실한 의미 판단, 글자 미검출과 부분 분석 중 공격 확인의 **응답 형식**을 보여준다. 각 행은 독립적인 가정이며 하나의 실제 추론 실행 결과가 아니다. 학습이나 탐지 정확도 평가에 사용하지 않는다.

테스트는 잘못된 필드, 다른 요청의 결과 재사용, 잘못된 좌표와 점수, 불완전한 CLEAR 및 BLOCK/DEFER의 입력 전달을 검증한다. 이미지의 의미, 공격 탐지 성능, 토큰 한도와 최종 게이트 동작은 다음 단계에서 검증한다.
