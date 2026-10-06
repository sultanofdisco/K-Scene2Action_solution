# J-DAPT의 데이터 사용 방식과 공개 데이터 확보 결과

확인일: 2026-10-05. 논문 v1과 공개 코드의 고정 리비전을 대조했다. 이번 작업은 원천 데이터 확보와 검증이며, 기존 F1-T 학습 데이터나 모델에 추가한 단계는 아니다.

## 1. 핵심 결론

J-DAPT는 DAQUAR에서 정상 질문과 이미지, JailBreakV의 JB28K에서 유해 요청과 이미지를 가져와 일반 도메인의 탐지 학습 자료로 사용한다. 이후 정상 로봇 환경 자료를 이용해 일반 데이터의 표현과 학습 가중치를 해당 환경에 맞춘다. 따라서 두 데이터셋을 많이 넣는 것만으로 논문의 성능을 재현할 수 있다고 해석하면 안 된다. [논문 IV 및 V-A](https://arxiv.org/html/2509.23281v1#S4)

확보한 데이터는 다음과 같다. 수량은 내려받은 파일을 직접 검사한 결과다.

| 자료 | 확보한 내용 | 현재 이미지까지 대응되는 행 | 의미 |
|---|---|---:|---|
| DAQUAR 전처리본 | 질문 12,468행, 원본 이미지 파일 1,449개 | 12,468행 | 정상 VQA 보조학습 후보 |
| JB28K | 전체 CSV 28,000행, 공개 이미지 파일 300개 | 360행 | 유해 멀티모달 보조학습 후보 |
| RedTeam-2K | 유해 질문 2,000행 | 이미지 없는 별도 질문 목록 | JB28K에 이미 포함된 원문이므로 추가 2,000개로 중복 계산하지 않음 |
| mini JB28K | 280행 | 3행 | 전체 CSV의 부분집합이며 별도 신규 자료가 아님 |

**JB28K 28,000행 전체의 이미지가 확보된 것은 아니다.** 공개 이미지 중 전체 CSV의 원래 경로와 대응되는 파일은 205개이고, 이를 공유하는 행이 360개다. 나머지 27,640행은 이미지 누락으로 표시했다. 공개 파일 300개는 바이트 해시 기준으로 269종이며, CSV와 대응되지 않는 95개 파일도 원본 자료로 보존했다.

공식 배포처는 전체 자료에 대해 별도 신청 링크를 안내한다. 지금 확보한 범위는 공개 Hugging Face 파일이다. [공식 데이터 카드](https://huggingface.co/datasets/JailbreakV-28K/JailBreakV-28k/blob/f949ca582fff13d396ac8fce59596afafb2b78d3/README.md), [전체 데이터 신청](https://forms.gle/e6boFoM9mNfPHb9H9)

## 2. 논문과 공개 코드에서 정확히 무엇을 사용했나

### 논문에 명시된 구성

| 자료 | 논문에 적힌 규모 | 탐지 실험에서의 역할 |
|---|---:|---|
| DAQUAR | train 6,794개, test 5,674개 | 질문을 safe로 취급 |
| JB28K | 28,000개: 텍스트 기반 전이 공격 20,000개, 이미지 기반 공격 8,000개 | red-team 요청과 jailbreak 요청을 unsafe로 취급 |

위 DAQUAR train/test 숫자는 데이터셋 설명이다. 이 분할을 그대로 최종 탐지기의 학습과 평가에 사용했다고 단정할 수는 없다. [논문 V-A1](https://arxiv.org/html/2509.23281v1#S5.SS1.SSS1)

### DAQUAR를 읽는 실제 코드

J-DAPT의 다운로드 설정은 `tezansahu/processed-daquar-dataset`이라는 Kaggle 전처리본을 가리킨다. 이번에는 그 작성자의 GitHub CSV와 Hugging Face 이미지 미러를 확보했다. 두 배포처의 `data.csv`, `data_train.csv`, `data_eval.csv`, `all_qa_pairs.txt`는 SHA-256이 모두 일치했다. [J-DAPT 설정](https://github.com/Mhackiori/J-DAPT/blob/32a2d1da5eb7feb4099728e42ae6f174ab70ba85/utils/params.py#L16), [전처리본 작성자 GitHub](https://github.com/tezansahu/VQA-With-Multimodal-Transformers/tree/e737e61ffa1553b6593579b7c296fbd724c782f8), [이미지 미러](https://huggingface.co/datasets/sudiphalder/daquar/tree/a558b3927dcf7136b06e2789eaea648a788ca9c0)

공개 코드의 `embedding_vqa`는 다음 방식으로 처리한다.

1. 전처리본의 `data.csv`를 읽는다.
2. 각 행의 `question`과 `images/{image_id}.png`를 가져온다.
3. 질문 임베딩, 이미지 임베딩, 융합 임베딩을 저장한다.
4. 로더가 이 항목을 정상 사례로 취급한다.

**정답인 `answer`를 탐지기 입력이나 예측 목표로 사용하지 않는다.** 원래 VQA 문제의 정답을 맞히는 학습이 아니다. 또한 이 함수는 `data_train.csv`와 `data_eval.csv`를 읽지 않는다. [DAQUAR 임베딩 코드](https://github.com/Mhackiori/J-DAPT/blob/32a2d1da5eb7feb4099728e42ae6f174ab70ba85/embeddings.py#L452)

### JB28K를 읽는 실제 코드

`embedding_jb28k`는 전체 `JailBreakV_28K.csv`의 각 행에서 다음 두 입력을 만든다.

```text
같은 행의 원래 이미지 + jailbreak_query → 우회 요청의 임베딩
같은 행의 원래 이미지 + redteam_query   → 원래 유해 요청의 임베딩
```

두 종류 모두 unsafe로 로드한다. 전체 이미지가 있고 모든 행을 처리한다면 28,000행에서 최대 56,000개의 유해 입력 표현을 만들 수 있는 코드다. 하지만 같은 원래 요청이 반복되므로 **56,000개의 독립적인 유해 상황이라는 뜻은 아니다.** 실제 확보한 CSV의 고유 `redteam_query`는 2,000개이고 고유 `jailbreak_query`는 8,641개다. [JB28K 임베딩 코드](https://github.com/Mhackiori/J-DAPT/blob/32a2d1da5eb7feb4099728e42ae6f174ab70ba85/embeddings.py#L250)

현재 이미지가 있는 360행에는 고유 red-team 문장 190개와 고유 jailbreak 문장 304개가 포함된다. 텍스트가 같아도 이미지가 다르면 별도 멀티모달 입력일 수 있으므로 텍스트 중복만으로 삭제하지 않았다.

### 융합 모델 학습과 최종 분류기 학습은 별개다

공개 코드의 `train_cross_attention_model`은 두 CSV 중 작은 행 수를 `cap`으로 잡고, 각각 **앞에서부터 `cap`행**을 사용한다. JB28K에서는 `jailbreak_query`, DAQUAR에서는 `question`을 원래 이미지와 연결한다. 두 임베딩의 평균을 목표로 MSE를 최소화하면서 융합층을 학습한다. 해당 함수가 최적화하는 것은 융합 모델이며, CLIP 전체를 유해성 라벨로 미세조정하는 코드가 아니다. [융합 학습 코드](https://github.com/Mhackiori/J-DAPT/blob/32a2d1da5eb7feb4099728e42ae6f174ab70ba85/embeddings.py#L25)

전체 CSV가 그대로 입력되었다면 각 12,468개, 합계 24,936개를 융합 학습에 쓰는 계산이다. 이는 코드 조건으로부터 계산한 수이며, 논문의 실제 실행에서 선택된 샘플 목록을 확보했다는 뜻은 아니다.

최종 분류기 단계에는 일반 자료의 클래스 균형 조정, 도메인 관련도에 따른 가중치, CORAL을 통한 표현 정렬이 포함된다. 대상 로봇 도메인의 정상 자료는 적응에 이용하고, 로봇 공격 사례는 평가에 사용하는 것이 논문의 설정이다. [논문 IV-B 및 V-B](https://arxiv.org/html/2509.23281v1#S4.SS2)

공개 저장소에는 최종 실행별 샘플 ID 목록이 없어, 최종 분류기가 정확히 어느 행을 몇 번 학습했는지까지는 확정할 수 없다. 노트북에는 별도 균형 조정과 샘플 수 제한도 있다. 공개 `jdapt-car` 노트북의 `label='bq+jb'` 호출은 로더의 허용 값과 맞지 않아, 실행 순서와 코드 상태를 확인하지 않고 그대로 재현 코드로 사용하는 것도 주의가 필요하다. [노트북](https://github.com/Mhackiori/J-DAPT/blob/32a2d1da5eb7feb4099728e42ae6f174ab70ba85/jdapt-car.ipynb), [로더](https://github.com/Mhackiori/J-DAPT/blob/32a2d1da5eb7feb4099728e42ae6f174ab70ba85/utils/helpers.py#L156)

## 3. 내려받은 자료에서 확인한 주의점

### DAQUAR 분할을 그대로 재사용하면 안 된다

전처리본은 질문 행을 무작위 80:20으로 나눠 train 9,974행, eval 2,494행으로 저장한다. 양쪽에 **동일 이미지 1,124개**가 겹친다. 따라서 새로운 장면에 대한 성능을 측정할 때 이 분할을 그대로 쓰면 장면 분리가 되지 않는다. [전처리 코드](https://github.com/tezansahu/VQA-With-Multimodal-Transformers/blob/e737e61ffa1553b6593579b7c296fbd724c782f8/src/process_dataset.py#L50)

비교를 위해 DAQUAR 공식 원본 주석도 받았다. 현재 공식 파일은 train 6,795쌍, test 5,673쌍으로 집계되어 논문에 적힌 숫자와 각각 1개 차이가 난다. 합계는 동일하게 12,468개다. 이 공식 분할은 이미지가 겹치지 않는다. 원본 문장의 오탈자 때문에 작성자 정규화 규칙으로 정확히 대응되지 않는 1행도 기록했다. 원본을 임의로 수정하거나 숫자를 논문에 맞추지 않았다. [공식 배포 페이지](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/vision-and-language/visual-turing-challenge)

질문, 정답, 이미지 ID가 완전히 같은 중복의 초과 행은 210개다. 1,449개 이미지 중 CSV가 실제 참조하는 것은 1,447개이며, 12,468행 모두 해당 이미지가 정상적으로 열린다. 노트북 체크포인트와 macOS 메타데이터 사본은 통계에서 제외했다.

### JB28K 이미지 유형과 우리 작업의 차이

| 공격 형식 | 전체 행 | 원래 이미지 확보 행 |
|---|---:|---:|
| Template | 18,336 | 200 |
| Persuade | 1,368 | 9 |
| Logic | 296 | 1 |
| figstep | 2,000 | 50 |
| SD_typo | 2,000 | 35 |
| SD | 2,000 | 34 |
| typo | 2,000 | 31 |
| 합계 | 28,000 | 360 |

일부 이미지는 빈 화면, 자연 이미지, 글자가 들어간 이미지 등이다. 전부 로봇이 보는 물리 환경 사진이라고 가정하면 안 된다. DAQUAR 사진을 정상, JB28K의 글자 이미지를 공격으로만 주면 모델이 위험 판단 대신 데이터 출처나 이미지 형식을 구별하는 지름길을 학습할 가능성이 있다. 이는 우리 학습 설계에서 점검해야 할 위험이다. [JB28K 구성 설명](https://huggingface.co/datasets/JailbreakV-28K/JailBreakV-28k/blob/f949ca582fff13d396ac8fce59596afafb2b78d3/README.md)

### 현재 다국어 CLIP의 입력 길이도 확인했다

기존 프로젝트의 텍스트 토크나이저를 사용해 특수 토큰을 포함한 길이를 측정했다. 모델 가중치를 불러와 학습하거나 임베딩을 추출한 것은 아니다.

| 텍스트 | 행 수 | 현재 한도 128토큰 초과 |
|---|---:|---:|
| JB28K 전체 `jailbreak_query` | 28,000 | 11,892 |
| 이미지 확보 360행의 `jailbreak_query` | 360 | 128 |
| RedTeam-2K의 `question` | 2,000 | 3 |
| DAQUAR의 `question` | 12,468 | 0 |

장문 공격을 앞부분만 잘라 넣으면 판정에 필요한 요청이 사라질 수 있다. 따라서 이미지가 있는 360행도 전부 현재 F1-T에 즉시 넣을 수 있는 확정 학습 자료는 아니다. 장문 처리와 라벨의 적용 범위를 별도로 결정해야 한다.

### 라벨 숫자와 적용 범위

J-DAPT 로더는 `0=유해`, `1=정상`이다. 우리 프로젝트의 `0=정상`, `1=유해`와 반대다. 이번 목록에는 의미가 명확한 `source_safety_role`을 기록하고 학습용 숫자 라벨로 자동 변환하지 않았다. [라벨 코드](https://github.com/Mhackiori/J-DAPT/blob/32a2d1da5eb7feb4099728e42ae6f174ab70ba85/utils/helpers.py#L193)

JB28K의 16개 정책 범주는 우리 프로젝트의 물리적 위험 기준보다 넓다. 원본 전체를 보관하는 것과 학습에 전부 채택하는 것은 다르다. 이전에 제외한 윤리와 사회적 내용, 권한 관련 범위를 되살리는 자동 선별도 하지 않았다. DAQUAR의 safe 라벨 역시 질문 자체를 정상으로 본다는 뜻이지 안전한 로봇 행동이 검증되었다는 뜻은 아니다.

## 4. 우리 F1에 어떻게 활용하는 것이 적절한가

1. **DAQUAR는 정상 이미지와 질문을 포함한 일반 도메인 보조 자료로 활용한다.** 우리 환경의 정상 작업 명령을 대체하지 않는다. 이미지 ID와 중복 문항을 기준으로 먼저 분할한다.
2. **JB28K는 이미지가 실제 대응되는 360행부터 내용과 길이를 검토한다.** 채택 시 원래 red-team 질문과 변형 요청이 서로 다른 평가 분할에 섞이지 않게 묶는다. 동일 이미지 해시와 공유 템플릿도 분할 점검에 사용한다.
3. **이미지 누락 27,640행은 텍스트 원천으로만 취급한다.** 임의의 우리 환경 이미지나 빈 이미지를 붙여 원래 멀티모달 자료인 것처럼 만들지 않는다.
4. **figstep과 typo 계열은 F1-O 검증 후보로도 분리할 수 있다.** 다만 요청 맥락과 OCR 결과에 맞는 라벨 검토가 필요하며, 현재 자료에 Laya 네 질문의 정답이 이미 있다고 가정하지 않는다.
5. **기존 우리 환경 평가셋을 고정하고 추가 자료의 효과를 비교한다.** 다국어 CLIP은 유지하되 일반 자료를 넣은 실험과 도메인 적응을 적용한 실험을 구분해야 데이터량의 효과와 방법의 효과를 판단할 수 있다.

## 5. 저장 위치와 재사용

프로젝트 기준 위치: `data/jdapt_public_sources_v1/`

| 경로 | 내용 |
|---|---|
| `raw/jailbreakv/` | 공식 HF 공개 CSV와 이미지 |
| `raw/daquar_hf/extracted/dataset/` | DAQUAR 이미지와 전처리 CSV |
| `raw/daquar_official/` | 공식 train/test 원문과 배포 페이지 |
| `raw/daquar_processed_github/` | 작성자의 CSV와 전처리 코드 원문 |
| `manifests/jailbreakv_inventory.jsonl` | 전체 28,000행과 이미지 가용성 |
| `manifests/jailbreakv_with_images.jsonl` | 이미지 대응을 검증한 360행 |
| `manifests/jailbreakv_missing_images.csv` | 누락된 이미지 경로와 원래 행 ID |
| `manifests/redteam2k_inventory.jsonl` | 원래 유해 질문 2,000행 |
| `manifests/daquar_inventory.jsonl` | 12,468행과 이미지 및 분할 대응 정보 |
| `manifests/image_inventory.jsonl` | 이미지의 크기, 해시, 디코딩 검사 결과 |
| `audit.json` | 상세 집계와 입력 길이 검사 결과 |
| `provenance/` | 고정 리비전, 원본 해시, J-DAPT 코드 사본 |

JSONL의 `image.image_path`는 이 자료 폴더를 기준으로 한 상대 경로다. `training_selected=false`는 학습에 선별되지 않았다는 뜻이다. `review_status=NOT_PROJECT_REVIEWED`는 우리 위험 기준으로 내용 검토가 완료되지 않았다는 뜻이다. DAQUAR의 `answer_metadata_only`는 추적을 위한 메타데이터이며 탐지기 입력이 아니다.

공개 파일의 다운로드 리비전과 SHA-256을 기록했고, HF LFS 파일은 원격에 등록된 해시와 대조했다. 이미지 검사는 경로와 디코딩의 기술적 유효성 검사이며, 공격 라벨의 의미가 우리 환경에 맞는지까지 검증한 것은 아니다.

다운로드 스크립트는 `scripts/download_jdapt_sources.py`, 목록 생성 및 검증 스크립트는 `scripts/audit_jdapt_sources.py`다. 전자는 데이터 배포처를 다시 받는 용도이고, `provenance/`의 J-DAPT 연구 코드 사본은 이번 조사에서 별도로 보존한 자료다.

JB28K의 HF 데이터 카드는 MIT로 표시되어 있다. DAQUAR는 원 연구 및 이미지 원천의 이용 조건을 별도로 확인해야 하며, 전처리 코드 저장소의 라이선스를 이미지 전체의 라이선스로 간주하지 않았다.
