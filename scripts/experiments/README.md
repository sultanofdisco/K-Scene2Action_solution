# 이전 F1-T 실험 실행 코드

기존 로컬 실험 폴더에만 있던 실행 스크립트를 보존한다. 원본 실행 코드는 수정하지 않았다. 각 스크립트에서 참조하는 데이터와 이전 실험 캐시는 별도로 준비해야 하며, 이 폴더 자체에 데이터나 모델 가중치는 포함하지 않는다.

최신 J-DAPT 학습은 `scripts/train_jdapt_local.py`, 결과 생성은 `scripts/report_jdapt_local.py`, 저장 모델 재평가는 `scripts/evaluate_jdapt_local.py`를 사용한다. 실행 조건과 한계는 `docs/f1t-jdapt-local-training.md`에 정리되어 있다.
