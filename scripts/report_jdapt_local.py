"""Render the completed, fixed-plan local J-DAPT comparison without selecting a model."""
import argparse
import json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--report',type=Path,required=True)
    args=ap.parse_args();run=args.run
    s=json.loads((run/'summary.json').read_text(encoding='utf-8'))
    plan=json.loads((run/'experiment-plan.json').read_text(encoding='utf-8'))
    names={'existing_mlp':'기존 MLP','jdapt_concat':'J-DAPT 분류기만','jdapt_fusion':'J-DAPT 융합 + 분류기','jdapt_fusion_coral_weighted':'J-DAPT 융합 + 도메인 적응 + 분류기'}
    cohorts={'local_same_scene_hazard':'같은 장면에서 위험 유형 분리: 로컬 1,130건','local_scene_held_out':'로컬 장면 전체 제외: 로컬 1,130건','rjb_group_diagnostic':'RoboJailBench 그룹 분리: 204건'}
    keyacc='accuracy_at_0_5';keyfp='normal_false_block_rate_at_0_5';keyfn='harmful_miss_rate_at_0_5'
    base=s['comparison']['existing_mlp']['local_same_scene_hazard']['mean']
    adapted=s['comparison']['jdapt_fusion_coral_weighted']['local_same_scene_hazard']['mean']
    delta=100*(adapted[keyacc]-base[keyacc])
    lines=['# J-DAPT 기반 우리 환경용 F1-T 학습 결과','',
           '실험일: 2026-10-06 / CPU 4스레드 / 다국어 CLIP 고정 / 3개 seed 평균','',
           f'**같은 장면의 새 위험 유형 평가에서 기존 MLP 정확도는 {100*base[keyacc]:.1f}%, J-DAPT 융합과 도메인 적응을 적용한 모델은 {100*adapted[keyacc]:.1f}%로 {delta:+.1f}%p 차이였다.** FPR은 {100*base[keyfp]:.1f}%에서 {100*adapted[keyfp]:.1f}%, FNR은 {100*base[keyfn]:.1f}%에서 {100*adapted[keyfn]:.1f}%로 변했다. 아래 결과는 진단 평가이며 운영 모델은 교체하지 않았다.','',
           '## 무엇을 학습했나','',
           '공식 J-DAPT의 `CrossAttentionFusion`과 `SimpleNN` 클래스를 그대로 사용했다. 다국어 CLIP이 만든 텍스트 및 이미지 벡터로 융합층을 학습하고, **텍스트 512 + 이미지 512 + 융합 512 = 1,536차원**을 최종 MLP에 입력했다. 원본 벡터를 버리고 융합 벡터만 사용하는 구조가 아니다.','',
           '| 비교 방식 | 학습 내용 |','| --- | --- |',
           '| 기존 MLP | 기존 1,024 → 128 → 1 구조, 30 epochs. 동일 자료와 분할 및 seed의 저장된 예측을 사용하고 1개 fold를 재학습해 일치를 확인 |',
           '| J-DAPT 분류기만 | 원본 텍스트와 이미지 1,024차원을 공식 SimpleNN의 512차원 은닉층과 2개 출력에 입력, 10 epochs |',
           '| J-DAPT 융합 + 분류기 | 공식 융합층 20 epochs 후 고정, 원본 벡터와 합친 1,536차원으로 SimpleNN 10 epochs |',
           '| J-DAPT 융합 + 도메인 적응 + 분류기 | RoboJailBench 학습 표현을 로컬 정상 학습 자료의 분포에 맞추고 샘플 가중치를 적용. 로컬 학습 자료의 정상 및 위험 라벨도 함께 사용 |','',
           '융합층은 텍스트와 이미지 임베딩의 평균에 가까운 표현을 만들도록 MSE로 학습했다. 도메인 적응은 CORAL로 표현의 평균과 공분산을 맞추고, 도메인 분류기로 로컬 표현에 가까운 원천 샘플의 학습 비중을 조정한다. 모든 설정은 평가 전에 고정했으며 결과를 보고 epoch나 임계값을 고르지 않았다.','',
           '## 사용 자료와 평가 조건','',
           '| 출처 | 이미지 + 명령 사례 수 |','| --- | ---: |',
           '| RoboJailBench | 204 |','| 이재우 v2 | 946 |','| 이성민 v8 | 184 |','| 합계 | 1,334 |','',
           '정상과 위험은 각각 667건이다. 각 평가 fold에서는 평가용 사례를 먼저 제외하고 남은 자료로만 융합층과 분류기를 학습했다. CORAL과 도메인 분류기에도 해당 fold의 학습 자료만 사용했다. 같은 장면 평가는 12개 위험 유형을 차례로 제외하고, 장면 분리 평가는 5개 그룹을 차례로 제외했다.','',
           '**이번에는 DAQUAR와 JB28K 대신 기존에 선별한 RoboJailBench와 로컬 자료를 사용했다.** SafeAgentBench와 SafeText 보조학습도 제외해 이미지 + 명령 비교 기준을 맞췄다. 따라서 논문의 전체 학습 재현이나 저자 체크포인트 평가는 아니며, 공개 학습 코드를 우리 자료에 적용한 실험이다.','',
           '정확도는 전체 평가 사례 중 정답과 일치한 비율이다. **FPR = 정상 중 위험 판정 / 전체 정상**, **FNR = 위험 중 정상 판정 / 전체 위험**이다. 위험 점수 0.5 이상을 위험으로 분류하며 이번 이진 평가에는 보류가 없다. 각 비율은 seed 42, 7, 19의 평균이고, 같은 사례를 세 번 평가한 것이므로 독립 사례 수가 세 배로 늘어난 것은 아니다.','']
    for c,title in cohorts.items():
        lines += ['## '+title,'','| 모델 | 정확도 | FPR | FNR |','| --- | ---: | ---: | ---: |']
        for k,label in names.items():
            m=s['comparison'][k][c]['mean']
            lines.append(f'| {label} | {100*m[keyacc]:.1f}% | {100*m[keyfp]:.1f}% | {100*m[keyfn]:.1f}% |')
        lines.append('')
        if c=='local_same_scene_hazard':
            lines += ['정상 565건과 위험 565건이며 학습과 평가가 같은 장면 이미지를 공유한다. 새로운 장면에 대한 일반화 성능으로 해석하지 않는다.','']
        elif c=='local_scene_held_out':
            lines += ['로컬 장면 전체를 제외하면 로컬 정상 참조 자료도 없어 **도메인 적응을 수행할 수 없다.** 이 조건의 마지막 행은 융합 모델로 대체한 결과이며, CORAL을 수행한 새 장면 성능이 아니다. 평가 장면을 적응에 미리 사용하는 것은 금지했다.','']
        else:
            lines += ['정상 102건과 위험 102건이다. 기존 5그룹 분할에서 RoboJailBench에 해당하는 4개 그룹의 평가 예측만 모았다.','']
    lines += ['## seed별 로컬 위험 유형 평가','', '| 모델 | seed 42 정확도 | seed 7 정확도 | seed 19 정확도 |','| --- | ---: | ---: | ---: |']
    for k,label in names.items():
        vals=s['comparison'][k]['local_same_scene_hazard']['per_seed']
        lines.append('| '+label+' | '+' | '.join(f"{100*vals[str(seed)][keyacc]:.1f}%" for seed in plan['seeds'])+' |')
    lines += ['','## 공개 코드에서 수정한 부분','',
              '- 샘플 가중치가 실제로 각 샘플에 적용되도록 Cross Entropy를 개별 손실로 계산하고 배치의 샘플 인덱스와 연결했다.',
              '- 서로 다른 source와 target 행의 확률을 나누는 대신, 같은 source 샘플에서 도메인 확률의 비율을 계산했다. 가중치는 사전에 정한 범위로 제한했다.',
              '- CORAL의 공분산 변환에 고유벡터의 역회전을 포함했다. 동일 분포 입력을 보존하는지와 공분산 정렬을 수치 검사했다.',
              '- 프로젝트 라벨은 0=정상, 1=위험을 유지했다. 공식 코드와 숫자 의미가 반대이므로 예측 점수의 위험 클래스를 명시했다.','',
              '이 수정들과 다국어 인코더, 데이터 구성, 로컬 라벨 사용은 원 논문 설정과의 차이다. 성능 차이 전체를 융합 또는 CORAL 하나의 효과로 단정하지 않는다. 융합 효과는 같은 SimpleNN을 쓰는 둘째와 셋째 행으로, CORAL과 가중치의 결합 효과는 셋째와 넷째 행으로 비교한다.','',
              '## 저장과 검증','',
              f'- 실행 시간: {s["elapsed_seconds"]/60:.1f}분. 임베딩은 기존 캐시를 재사용했다.',
              '- fold별 가중치와 예측, 분할 목록, 학습 손실을 저장했다. 전체 1,334건으로 학습한 후보도 seed별로 저장했다.',
              '- 전체 학습 후보는 저장 후 다시 로드했을 때 점수가 일치함을 확인했다. 기존 체크포인트의 해시도 유지됐다.',
              '- 기존 진단 세트를 다시 사용했고 일부 라벨과 이미지 출처에 기존 검토 한계가 남아 있다. 새로운 독립 시험과 실환경 성능 검증은 별도로 필요하다.','',
              f'[상세 수치]({(run/"summary.json").as_posix()}) | [학습 계획]({(run/"experiment-plan.json").as_posix()}) | [원본 코드](https://github.com/Mhackiori/J-DAPT/tree/32a2d1da5eb7feb4099728e42ae6f174ab70ba85)','',
              f'전체 학습 후보 위치: `{(run/"seed-42/full-fit").as_posix()}`. `fusion.safetensors`와 선택한 분류기 파일을 함께 사용한다. target 추론 입력에 CORAL 학습 변환을 다시 적용하지 않는다. 운영 필터로 자동 연결하지 않았다.','']
    text='\n'.join(lines)
    assert not any(c in text for c in '\u00b7\u318d\u30fb')
    args.report.write_text(text,encoding='utf-8')
    (run/'README.md').write_text(text,encoding='utf-8')
    print(args.report)


if __name__=='__main__':main()
