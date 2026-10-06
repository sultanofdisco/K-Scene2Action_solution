"""Audit and summarize the two frozen-data Laya CPU experiments."""
import csv, hashlib, html, json, os, runpy
from collections import Counter
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
ROOT = Path('artifacts/laya')
OUT = ROOT / 'deepset-decision-cpu-v1'
DATA = Path('data/laya_deepset_domain_v1')
Q = ['harmful_instruction', 'task_override', 'authority_spoofing', 'benign_observation']

def read(p):
    return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]

def dump(p, v):
    p.write_text(json.dumps(v, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def pct(v):
    return '-' if v is None else f'{100*v:.1f}%'

def render(title, description, headers, rows):
    e = lambda x: html.escape(str(x))
    return '<!doctype html><html lang="ko"><meta charset="utf-8"><title>' + e(title) + '</title><style>body{font:16px system-ui;margin:32px;color:#142334;background:#f5f7fb}h1{font-size:26px}table{border-collapse:collapse;width:100%;background:white}th,td{border:1px solid #d6dfea;padding:10px;text-align:left;vertical-align:top;white-space:pre-wrap}th{position:sticky;top:0;background:#dae6f5}p{line-height:1.7;max-width:1100px}</style><h1>' + e(title) + '</h1><p>' + e(description) + '</p><table><thead><tr>' + ''.join('<th>'+e(x)+'</th>' for x in headers) + '</tr></thead><tbody>' + ''.join('<tr>'+''.join('<td>'+e(x)+'</td>' for x in row)+'</tr>' for row in rows) + '</tbody></table></html>'

def main():
    lock = json.loads((DATA/'data-lock.json').read_text(encoding='utf-8'))
    scorer = json.loads((ROOT/'deepset-scorer-cpu-v1/summary.json').read_text(encoding='utf-8'))
    decision = json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    assert 'completed_utc' in scorer and 'completed_utc' in decision
    checks = {}
    for split in ['train', 'validation']:
        checks[split+'_hash'] = sha(DATA/(split+'.jsonl')) == lock['splits'][split]['sha256']
    for p, digest in lock['protected_evaluation'].items():
        checks[p+'_unchanged'] = sha(Path(p)) == digest
    checks['source_unchanged'] = sha(Path('data/public_f1_bundle_v1/source/deepset/train.jsonl')) == lock['source_file_sha256']
    cases = read(DATA/'train.jsonl') + read(DATA/'validation.jsonl')
    groups = {}
    for c in cases:
        groups.setdefault(c['group_id'],set()).add(c['split'])
    checks['groups_do_not_cross_splits'] = all(len(v)==1 for v in groups.values())
    fields = ['case_id','split','origin','source_case_id','command','ocr_text','q1','q2','q3','q4','expected_status','rationale','human_reviewed']
    review = [[c['case_id'],c['split'],c['origin'],(c.get('source_row') or {}).get('case_id',''),c['command'],'\n'.join(c['ocr_texts']),*[c['expected_questions'][q] for q in Q],c['expected_status'],c['rationale'],False] for c in cases]
    with (DATA/'review.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(fields);w.writerows(review)
    (DATA/'review.html').write_text(render('F1-O 학습 데이터 검토표','공개 원문 23건, 인용 변형 15건, 합성 42건. Codex가 작성한 정답이며 사람 검토 전이다. OCR를 거치지 않은 텍스트 실험이다.',fields,review),encoding='utf-8')
    detail=[]; metric_rows=[]; question_rows=[]; language_rows=[]; all_results={}
    for name,base in [('prior-gold','development-v2'),('prior-validation','validation-v2')]:
        base_rows=read(ROOT/'prompt-comparison-cpu-v2'/base/'predictions.jsonl')
        runs={'기본 모델':base_rows,'채점부 학습':read(ROOT/'deepset-scorer-cpu-v1'/name/'predictions.jsonl'),'판단층과 채점부 학습':read(OUT/name/'predictions.jsonl')}
        all_results[name]=runs
        assert all(len(v)==len(base_rows) for v in runs.values())
        for label,rows in runs.items():
            sem=[r for r in rows if r['suite']=='semantic']
            met=runpy.run_path('scripts/evaluate_laya_gold.py')['metrics'](sem)
            metric_rows.append([name,label,str(met['n']),pct(met['accuracy']),pct(met['benign_clear_rate']),pct(met['attack_recall']),pct(met['attack_clear_rate']),str(met['abstain_or_error_count'])])
            question_rows.append([name,label,*[f"{sum(q not in r['question_errors'] for r in sem)}/{len(sem)}" for q in Q]])
            for lang in ['en','ko']:
                m=runpy.run_path('scripts/evaluate_laya_gold.py')['metrics']([r for r in sem if r['language']==lang])
                language_rows.append([name,label,lang,pct(m['accuracy']),pct(m['benign_clear_rate']),pct(m['attack_recall'])])
            if label!='기본 모델':
                folder=(ROOT/'deepset-scorer-cpu-v1' if label=='채점부 학습' else OUT)/name
                traces=[json.loads((folder/(r['case_id']+'-trace.json')).read_text(encoding='utf-8')) for r in rows]
                checks[name+label+'_complete'] = all(r['analysis_complete'] for r in sem)
                checks[name+label+'_no_truncation'] = all(not q['truncated'] for t in traces if t['laya_executed'] for q in t['token_budget'].values())
                checks[name+label+'_guards'] = all(r.get('guard_execution_correct',True) for r in rows) and all(r['status_correct'] for r in rows if r['suite']=='guard')
        by_id={label:{r['case_id']:r for r in rows} for label,rows in runs.items()}
        for r in base_rows:
            variants=[by_id[label][r['case_id']] for label in runs]
            detail.append([name,r['case_id'],r['suite'],r['language'],r['command'],'\n'.join(r['ocr_texts']),r['expected_status'],*[v['actual_status'] for v in variants],json.dumps(r.get('expected_questions'),ensure_ascii=False),json.dumps({q:(variants[-1]['answers'].get(q) or {}).get('decision') for q in Q},ensure_ascii=False)])
    # Verify exported weights through a new backend instance loaded from disk.
    from f1.laya.finetuned import DecisionHeadTunedLayaBackend
    from f1.laya.questions_v2 import QUESTIONS
    from f1.laya.pipeline import assess
    helper=runpy.run_path('scripts/compare_laya_prompts.py')
    backend=DecisionHeadTunedLayaBackend(OUT,cpu_threads=4)
    adapter=helper['CandidateBackend'](backend,QUESTIONS)
    reload_results=[]
    for case_id in ['B01-en','A01-en']:
        c=next(r for r in all_results['prior-gold']['판단층과 채점부 학습'] if r['case_id']==case_id)
        trace={};result=assess(*helper['inputs'](c),adapter,trace)
        exact=str(result.status)==c['actual_status'] and result.questions.model_dump(mode='json')==c['answers']
        checks['fresh_reload_'+case_id]=exact
        reload_results.append({'case_id':case_id,'exact_prediction_match':exact,'status':str(result.status)})
    checks['all_passed']=all(checks.values())
    dump(OUT/'verification.json',{'checks':checks,'tests':'39 passed','fresh_reload':reload_results})
    assert checks['all_passed'],checks
    headers=['세트','모델','의미 사례 수','정확도','정상 통과율','공격 탐지율','공격 통과율','UNKNOWN/ERROR 수']
    description='두 학습 실험과 기본 모델의 같은 사례 비교. 기존 오답을 알고 만든 학습 세트이므로 독립 외부 평가가 아니다. UNKNOWN은 정상 통과가 아니다. 이미지와 OCR 인식 성능은 측정하지 않았다.'
    page=render('Laya F1-O 추가 학습 결과',description,headers,metric_rows)
    page=page.replace('</html>','<h2>사례별 비교</h2>'+render('사례별 결과','네 질문의 정답과 마지막 모델 답변을 비교할 수 있다.',['세트','ID','유형','언어','승인 명령','관측 문자열','정답','기본','채점부','판단층+채점부','질문 정답','질문 예측'],detail).split('<table>',1)[1].replace('<thead>','<table><thead>',1))
    (OUT/'comparison.html').write_text(page,encoding='utf-8')
    def table(headers,rows):return '| '+' | '.join(headers)+' |\n|'+ '|'.join(['---']*len(headers))+'|\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)
    report='''# Laya F1-O 공개 데이터 추가 학습 결과

실제로 가중치를 업데이트한 두 CPU 실험이다. 단순 질문 수정이나 임계값 보정이 아니다. **이번 소규모 데이터와 학습 설정에서는 실용적인 성능 개선을 확인하지 못했다.** 기본 운영 연결은 변경하지 않았으며 학습 가중치는 별도 후보로 저장했다.

## 사용 데이터

보류해 둔 [deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections)의 train에서 공개 원문 23건을 선별했다. 공격 원문의 교육용 인용 변형 15건과 프로젝트 맥락의 합성 예시 42건을 더해 총 80건으로 만들었다. 58건으로 학습하고 22건으로 체크포인트를 선택했다. 공개 원문 80건 또는 독립 학습 사례 232건이라고 해석하면 안 된다.

공개 원문의 이진 라벨을 그대로 네 질문의 답으로 복사하지 않았다. 승인 명령과 목표를 예시로 부여하고 유해 행동, 작업 변경, 권한 사칭, 정상 관측을 각각 검토했다. 정답은 Codex 작성이며 아직 사람 검토 전이다. 원본 test 116건, CyberSecEval3, SIUO는 사용하지 않았다.

학습은 영어 문자열만 사용했다. 실제 이미지에서 OCR를 수행한 학습이 아니며 관측 텍스트가 정확히 추출됐다고 가정한다. 데이터 설명과 검토표는 `data/laya_deepset_domain_v1/README.md`, `review.html`, `review.csv`에 있다.

## 학습 방법

기본 모델은 `convaiinnovations/laya-multilingual`, revision `1720e3e3357cfe1e281542e223f8273b0890ca34`, SDK 0.3.27이다. 질문은 실험용 V2 네 질문으로 고정했다. 통합 규칙도 그대로 유지했다. 앞의 세 질문 중 YES가 있으면 ATTACK, NO/NO/NO/YES이면 CLEAR, 나머지는 UNKNOWN이다.

1. 채점부 실험: 인코더와 판단층을 고정하고 Laya 원래 scorer의 592,897개 파라미터를 30 epoch 학습했다. AdamW, 학습률 1e-4, 배치 32를 사용했다. 정상 검증 사례를 하나도 통과시키지 못해 학습 범위를 확대한 두 번째 실험을 진행했다.
2. 판단층 실험: 원래 기본 모델에서 새로 시작해 두 Transformer 판단층과 scorer를 함께 10 epoch 학습했다. 인코더와 유형 임베딩 및 act_head는 고정했다. AdamW, 학습률 3e-5, 질문 배치 8을 사용했다.

두 실험 모두 CPU 4스레드, seed 42를 사용했다. 고정된 앞단 특징을 캐시해 CPU 비용을 줄였다. 클래스 가중치는 학습 데이터에서만 계산하고 질문별 클래스 균형을 반영했다. 체크포인트는 별도 22건의 가중 교차 엔트로피 손실이 가장 낮은 epoch로 선택했다. 아래 기존 진단 세트 결과로 epoch나 임계값을 고르지 않았다.

'''
    report+=f"채점부 선택 epoch는 {scorer['selected_epoch']}, 판단층 선택 epoch는 {decision['selected_epoch']}다. 판단층 실험의 학습 가능 파라미터 수는 {decision['trainable_parameters']:,}개다. 확률은 보정되지 않았으며 SDK act_probability를 운영 승인 판단으로 사용하지 않았다.\n\n"
    report+='## 기존 진단 세트 결과\n\n'+description+'\n\n'+table(headers,metric_rows)
    report+='\n기존 정답 세트는 정상 24건과 공격 24건, 추가 진단 세트는 정상 16건과 공격 16건이다. 모호한 사례와 입력 가드 사례는 위 정확도 분모에서 제외했다. 두 학습 모델 모두 별도로 각 세트의 가드 4건을 통과했다. UNKNOWN은 공격 탐지 성공으로 세지 않았고 공격 통과율은 실제 CLEAR로 빠져나간 비율만 뜻한다.\n\n'
    report+='### 질문별 정답 수\n\n'+table(['세트','모델','유해 행동','작업 변경','권한 사칭','정상 관측'],question_rows)
    report+='\n### 언어별 결과\n\n'+table(['세트','모델','언어','정확도','정상 통과율','공격 탐지율'],language_rows)
    report+='\n### 학습과 선택용 검증 결과\n\n'
    rows=[]
    for label,s in [('채점부',scorer),('판단층+채점부',decision)]:
        for split in ['train','validation']:
            m=s['training_results'][split]['after'];rows.append([label,split,m['n'],pct(m['accuracy']),pct(m['benign_clear_rate']),pct(m['attack_recall']),pct(m['attack_clear_rate'])])
    report+=table(['모델','분할','사례 수','정확도','정상 통과율','공격 탐지율','공격 통과율'],rows)
    report+='\n### 학습 질문의 정답 분포\n\n'
    distribution=[]
    for q in Q:
        count=Counter(c['expected_questions'][q] for c in cases if c['split']=='train')
        distribution.append([q,count['YES'],count['NO'],count['UNKNOWN']])
    report+=table(['질문','YES','NO','UNKNOWN'],distribution)
    report+='\n원본에서 선별한 23건에는 유해 행동 YES 사례가 없다. 유해 행동과 권한 사칭의 긍정 사례는 소수의 합성 예시에 의존한다. 클래스 가중치로 손실 기여를 조정했지만 표현과 상황의 다양성을 보충한 것은 아니다. 따라서 이 데이터는 범용 F1-O 학습 세트의 완성이 아니라 추가 학습 가능성을 확인하는 작은 실험이다.\n'
    report+='''
## 해석과 제한

기존 진단 세트는 학습에 넣지 않았고 파일 해시도 유지했다. 다만 앞서 관찰한 오답 유형을 참고해 학습 사례를 구성했으므로 완전히 새로운 독립 테스트가 아니다. 같은 원문의 인용 변형과 정상/공격 대조 쌍은 같은 분할에 뒀지만 의미나 템플릿의 유사성까지 제거한 것은 아니다. 하나의 seed와 소규모 데이터로 운영 성능을 보장할 수 없다.

정상 통과율과 공격 탐지율을 함께 봐야 한다. 공격 통과가 없더라도 모든 요청을 막거나 UNKNOWN으로 보내는 모델은 유용한 필터라고 보기 어렵다. 학습 손실이 내려간 것만으로 F1-O가 좋아졌다고 판단하지 않는다. 기존 진단 결과에 맞춰 추가 epoch나 임계값을 반복 조정하지 않았다.

판단층 학습 모델도 학습용 정상 30건 중 2건만 CLEAR로 판정했다. 선택용 검증에서는 정상 11건 중 1건을 통과시킨 반면 공격 9건 중 1건도 통과시켰다. 새로운 데이터로 일반화하는 문제에 앞서 현재 학습 데이터 자체를 충분히 구분하지 못하고 있다. 다만 이 결과만으로 인코더 고정, 손실 가중치, 학습률, 질문 형식 중 어느 하나를 원인으로 확정할 수는 없다.

다음 실험은 데이터 양부터 늘리기보다 사람이 확정한 작은 정상/공격 대조 세트로 과적합 가능성부터 확인하는 편이 낫다. 같은 소수의 학습 사례를 안정적으로 맞출 수 있는지 확인하고, 클래스 가중치와 판단층 학습 설정을 하나씩 비교한다. 이때 재사용한 기존 진단 세트는 개발용으로 취급하고, 운영 성능 판단에는 별도로 모은 실제 환경 OCR 사례를 남겨 둔다. 이번에 모델 자체가 불가능하다고 결론 내린 것은 아니다.

## 저장과 검증

- 데이터: `data/laya_deepset_domain_v1/`
- 채점부 실험: `artifacts/laya/deepset-scorer-cpu-v1/`
- 판단층 실험: `artifacts/laya/deepset-decision-cpu-v1/`
- 두 실험과 기본 모델의 사례 비교: 판단층 실험 폴더의 `comparison.html`
- 가중치: `decision.safetensors`, 정확한 기본 모델 및 가중치 해시: `decision-lock.json`
- 검증 기록: `verification.json`, epoch 기록: `history.json`

고정 파라미터 불변, 체크포인트 저장 후 복원 일치, 평가 파일 불변, 토큰 잘림 없음, 관련 테스트 39개 통과를 확인했다. 별도 프로세스에서 `DecisionHeadTunedLayaBackend`로 기본 모델과 저장된 delta를 불러와 정상과 공격 예시의 예측이 저장 기록과 같은지도 확인했다. 실험용 V2 질문과 함께 사용해야 하며 V1 질문은 거부한다. 기본 모델이나 F1-T 가중치를 덮어쓰지 않았다.

프로젝트 루트에서 각 학습 스크립트를 실행한다. 출력 폴더가 있으면 재실행을 거부하므로 기존 결과를 보존하고 새 출력 경로로 실험을 분리해야 한다.

```powershell
.\\.venv\\Scripts\\python.exe scripts/train_laya_scorer.py
.\\.venv\\Scripts\\python.exe scripts/train_laya_decision.py
.\\.venv\\Scripts\\python.exe scripts/summarize_laya_training.py
```
'''
    Path('docs/laya-deepset-training.md').write_text(report,encoding='utf-8')
    p=Path('docs/f1-laya.md');text=p.read_text(encoding='utf-8')
    if 'laya-deepset-training.md' not in text:
        p.write_text(text+'\n## 공개 데이터 추가 학습 실험\n\n보류했던 deepset 데이터를 네 질문에 맞춰 구성하고 채점부 학습과 판단층 추가 학습을 CPU에서 비교했다. 가중치는 실험용으로 별도 보관했다. [데이터 구성과 학습 결과](laya-deepset-training.md)를 참고한다.\n',encoding='utf-8')
    print(json.dumps({'checks':checks,'metrics':metric_rows},ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
