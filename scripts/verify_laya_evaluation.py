"""Independent bookkeeping checks for the completed diagnostic evaluation."""
import csv
import hashlib
import json
import runpy
from html.parser import HTMLParser
from pathlib import Path


def main():
    root=Path('artifacts/laya/gold-eval-cpu-v1')
    gold=Path('data/laya_gold_v1')
    rows=[json.loads(s) for s in (root/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    summary=json.loads((root/'summary.json').read_text(encoding='utf-8'))
    lock=json.loads((gold/'gold-lock.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((gold/'cases.jsonl').read_bytes()).hexdigest()==lock['sha256']==summary['gold_sha256']
    assert (gold/'cases.jsonl').read_bytes()==(root/'cases.jsonl').read_bytes()
    assert len(rows)==56 and len({r['case_id'] for r in rows})==56
    counts={suite:sum(r['suite']==suite for r in rows) for suite in ['semantic','ambiguous','guard']}
    assert counts=={'semantic':48,'ambiguous':4,'guard':4}
    mapping={'A':'YES','B':'NO','C':'UNKNOWN'}
    inference_count=0
    for row in rows:
        trace=json.loads((root/(row['case_id']+'-trace.json')).read_text(encoding='utf-8'))
        if not row['laya_executed']:continue
        inference_count+=1
        raw=trace['raw_result']
        assert raw['usage']['truncated'] is False and raw['usage']['state_tokens_dropped']==0
        assert all(not b['truncated'] for b in trace['token_budget'].values())
        for q,answer in raw['answers'].items():
            assert mapping[answer['choice']]==row['answers'][q]['model_answer']
    assert inference_count==52
    semantic=[r for r in rows if r['suite']=='semantic']
    assert sum(r['actual_status']==r['expected_status'] for r in semantic)==summary['suites']['semantic']['correct']==24
    assert sum(r['expected_status']=='CLEAR' and r['actual_status']=='ATTACK' for r in semantic)==24
    assert all(r['status_correct'] and r['guard_execution_correct'] for r in rows if r['suite']=='guard')
    assert all(not r['laya_executed'] for r in rows if r['suite']=='guard')
    namespace=runpy.run_path('scripts/evaluate_laya_gold.py')
    metric=namespace['metrics']
    synthetic=[dict(expected_status='CLEAR',actual_status='CLEAR',status_correct=True),dict(expected_status='ATTACK',actual_status='UNKNOWN',status_correct=False)]
    m=metric(synthetic)
    assert m['accuracy']==.5 and m['attack_recall']==0 and m['benign_clear_rate']==1 and m['abstain_or_error_count']==1
    assert metric([])['accuracy'] is None
    # Refresh the viewer without another model run.
    namespace['render_review'](root,rows)
    class AuditHTML(HTMLParser):
        articles=0
        def handle_starttag(self,tag,attrs):
            if tag=='article':self.articles+=1
    parser=AuditHTML();parser.feed((root/'error-review.html').read_text(encoding='utf-8'))
    assert parser.articles==56
    with (root/'error-review.csv').open(encoding='utf-8-sig',newline='') as f:
        assert len(list(csv.DictReader(f)))==52
    cases=[json.loads(s) for s in (gold/'cases.jsonl').read_text(encoding='utf-8').splitlines()]
    with (gold/'gold-review.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['case_id','suite','language','category','command','constraints','ocr_text','expected_status',*mapping.values(),'rationale']
        fields=['case_id','suite','language','category','command','constraints','ocr_text','expected_status','harmful_instruction','task_override','authority_spoofing','benign_observation','rationale','human_review']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for c in cases:
            texts=' | '.join(c['ocr_texts'])
            # Full long guard text remains available in cases.jsonl.
            if len(texts)>1000:texts=texts[:1000]+' [표에서만 생략. cases.jsonl에 전체 보존]'
            writer.writerow({**{k:c[k] for k in ['case_id','suite','language','category','command','expected_status','rationale']},
                'constraints':' | '.join(c['constraints']),'ocr_text':texts,**(c['expected_questions'] or {}),'human_review':'PENDING'})
    report=dict(passed=True,gold_unchanged=True,case_counts=counts,inference_count=inference_count,
        raw_answer_mapping_verified=True,all_inferences_untruncated=True,guard_checks=4,
        metrics_denominators_verified=True,html_case_cards=parser.articles,error_review_rows=52,
        verification_scope='Data integrity, decoding, budget records, aggregation and HTML structure; not semantic label approval')
    (root/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
