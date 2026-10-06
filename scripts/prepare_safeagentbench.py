"""Build an auditable text-only candidate pool from a pinned, reviewed release.

This script does not train a model or assign train/validation/test splits.
"""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import html
import json
from pathlib import Path
import re
import shutil

REVISION = 'f5d2df6473e519a4f9ece5a17fe8b755d8256819'
MODEL = 'sentence-transformers/clip-ViT-B-32-multilingual-v1'
MODEL_REVISION = '58edf8cada9e398793dca955574a48cbb7f18be2'
CATEGORIES = {
    'normal_household_task':'정상 가정 작업',
    'electrical_liquid_damage':'전자기기와 전기 제어부의 액체 손상',
    'fire_heat_microwave':'화재, 가열 및 전자레인지 오용',
    'impact_breakage_sharp_objects':'충격, 파손 및 날카로운 물체',
    'slip_spill_instability':'미끄러짐, 쏟음 및 불안정한 배치',
    'property_liquid_damage':'생활 물품의 액체 오염과 손상',
    'plant_alcohol_damage':'식물에 술을 부어 발생하는 손상',
    'nonfood_ingestion':'비식품 섭취 유도',
    'electrical_contact':'가동 중 전기 장치에 금속 도구 접촉',
    'context_required':'추가 맥락 필요',
    'source_label_conflict':'원본 라벨에 이견',
    'ambiguous_instruction':'불명확한 명령',
    'unsupported_causal_claim':'근거가 불분명한 인과관계',
    'outside_physical_risk_scope':'현재 물리적 위험 분류 범위 밖',
}

def digest(raw):
    return sha256(raw).hexdigest()

def norm(text):
    # Used only for duplicate/contrast links, never to rewrite model input.
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text).casefold()
    return ' '.join(re.findall(r'\w+', text))

def read_jsonl(path):
    return [json.loads(s) for s in path.read_text(encoding='utf8').splitlines() if s.strip()]

def write_json(path, obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf8')

def build(inputs, output, baseline_root=None, verify_tokenizer=False):
    provenance=json.loads((inputs/'provenance.json').read_text(encoding='utf8'))
    decisions=json.loads((inputs/'review_decisions.json').read_text(encoding='utf8'))
    assert provenance['revision']==decisions['source_revision']==REVISION
    for item in provenance['downloads']:
        raw=(inputs/'source'/item['file']).read_bytes()
        assert digest(raw)==item['sha256'] and len(raw)==item['bytes'], item['file']
    rows=[]
    raw_by_part={}
    for part in ('safe','unsafe'):
        raw_rows=read_jsonl(inputs/'source'/f'{part}_detailed.jsonl')
        assert len(raw_rows)==300
        raw_by_part[part]=raw_rows
        for i,raw in enumerate(raw_rows,1):
            key=f'sab-{part}-{i:03d}'
            d=decisions['decisions'][key]
            assert d['status'] in {'SELECTED','HELD','EXCLUDED'}
            text=raw['instruction']
            assert isinstance(text,str) and text.strip()
            rows.append({
                'schema_version':'f1t-text-candidate-1', 'sample_id':key,
                'source':'safeagentbench/SafeAgentBench','source_revision':REVISION,
                'source_file':f'{part}_detailed.jsonl','source_line':i,
                'source_label':part,'source_risk_category':raw.get('risk_category'),
                'command':text,'text':text,'text_sha256':digest(text.encode()),
                'source_scene_id':raw['scene_name'],
                'text_group_id':'sab-text-'+digest(norm(text).encode())[:16],
                'source_contrast_instruction':raw.get('risk_instruction'),
                'language':'en','modality':'text_only','image_path':None,
                'review_status':d['status'],'category':d['category'],
                'category_ko':CATEGORIES[d['category']],
                'review_reason_ko':d['reason_ko'],'reviewer':'Codex agent',
                'human_reviewed':False,'source_label_changed':False,
                'label':int(part=='unsafe') if d['status']=='SELECTED' else None,
                'split':'UNASSIGNED','usage':'text_auxiliary_candidate_only',
                'label_basis':'SOURCE_INSTRUCTION_SAFETY_PLUS_AGENT_REVIEW',
                'duplicate_of':None,'token_count':None,
            })
    assert set(decisions['decisions'])=={r['sample_id'] for r in rows}

    # Quarantine exact-normalized cross-label conflicts; deduplicate selected inputs.
    grouped=defaultdict(list)
    for r in rows: grouped[norm(r['text'])].append(r)
    dedup=[]
    for group in grouped.values():
        chosen=[r for r in group if r['review_status']=='SELECTED']
        if len({r['source_label'] for r in group})>1:
            for r in chosen:
                r.update(review_status='HELD',label=None,category='source_label_conflict',
                         category_ko=CATEGORIES['source_label_conflict'],
                         review_reason_ko='구두점과 대소문자를 정규화한 동일 명령에 원본 정상 및 위험 라벨이 함께 있어 보류함.')
        elif len(chosen)>1:
            representative=chosen[0]
            for r in chosen[1:]:
                r.update(review_status='DUPLICATE',label=None,duplicate_of=representative['sample_id'])
                r['review_reason_ko']='중복 명령으로 선별 개수에서 제외함. 원본 장면과 정답은 원장에 보존함.'
                dedup.append({'sample_id':r['sample_id'],'representative_id':representative['sample_id']})

    # Source-provided contrasts are links, not 300 additional training examples.
    unsafe_lookup=defaultdict(list)
    for r in rows:
        if r['source_label']=='unsafe': unsafe_lookup[norm(r['command'])].append(r)
    links=[]
    parents={r['sample_id']:r['sample_id'] for r in rows}
    def find(x):
        while parents[x]!=x:
            parents[x]=parents[parents[x]]; x=parents[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b: parents[max(a,b)]=min(a,b)
    for group in grouped.values():
        for r in group[1:]: union(group[0]['sample_id'],r['sample_id'])
    for r in rows:
        if r['source_label']=='safe' and r['source_contrast_instruction']:
            matches=unsafe_lookup.get(norm(r['source_contrast_instruction']),[])
            links.append({'safe_id':r['sample_id'],'unsafe_ids':[m['sample_id'] for m in matches],
                          'match_type':'normalized_source_risk_instruction',
                          'new_training_row_created':False})
            for m in matches: union(r['sample_id'],m['sample_id'])
    for r in rows: r['contrast_group_id']='sab-contrast-'+find(r['sample_id'])

    # Compare against existing project pools without changing them or evaluating models.
    overlap=[]
    checked=[]
    if baseline_root:
        for rel in ('data/robojailbench_f1t_english_v1/selected.jsonl',
                    'data/public_f1_bundle_v1/text_aux/selected.jsonl'):
            path=baseline_root/rel
            if not path.exists(): continue
            checked.append({'file':rel,'sha256':digest(path.read_bytes())})
            other=defaultdict(list)
            for item in read_jsonl(path):
                for field in ('command','text','proposed_action'):
                    if isinstance(item.get(field),str): other[norm(item[field])].append(item.get('sample_id',item.get('id')))
            for r in rows:
                if norm(r['text']) in other:
                    overlap.append({'sample_id':r['sample_id'],'existing_file':rel,
                                    'existing_ids':sorted(set(str(x) for x in other[norm(r['text'])]))})
                    if r['review_status']=='SELECTED':
                        r.update(review_status='HELD',label=None,category='source_label_conflict',
                                 category_ko='기존 데이터 중복 검토',review_reason_ko='기존 프로젝트 데이터와 명령이 중복되어 기존 평가 분할을 확인할 때까지 보류함.')

    if verify_tokenizer:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_REVISION,local_files_only=True)
        tokens=tokenizer([r['text'] for r in rows],add_special_tokens=True,truncation=False)['input_ids']
        for r,t in zip(rows,tokens):
            r['token_count']=len(t)
            if len(t)>128 and r['review_status']=='SELECTED':
                r.update(review_status='HELD',label=None,category='context_required',
                         category_ko='입력 길이 초과',review_reason_ko='인코더의 128토큰 한도를 넘어 절단 없이 입력 형식 검토가 필요함.')

    selected=[r for r in rows if r['review_status']=='SELECTED']
    assert len({norm(r['text']) for r in selected})==len(selected)
    assert all(r['label']==int(r['source_label']=='unsafe') for r in selected)
    assert all(r['label'] is None for r in rows if r['review_status']!='SELECTED')
    assert all(r['text']==r['command'] and r['image_path'] is None and r['split']=='UNASSIGNED' for r in rows)
    deferred=[]
    for part in ('abstract','long_horizon'):
        for i,r in enumerate(read_jsonl(inputs/'source'/f'{part}.jsonl'),1):
            deferred.append({'source_file':f'{part}.jsonl','source_line':i,
                             'review_status':'DEFERRED_NOT_REVIEWED','label':None,
                             'reason_ko':'이번 검토는 구체적 명령 600개에 한정함. 추상 과제와 시간 순서 제약 과제는 이진 라벨 의미를 별도로 검토해야 함.'})
    assert len(deferred)==150
    summary={
        'dataset_version':'safeagentbench-f1t-en-candidates-v1','source_revision':REVISION,
        'source_detailed_rows':600,'deferred_source_tasks':150,
        'status_counts':dict(Counter(r['review_status'] for r in rows)),
        'selected_label_counts':dict(Counter(r['label'] for r in selected)),
        'selected_categories':dict(Counter(r['category_ko'] for r in selected)),
        'held_categories':dict(Counter(r['category_ko'] for r in rows if r['review_status']=='HELD')),
        'source_label_conflicts':sum(r['category']=='source_label_conflict' and r['review_status']=='HELD' for r in rows),
        'source_contrast_links':len(links),'matched_source_contrast_links':sum(bool(l['unsafe_ids']) for l in links),
        'selected_source_contrast_links':sum(any(i in {r['sample_id'] for r in selected} for i in l['unsafe_ids']) and l['safe_id'] in {r['sample_id'] for r in selected} for l in links),
        'exact_normalized_overlap_count':len(overlap),'existing_pools_checked':checked,
        'tokenizer_checked':verify_tokenizer,'tokenizer_model':MODEL,'tokenizer_revision':MODEL_REVISION,
        'max_selected_tokens':max((r['token_count'] or 0) for r in selected) if verify_tokenizer else None,
        'human_reviewed':False,'split_status':'UNASSIGNED',
        'training_performed':False,'training_ready':False,'images_included':False,
        'compatibility_note':'Candidate manifest only. Existing SafeText-specific loader is unchanged and does not accept this source.',
        'dedup_policy':'CamelCase boundaries, case, whitespace, punctuation. Semantic paraphrases still need grouping before split.'
    }
    output.mkdir(parents=True,exist_ok=False)
    shutil.copytree(inputs/'source',output/'source')
    shutil.copy2(inputs/'provenance.json',output/'provenance.json')
    shutil.copy2(inputs/'review_decisions.json',output/'review_decisions.json')
    write_jsonl(output/'review_ledger.jsonl',rows)
    for name,status in [('selected','SELECTED'),('held','HELD'),('excluded','EXCLUDED'),('duplicates','DUPLICATE')]:
        write_jsonl(output/f'{name}.jsonl',[r for r in rows if r['review_status']==status])
    write_jsonl(output/'deferred.jsonl',deferred)
    write_jsonl(output/'contrast_links.jsonl',links)
    write_jsonl(output/'existing_overlap.jsonl',overlap)
    write_json(output/'summary.json',summary)
    write_html(output/'review.html',rows,summary)
    write_readme(output/'README.md',summary)
    return summary

def write_html(path,rows,summary):
    statuses={'SELECTED':'선별','HELD':'보류','EXCLUDED':'제외','DUPLICATE':'중복'}
    body=[]
    for r in rows:
        fields=[r['sample_id'],statuses[r['review_status']], '정상' if r['source_label']=='safe' else '위험',
                r['category_ko'],r['command'],r['review_reason_ko'],r['source_risk_category'] or '정상 원본',
                r['source_scene_id'],str(r['token_count']),r['duplicate_of'] or '']
        cells=''.join('<td>'+html.escape(s)+'</td>' for s in fields)
        body.append('<tr data-status="'+r['review_status']+'" data-label="'+r['source_label']+'">'+cells+'</tr>')
    s=summary['status_counts']; labels=summary['selected_label_counts']
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>SafeAgentBench 선별 검토표</title>
<style>body{font:15px system-ui,sans-serif;margin:28px;color:#182339;background:#f6f8fc}h1{font-size:26px}p{max-width:1100px;line-height:1.65}.stats{display:flex;gap:14px;flex-wrap:wrap}.stat{background:white;padding:15px 22px;border-radius:10px;border:1px solid #dbe1eb}.toolbar{position:sticky;top:0;background:#f6f8fc;padding:14px 0;display:flex;gap:12px;flex-wrap:wrap}input,select{font:inherit;padding:9px;border:1px solid #bac5d5;border-radius:6px}input{width:360px}table{border-collapse:collapse;width:100%;background:white}th,td{text-align:left;vertical-align:top;padding:12px;border-bottom:1px solid #dee5ee;line-height:1.5}th{background:#e9eef7}td:nth-child(5){min-width:300px}td:nth-child(6){min-width:300px}tr[data-status="HELD"]{background:#fff9eb}tr[data-status="EXCLUDED"],tr[data-status="DUPLICATE"]{color:#657085}a{color:#185bbd}</style>
<h1>SafeAgentBench 선별 검토표</h1>
<p>구체적인 명령 600개를 원문별로 검토한 텍스트 보조학습 후보입니다. 정상/위험은 원본 라벨이며, 보류와 제외 항목에는 학습 라벨을 부여하지 않았습니다. 에이전트 검토 결과로 사람의 최종 확인은 남아 있습니다. 이미지, 학습 분할, 학습된 모델은 포함하지 않습니다.</p>
<div class="stats">STATS</div>
<p><a href="README.md">선별 기준</a> / <a href="selected.jsonl">선별 JSONL</a> / <a href="review_ledger.jsonl">전체 원장</a> / <a href="summary.json">집계</a></p>
<div class="toolbar"><input id="q" placeholder="명령, 범주, 이유 검색"><select id="status"><option value="">전체 상태</option><option value="SELECTED">선별</option><option value="HELD">보류</option><option value="EXCLUDED">제외</option><option value="DUPLICATE">중복</option></select><select id="label"><option value="">원본 라벨 전체</option><option value="safe">정상 원본</option><option value="unsafe">위험 원본</option></select><span id="count"></span></div>
<table><thead><tr><th>ID</th><th>검토</th><th>원본 라벨</th><th>우리 분류</th><th>원문 명령</th><th>판단 근거</th><th>원본 위험 범주</th><th>장면 ID</th><th>토큰</th><th>중복 대표</th></tr></thead><tbody>ROWS</tbody></table>
<script>const rows=[...document.querySelectorAll('tbody tr')];const q=document.getElementById('q'),s=document.getElementById('status'),l=document.getElementById('label');function filter(){let n=0;for(const r of rows){const show=(!s.value||r.dataset.status===s.value)&&(!l.value||r.dataset.label===l.value)&&r.textContent.toLowerCase().includes(q.value.toLowerCase());r.hidden=!show;if(show)n++;}document.getElementById('count').textContent=n+'개 표시';}q.addEventListener('input',filter);s.addEventListener('change',filter);l.addEventListener('change',filter);filter();</script></html>'''
    stats=''.join('<div class="stat">'+html.escape(k)+': <b>'+str(v)+'</b></div>' for k,v in [('선별',s.get('SELECTED',0)),('선별 정상',labels.get(0,0)),('선별 위험',labels.get(1,0)),('보류',s.get('HELD',0)),('제외',s.get('EXCLUDED',0)),('중복',s.get('DUPLICATE',0))])
    path.write_text(page.replace('STATS',stats).replace('ROWS',''.join(body)),encoding='utf8')

def write_readme(path,s):
    status=s['status_counts']; labels=s['selected_label_counts']
    cats='\n'.join(f'| {k} | {v} |' for k,v in s['selected_categories'].items())
    path.write_text(f'''# SafeAgentBench F1-T 텍스트 보조학습 후보

## 선별 결과

구체적인 정상 명령 300개와 위험 명령 300개를 검토했다. **선별 {status.get('SELECTED',0)}개: 정상 {labels.get(0,0)}개, 위험 {labels.get(1,0)}개**다. 보류 {status.get('HELD',0)}개, 제외 {status.get('EXCLUDED',0)}개, 중복 {status.get('DUPLICATE',0)}개를 따로 보존했다. 추상 과제 100개와 장기 과제 50개는 원본만 보관했으며 미검토 상태다.

| 선별 범주 | 개수 |
|---|---:|
{cats}

## 라벨과 선별 기준

- `label=0`: 일반적인 가정 환경과 정상적인 도구 사용 조건에서 구체적인 위해가 드러나지 않는 명령.
- `label=1`: 명령에 드러난 행동으로 인명 또는 물품의 물리적 손상 위험을 설명할 수 있는 명령. 악의적 의도 여부나 프롬프트 인젝션 여부의 라벨은 아니다.
- 원본 라벨을 반대로 고치지 않았다. 정상 원본에 위험이 의심되거나 위험 원본의 위해 근거가 부족하면 보류했다. 모든 미선별 항목의 `label`은 null이다.
- 물체의 존재만으로 위험으로 취급하지 않는다. 칼로 음식을 써는 일과 일반적인 조리도 정상 후보에 포함한다. 다만 재질, 작동 상태, 시간 등 누락된 조건이 정답을 바꾸는 사례는 보류한다.
- 원본의 위험 범주는 그대로 보존하되 학습용 분류와 구분했다. 노트북에 물을 붓는 행동이 원본에서 폭발로 분류되어도 우리는 전자기기 액체 손상으로 분류한다.
- Ethical Social과 Permission Related 제외 방침을 유지한다. 구체적인 물리적 위해 없이 외관이나 청결만 위반하는 사례, 소유 및 폐기 의도에 따라 판단이 바뀌는 사례는 제외했다. 이 자료의 원본에는 해당 두 이름의 범주가 없으므로 우리 검토 사유로 기록했다.
- 세부 범주는 멀티라벨 전체 위험 목록이 아닌 검토용 대표 범주다.
- 에이전트가 600개 명령을 읽고 정한 초기 후보이며 `human_reviewed=false`다. 원본 정상 라벨도 실제 로봇 환경 전체의 안전성을 보증하지 않는다.

## 입력과 보존 정보

모델 입력 후보인 `text`는 원문 `instruction`과 동일하다. 오탈자, 따옴표, 강조 기호도 바꾸지 않았다. 원본 전체 레코드는 `source/`에 보존했다.

`source_risk_category`, `review_reason_ko`, `source_contrast_instruction`, 원본 `step` 및 `final_state`는 모델 입력으로 넣지 않는다. 특히 정상 원본의 `risk_instruction`은 비교 대상 명령이지 정상 입력에 합칠 내용이 아니다. 그 필드로 별도 위험 샘플을 추가 생성하지 않았다.

이미지는 없으며 `image_path=null`이다. `source_scene_id`는 AI2-THOR 장면 식별자일 뿐 환경 설명이나 실제 이미지가 아니다. 흰 이미지나 무관한 이미지를 붙이지 않았다.

## 중복과 비교 쌍

중복 비교에만 CamelCase 경계, 대소문자, 구두점과 공백을 정규화했다. 선별 파일에는 정규화된 명령당 대표 레코드 하나만 남겼다. 중복 원본의 장면은 원장에 남겨 이후 분할에서도 함께 묶을 수 있다.

정상 레코드의 원본 `risk_instruction`이 위험 원본과 일치할 때만 비교 관계를 연결했다. 행 번호가 같다는 이유로 정상과 위험을 짝짓지 않았다. 확인된 원본 비교 연결은 {s['matched_source_contrast_links']}개이고, 양쪽 모두 선별된 연결은 {s['selected_source_contrast_links']}개다. 이 개수는 반드시 고유한 독립 쌍의 수와 같지는 않다.

`contrast_group_id`는 확인된 비교 관계와 동일 명령을 묶는다. 모든 의미적 유사 문장을 식별한 것은 아니다. 향후 분할 시 원본 장면, 동일 및 유사 명령, 비교 관계를 함께 확인해야 한다. 위험 범주 전체를 검증으로 떼는 평가와 학습에 포함된 위험 범주의 새로운 상황 평가는 별도로 설계한다.

기존 RoboJailBench와 SafeText의 프로젝트 선별 풀을 정규화한 문자열로 비교한 중복 기록은 {s['exact_normalized_overlap_count']}개다. 의미적 중복이나 원본 이미지 중복까지 없다는 뜻은 아니다. 기존 데이터와 모델은 변경하지 않았다.

## 현재 완료 범위

원본 고정, 600개 검토, 대표 위험 범주 부여, 보류 및 제외 사유 기록, 중복 제거, 출처와 비교 관계 보존까지 완료했다. 다국어 CLIP 토크나이저 검사 여부: {s['tokenizer_checked']}. 선별 입력 최대 토큰: {s['max_selected_tokens']} / 128.

**아직 학습 분할과 재학습은 하지 않았다.** `split=UNASSIGNED`, `training_ready=false`다. 현재 SafeText 전용 로더는 이 자료를 받지 않으며 그대로 유지했다. 학습 전에 이 소스를 지원하는 어댑터와 분할을 준비해야 한다. 이미지+텍스트 F1-T의 학습셋에 바로 합칠 수 있는 자료는 아니다.

SafeAgentBench는 원래 평가 벤치마크다. 이 후보를 학습에 쓰면 해당 사례를 다시 독립적인 SafeAgentBench 평가로 사용하지 않는다. 프로젝트 내 학습과 검증 분할 및 별도 외부 평가를 먼저 정해야 한다. 현재 247 대 182 같은 클래스 비율을 억지로 맞추기 위해 정상 사례를 버리지 않으며, 가중치나 샘플링 방식은 학습 단계에서 결정한다.

## 파일 구성

- `review.html`: 검색과 상태 필터가 있는 600개 검토표.
- `selected.jsonl`: 중복 제거 후 정상 및 위험 후보.
- `held.jsonl`, `excluded.jsonl`, `duplicates.jsonl`: 미선별 레코드와 사유.
- `review_ledger.jsonl`: 구체적인 명령 600개의 전체 원장.
- `review_decisions.json`: 재현 가능한 에이전트 판단 기록.
- `contrast_links.jsonl`: 원본이 제공한 비교 명령의 연결.
- `deferred.jsonl`: 별도 검토가 필요한 150개 과제의 위치와 상태.
- `existing_overlap.jsonl`: 기존 프로젝트 데이터와 문자열 중복 기록.
- `source/`, `provenance.json`: 원본, 고정 revision, 다운로드 URL 및 SHA-256.

## 출처

[SafeAgentBench 데이터](https://huggingface.co/datasets/safeagentbench/SafeAgentBench), [원본 코드](https://github.com/shengyin1224/SafeAgentBench), [논문](https://arxiv.org/abs/2412.13178).

데이터 카드 라이선스: CC BY 4.0. 원본 저자: Sheng Yin, Xianghe Pang 외, SafeAgentBench. 가공 내용: 후보 선별, 검토 상태와 대표 범주 추가, 중복 제거. 원문 명령과 원본 라벨은 보존했다. 고정 revision: `{REVISION}`.
''',encoding='utf8')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--baseline-root',type=Path)
    p.add_argument('--verify-tokenizer',action='store_true')
    a=p.parse_args()
    print(json.dumps(build(a.input,a.output,a.baseline_root,a.verify_tokenizer),ensure_ascii=False,indent=2))
