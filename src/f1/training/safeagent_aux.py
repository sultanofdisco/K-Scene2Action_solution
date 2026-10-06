"""Group-aware preparation and validation of reviewed SafeAgentBench text data."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re

from .data import digest, write_json

SOURCE = 'safeagentbench/SafeAgentBench'
REVISION = 'f5d2df6473e519a4f9ece5a17fe8b755d8256819'
SCHEMA = 'f1t-text-aux-sab-1'
POLICY = 'sab-scene-contrast-neartext-v1'
MANUAL_LINKS = [
    ('safe-276','safe-239'), ('safe-281','safe-262'),
    ('safe-277','safe-251'), ('safe-126','safe-027'),
    ('unsafe-217','unsafe-211'), ('unsafe-181','unsafe-030'),
    ('safe-080','safe-062'), ('safe-285','safe-279'),
    ('unsafe-212','unsafe-128'), ('unsafe-207','unsafe-048'),
    ('safe-209','safe-061'), ('safe-194','safe-002'),
    ('safe-262','safe-036'),
    ('unsafe-120','unsafe-077'), ('unsafe-109','unsafe-085'),
    ('safe-263','safe-236'), ('safe-247','safe-220'),
    ('safe-285','safe-247'), ('safe-284','safe-130'),
    ('safe-092','safe-010'), ('safe-199','safe-136'),
    ('safe-268','safe-005'),
    ('safe-293','safe-201'), ('safe-232','safe-201'),
]
MANUAL_GROUPS = [
    ['safe-'+str(i).zfill(3) for i in [54,95,98,101,104,107,110,113,121,127,133,138,143,156,162,188,205,208,211,213,219,231,235,238,251,260,277,292,296]],
    ['safe-'+str(i).zfill(3) for i in [11,23,49,81,130,284,291]],
    ['safe-'+str(i).zfill(3) for i in [220,236,247,263,279,285]],
    ['unsafe-'+str(i).zfill(3) for i in [38,91,186]],
]


def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf8').splitlines() if s.strip()]


def near_text(text):
    """Conservative grouping only: this never becomes encoder input."""
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text).casefold()
    text = re.sub(r'\b(put|placing|placed)\b', 'place', text)
    text = re.sub(r'\b(onto|into|inside)\b', 'in', text)
    return ' '.join(w for w in re.findall(r'\w+', text)
                    if w not in {'the', 'a', 'an', 'then', 'and', 'gently', 'safely'})


def validate_sab_rows(rows):
    from hashlib import sha256
    ids = set(); links = defaultdict(set)
    for r in rows:
        if r.get('schema_version') != SCHEMA or r.get('source') != SOURCE or r.get('source_revision') != REVISION:
            raise ValueError('Unsupported SafeAgentBench source/schema/revision')
        if r.get('modality') != 'text_only' or r.get('image_path') is not None or r.get('language') != 'en':
            raise ValueError('Invalid SafeAgentBench modality/language')
        if r.get('review_status') != 'SELECTED' or r.get('usage') != 'auxiliary_text_experiment_only':
            raise ValueError('Unselected SafeAgentBench record')
        if r.get('split_policy') != POLICY or r.get('split') not in {'train', 'validation', 'test'}:
            raise ValueError('Unassigned or unknown SafeAgentBench split')
        if type(r.get('label')) is not int or r['label'] not in (0, 1) or r.get('source_label') != ('unsafe' if r['label'] else 'safe'):
            raise ValueError('SafeAgentBench label mismatch')
        text = r.get('text')
        if not isinstance(text, str) or not text.strip() or text != r.get('command') or sha256(text.encode()).hexdigest() != r.get('text_sha256'):
            raise ValueError('SafeAgentBench command modified or metadata included in input')
        if r.get('sample_id') in ids or not r.get('sample_id'):
            raise ValueError('Duplicate or missing SafeAgentBench ID')
        ids.add(r['sample_id'])
        for field in ('group_id', 'source_scene_id', 'contrast_group_id', 'text_group_id'):
            if not isinstance(r.get(field),str) or not r[field]:
                raise ValueError('Missing SafeAgentBench grouping key')
            links[(field,r[field])].add(r['split'])
        links[('normalized_command',near_text(text))].add(r['split'])
    if any(len(splits)>1 for splits in links.values()):
        raise ValueError('SafeAgentBench group/scene/contrast/text leakage')
    for split in ('train','validation','test'):
        if {r['label'] for r in rows if r['split']==split}!={0,1}:
            raise ValueError('Both classes required in each SafeAgentBench split')


def prepare(input_dir, output_dir, seed=42):
    """No model outputs are used to form or choose the split."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from importlib.metadata import version
    input_dir=Path(input_dir); output_dir=Path(output_dir)
    ledger=read_rows(input_dir/'review_ledger.jsonl')
    selected=read_rows(input_dir/'selected.jsonl')
    original_summary=json.loads((input_dir/'summary.json').read_text(encoding='utf8'))
    if selected != [r for r in ledger if r['review_status']=='SELECTED']:
        raise ValueError('Selected file differs from reviewed ledger')
    if len(ledger)!=600 or len(selected)!=429:
        raise ValueError('Unexpected reviewed version; create a new split policy for a different pool')
    source_rows={part:read_rows(input_dir/'source'/f'{part}_detailed.jsonl') for part in ('safe','unsafe')}
    provenance=json.loads((input_dir/'provenance.json').read_text(encoding='utf8'))
    for entry in provenance['downloads']:
        if digest(input_dir/'source'/entry['file'])!=entry['sha256']:
            raise ValueError('Source file checksum mismatch')
    for r in ledger:
        raw=source_rows[r['source_label']][r['source_line']-1]
        if r['command']!=raw['instruction'] or r['text']!=r['command'] or r['source_scene_id']!=raw['scene_name']:
            raise ValueError('Reviewed row no longer matches source')

    # Withheld/excluded commands cannot leak into model fitting because they are absent.
    # Retain duplicate originals so the representative inherits all known source scenes.
    active=[r for r in ledger if r['review_status'] in {'SELECTED','DUPLICATE'}]
    parent=list(range(len(active)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]; i=parent[i]
        return i
    def union(i,j):
        a,b=find(i),find(j)
        if a!=b: parent[max(a,b)]=min(a,b)
    seen={}
    for i,r in enumerate(active):
        for field in ('source_scene_id','contrast_group_id','text_group_id'):
            key=(field,r[field])
            if key in seen: union(i,seen[key])
            else: seen[key]=i
    normalized=[near_text(r['text']) for r in active]
    features=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5)).fit_transform(normalized)
    similarities=cosine_similarity(features)
    edges=[]
    for i in range(len(active)):
        for j in range(i):
            if similarities[i,j]>=.90:
                union(i,j)
                edges.append({'a':active[i]['sample_id'],'b':active[j]['sample_id'],
                              'similarity':float(similarities[i,j]),'method':'char_tfidf_ge_0.90'})
    by_id={r['sample_id']:i for i,r in enumerate(active)}
    reviewed_links=MANUAL_LINKS+[(group[0],other) for group in MANUAL_GROUPS for other in group[1:]]
    for a,b in reviewed_links:
        a,b='sab-'+a,'sab-'+b
        union(by_id[a],by_id[b])
        edges.append({'a':a,'b':b,'method':'agent_reviewed_shared_action_or_paraphrase'})
    groups=defaultdict(list); ledger_groups={}
    for i,r in enumerate(active):
        key='sab-group-'+active[find(i)]['sample_id']
        ledger_groups[r['sample_id']]=key
        if r['review_status']=='SELECTED': groups[key].append(r)
    category_groups=defaultdict(set)
    for g,rs in groups.items():
        for r in rs: category_groups[r['category']].add(g)
    # Cover all risk categories in training, without forcing every scarce group there.
    largest=max(groups,key=lambda g:len(groups[g])); forced_train={largest}
    covered={r['category'] for r in groups[largest]}
    while set(category_groups)-covered:
        missing=set(category_groups)-covered
        g=min((g for g in groups if g not in forced_train and any(r['category'] in missing for r in groups[g])),
              key=lambda g:(len(groups[g]),g))
        forced_train.add(g); covered.update(r['category'] for r in groups[g])
    available=sorted(set(groups)-forced_train)
    rng=random.Random(seed); target=round(len(selected)*.15)
    eligible={c for c,gs in category_groups.items() if len(gs)>=3}
    all_groups=set(groups)
    def summarize(gs):
        rs=[r for g in gs for r in groups[g]]
        return len(rs),Counter(r['label'] for r in rs),{r['category'] for r in rs}
    candidates=[]
    # Search by counts and coverage only, never by scores/embeddings of a trained guard.
    for attempt in range(12000):
        order=available.copy(); rng.shuffle(order)
        val=[]; test=[]; counts={'validation':0,'test':0}
        for g in order:
            if counts['validation']<target or counts['test']<target:
                name=min(counts,key=counts.get)
                (val if name=='validation' else test).append(g); counts[name]+=len(groups[g])
        train=all_groups-set(val)-set(test)
        sets={'train':train,'validation':set(val),'test':set(test)}
        stats={k:summarize(v) for k,v in sets.items()}
        if any(set(v[1])!={0,1} for v in stats.values()): continue
        missing=sum(len(eligible-v[2]) for v in stats.values())
        size_penalty=sum(abs(stats[k][0]-target) for k in ('validation','test'))
        # Keep normal/unsafe both represented, without claiming balanced accuracy from raw accuracy.
        balance_penalty=sum(abs(stats[k][1][1]-stats[k][0]*(182/429)) for k in ('validation','test'))
        score=missing*100+size_penalty+balance_penalty*.1
        candidates.append((score,attempt,sets,stats))
    if not candidates: raise ValueError('No valid leakage-controlled partition; collect more independent groups')
    _,chosen_attempt,sets,stats=min(candidates,key=lambda v:(v[0],v[1]))
    assignment={g:split for split,gs in sets.items() for g in gs}
    prepared=[]
    for r in selected:
        row=dict(r,schema_version=SCHEMA,usage='auxiliary_text_experiment_only',
                 group_id=ledger_groups[r['sample_id']],split_policy=POLICY)
        row['split']=assignment[row['group_id']]
        prepared.append(row)
    validate_sab_rows(prepared)
    # Maximum detected near-text similarity across splits for a follow-up manual audit.
    cross=[]
    selected_ids={r['sample_id'] for r in selected}
    for i,r in enumerate(active):
        if r['sample_id'] not in selected_ids: continue
        for j in range(i):
            other=active[j]
            if other['sample_id'] in selected_ids and assignment[ledger_groups[r['sample_id']]]!=assignment[ledger_groups[other['sample_id']]]:
                cross.append({'a':r['sample_id'],'b':other['sample_id'],'similarity':float(similarities[i,j]),
                              'command_a':r['text'],'command_b':other['text']})
    cross=sorted(cross,key=lambda r:-r['similarity'])
    output_dir.mkdir(parents=True,exist_ok=False)
    hashes={}
    for split in ('train','validation','test'):
        path=output_dir/f'{split}.jsonl'
        path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in prepared if r['split']==split),encoding='utf8')
        hashes[split]=digest(path)
    split_stats={split:{'n':stats[split][0],'labels':dict(stats[split][1]),
                       'groups':len(sets[split]),'categories':dict(Counter(r['category'] for r in prepared if r['split']==split))}
                 for split in sets}
    summary={'schema':SCHEMA,'split_policy':POLICY,'seed':seed,'selected_attempt':chosen_attempt,
             'input_hashes':{name:digest(input_dir/name) for name in ('selected.jsonl','review_ledger.jsonl','review_decisions.json','provenance.json')},
             'manifest_hashes':hashes,'split_stats':split_stats,
             'selected_n':len(prepared),'connected_components':len(groups),'largest_group_n':len(groups[largest]),
             'near_text_threshold':.90,'near_text_edges':len(edges),'sklearn_version':version('scikit-learn'),
             'source_scene_disjoint':True,'contrast_groups_disjoint':True,'normalized_commands_disjoint':True,
             'rare_categories_train_only':sorted(c for c in category_groups if all(assignment[g]=='train' for g in category_groups[c])),
             'categories_with_fewer_than_three_groups':sorted(c for c,gs in category_groups.items() if len(gs)<3),
             'cross_split_near_text_max':cross[0]['similarity'] if cross else None,
             'track':'known_risk_categories_unseen_scene_and_command_groups',
             'ood_hazard_family_track_created':False,'model_scores_used_for_split':False,
             'test_evaluated':False,'training_performed':False,
             'tokenizer_model':original_summary['tokenizer_model'],
             'tokenizer_revision':original_summary['tokenizer_revision']}
    write_json(output_dir/'summary.json',summary)
    write_json(output_dir/'group_assignment.json',{'assignment':assignment,'ledger_groups':ledger_groups})
    write_json(output_dir/'near_text_edges.json',edges)
    write_json(output_dir/'cross_split_near_text_review.json',cross[:60])
    return summary
