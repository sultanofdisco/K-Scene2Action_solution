"""Audit acquired public sources without selecting training rows or running attacks."""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def text_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_jsonl(path, records):
    with path.open('w', encoding='utf-8', newline='\n') as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def qa_key(row):
    return row['question'], row['answer'], row['image_id']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--tokenizer-audit', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    out = root / 'manifests'
    out.mkdir(exist_ok=True)
    jbdir = root / 'raw/jailbreakv/JailBreakV_28K'
    dadir = root / 'raw/daquar_hf/extracted/dataset'
    ghdir = root / 'raw/daquar_processed_github/dataset'
    images = {}
    for folder in (jbdir, dadir / 'images'):
        for p in sorted(folder.rglob('*')):
            if any(part.startswith('.') or part == '__MACOSX' for part in p.relative_to(folder).parts):
                continue
            if p.suffix.lower() not in ('.png', '.jpg', '.jpeg'):
                continue
            result = {'image_path': p.relative_to(root).as_posix(), 'sha256': digest(p)}
            try:
                with Image.open(p) as im:
                    im.load()
                    result.update(valid=True, width=im.width, height=im.height)
            except Exception as exc:
                result.update(valid=False, error=str(exc))
            images[p.resolve()] = result
    write_jsonl(out / 'image_inventory.jsonl', images.values())

    jb = rows(jbdir / 'JailBreakV_28K.csv')
    mini = rows(jbdir / 'mini_JailBreakV_28K.csv')
    rt = rows(jbdir / 'RedTeam_2K.csv')
    jb_records = []
    for index, row in enumerate(jb):
        p = (jbdir / row['image_path']).resolve()
        if not p.is_relative_to(jbdir):
            raise ValueError('Unsafe upstream image path')
        im = images.get(p)
        jb_records.append({
            'inventory_id': f'jb28k:{index}', 'source_row_index': index,
            'source': row, 'source_safety_role': 'unsafe_in_jdapt',
            'training_selected': False, 'review_status': 'NOT_PROJECT_REVIEWED',
            'redteam_text_sha256': text_hash(row['redteam_query']),
            'jailbreak_text_sha256': text_hash(row['jailbreak_query']),
            'image_available': bool(im and im['valid']), 'image': im,
        })
    paired = [r for r in jb_records if r['image_available']]
    write_jsonl(out / 'jailbreakv_inventory.jsonl', jb_records)
    write_jsonl(out / 'jailbreakv_with_images.jsonl', paired)
    with (out / 'jailbreakv_missing_images.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['source_row_index', 'id', 'image_path', 'format', 'policy'])
        writer.writeheader()
        for record in jb_records:
            if not record['image_available']:
                row = record['source']
                writer.writerow({'source_row_index': record['source_row_index'], **{k: row[k] for k in ['id', 'image_path', 'format', 'policy']}})
    write_jsonl(out / 'redteam2k_inventory.jsonl', [
        {'inventory_id': f'redteam2k:{i}', 'source': r, 'source_safety_role': 'unsafe_in_jdapt',
         'training_selected': False, 'review_status': 'NOT_PROJECT_REVIEWED',
         'question_sha256': text_hash(r['question']), 'image_available': False}
        for i, r in enumerate(rt)
    ])

    da = rows(dadir / 'data.csv')
    processed = {s: rows(dadir / f'data_{s}.csv') for s in ('train', 'eval')}
    processed_keys = {s: Counter(qa_key(r) for r in rr) for s, rr in processed.items()}
    official_keys = defaultdict(Counter)
    official_counts, official_image_ids = {}, {}
    normalization_errors = []
    pattern = re.compile(r'( (in |on |of )?(the |this )?(image\d*) \?)')
    for split in ('train', 'test'):
        lines = (root / f'raw/daquar_official/qa.894.raw.{split}.txt').read_text(encoding='utf-8').splitlines()
        assert len(lines) % 2 == 0 and all(lines)
        official_counts[split] = len(lines) // 2
        image_ids = set()
        for i in range(0, len(lines), 2):
            match = pattern.findall(lines[i])
            if not match:
                normalization_errors.append({'split': split, 'line_1based': i + 1, 'question': lines[i]})
                question = lines[i]
                image_match = re.search(r'image\d+', lines[i])
                image_id = image_match.group(0) if image_match else 'UNPARSED'
            else:
                question = lines[i].replace(match[0][0], '')
                image_id = match[0][3]
            image_ids.add(image_id)
            official_keys[(question, lines[i + 1], image_id)][split] += 1
        official_image_ids[split] = image_ids
    da_records = []
    for i, row in enumerate(da):
        im = images.get((dadir / 'images' / (row['image_id'] + '.png')).resolve())
        da_records.append({
            'inventory_id': f'daquar:{i}', 'source_row_index': i,
            'question': row['question'], 'answer_metadata_only': row['answer'],
            'source_image_id': row['image_id'], 'source_safety_role': 'safe_in_jdapt',
            'training_selected': False, 'review_status': 'NOT_PROJECT_REVIEWED',
            'official_split_matches': dict(official_keys[qa_key(row)]),
            'processed_split_matches': {s: c[qa_key(row)] for s, c in processed_keys.items() if c[qa_key(row)]},
            'image_available': bool(im and im['valid']), 'image': im,
        })
    write_jsonl(out / 'daquar_inventory.jsonl', da_records)
    all_official = Counter({key: sum(c.values()) for key, c in official_keys.items()})
    all_processed = Counter(qa_key(r) for r in da)
    train_images = {r['image_id'] for r in processed['train']}
    eval_images = {r['image_id'] for r in processed['eval']}
    paired_images = {r['image']['image_path'] for r in paired}
    jb_images = [im for p, im in images.items() if p.is_relative_to(jbdir)]
    jb_content_keys = {(r['jailbreak_query'], r['redteam_query'], r['image_path']) for r in jb}
    summary = {
        'audited_utc': datetime.now(timezone.utc).isoformat(),
        'purpose': 'Source inventory only. No rows selected for training; no attacks executed.',
        'image_validation': {'files': len(images), 'valid': sum(im['valid'] for im in images.values()),
                             'invalid': [im for im in images.values() if not im['valid']]},
        'jailbreakv': {
            'full_rows': len(jb), 'image_files': len(jb_images),
            'image_unique_hashes': len({im['sha256'] for im in jb_images}),
            'referenced_unique_paths': len({r['image_path'] for r in jb}),
            'rows_with_valid_original_image': len(paired),
            'rows_missing_original_image': len(jb) - len(paired),
            'referenced_available_image_files': len(paired_images),
            'unreferenced_downloaded_image_files': len(jb_images) - len(paired_images),
            'format_counts': dict(Counter(r['format'] for r in jb)),
            'paired_format_counts': dict(Counter(r['source']['format'] for r in paired)),
            'policy_counts': dict(Counter(r['policy'] for r in jb)),
            'paired_policy_counts': dict(Counter(r['source']['policy'] for r in paired)),
            'unique_redteam_texts': len({r['redteam_query'] for r in jb}),
            'unique_jailbreak_texts': len({r['jailbreak_query'] for r in jb}),
            'paired_unique_redteam_texts': len({r['source']['redteam_query'] for r in paired}),
            'paired_unique_jailbreak_texts': len({r['source']['jailbreak_query'] for r in paired}),
            'exact_text_image_duplicate_excess': sum(n - 1 for n in Counter((r['jailbreak_query'], r['redteam_query'], r['image_path']) for r in jb).values()),
            'redteam2k_rows': len(rt), 'redteam2k_unique_questions': len({r['question'] for r in rt}),
            'redteam2k_questions_not_in_full': len({r['question'] for r in rt} - {r['redteam_query'] for r in jb}),
            'mini_rows': len(mini),
            'mini_rows_with_valid_original_image': sum(bool(images.get((jbdir / r['image_path']).resolve(), {}).get('valid')) for r in mini),
            'mini_rows_matching_full_by_content': sum((r['jailbreak_query'], r['redteam_query'], r['image_path']) in jb_content_keys for r in mini),
        },
        'daquar': {
            'processed_rows': len(da), 'image_files': len(images) - len(jb_images),
            'rows_with_valid_original_image': sum(r['image_available'] for r in da_records),
            'unique_referenced_images': len({r['image_id'] for r in da}),
            'processed_split_rows': {s: len(rr) for s, rr in processed.items()},
            'processed_split_image_overlap': len(train_images & eval_images),
            'processed_train_unique_images': len(train_images), 'processed_eval_unique_images': len(eval_images),
            'official_downloaded_split_rows': official_counts,
            'official_split_unique_images': {s: len(ids) for s, ids in official_image_ids.items()},
            'official_split_image_overlap': len(official_image_ids['train'] & official_image_ids['test']),
            'official_vs_processed_same_record_multiset': all_official == all_processed,
            'official_questions_unmatched_by_author_normalizer': normalization_errors,
            'official_unmatched_processed_rows': sum(not r['official_split_matches'] for r in da_records),
            'official_ambiguous_split_rows': sum(len(r['official_split_matches']) > 1 for r in da_records),
            'exact_qa_image_duplicate_excess': sum(n - 1 for n in all_processed.values()),
            'hf_vs_github_same_sha256': {name: digest(dadir / name) == digest(ghdir / name)
                                       for name in ['data.csv', 'data_train.csv', 'data_eval.csv', 'all_qa_pairs.txt']},
        },
    }
    if args.tokenizer_audit:
        try:
            from transformers import AutoTokenizer
            model = 'sentence-transformers/clip-ViT-B-32-multilingual-v1'
            revision = '58edf8cada9e398793dca955574a48cbb7f18be2'
            tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, local_files_only=True)
            token_summary = {'model': model, 'revision': revision, 'project_max_tokens': 128,
                             'includes_special_tokens': True, 'truncation': False, 'groups': {}}
            for name, texts in {
                'jb28k_jailbreak_query': [r['jailbreak_query'] for r in jb],
                'jb28k_paired_jailbreak_query': [r['source']['jailbreak_query'] for r in paired],
                'redteam2k_question': [r['question'] for r in rt],
                'daquar_question': [r['question'] for r in da],
            }.items():
                lengths = []
                for start in range(0, len(texts), 256):
                    enc = tokenizer(texts[start:start + 256], truncation=False, padding=False, verbose=False)
                    lengths.extend(map(len, enc['input_ids']))
                lengths.sort()
                token_summary['groups'][name] = {
                    'rows': len(lengths), 'max': max(lengths), 'median': lengths[len(lengths) // 2],
                    'over_project_limit': sum(n > 128 for n in lengths),
                }
            summary['token_lengths'] = token_summary
        except Exception as exc:
            summary['token_lengths'] = {'status': 'not_measured', 'error': str(exc)}
    (root / 'audit.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
