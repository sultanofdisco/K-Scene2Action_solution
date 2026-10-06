"""Offline CPU experiment: verified paired data, grouped diagnostics, source weighting."""
import argparse
from collections import Counter, defaultdict
import difflib
import json
from pathlib import Path
import random
import re
import shutil
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--local-diagnostic-only', action='store_true')
    args = parser.parse_args()
    repo, out = args.repo.resolve(), args.output.resolve()
    sys.path.insert(0, str(repo / 'src'))
    import numpy as np
    import torch
    from safetensors.torch import load_file, save_file
    from f1.training.data import digest, load_manifest, read_config, write_json
    from f1.training.encoder import load_cache
    from f1.training.learner import create_mlp, train_model, scores, metrics

    started = time.monotonic()
    if args.local_diagnostic_only:
        local_diagnostic(repo, out)
        return
    assert not out.exists(), 'Do not overwrite an experiment'
    protected = {str(p): digest(p) for p in (repo/'artifacts/f1t').rglob('model.safetensors')}
    local_root = repo/'artifacts/f1t/combined-jaewoo-v2-sungmin-v8-cpu-run1'
    specs = [
        ('robojailbench', repo/'data/robojailbench_f1t_english_v1/selected.jsonl', repo/'artifacts/f1t/cache-cpu-v1.2'),
        ('local', local_root/'dataset/selected.jsonl', local_root/'cache'),
    ]
    config = read_config(local_root/'config.json')
    lock = {k: {'model': config[k+'_model'], 'revision': config[k+'_revision']} for k in ('text', 'image')}
    torch.set_num_threads(config['cpu_threads'])
    rows, features, origins, provenance = [], [], [], []
    metas = []
    for domain, manifest, cache in specs:
        source_rows = load_manifest(manifest)
        x, meta = load_cache(cache)
        assert digest(manifest) == meta['manifest_sha256']
        assert meta['encoder_lock'] == lock, 'Encoder revisions differ'
        for key in ('text_model', 'image_model', 'preprocessing', 'embedding_dimension', 'max_tokens'):
            assert meta['config'][key] == config[key], key
        assert meta['feature_order'] == ['text_l2_512', 'image_l2_512']
        lookup = {r['sample_id']: (i, r) for i, r in enumerate(meta['rows'])}
        assert len(lookup) == len(source_rows)
        for raw in source_rows:
            i, old = lookup[raw['sample_id']]
            for key in ('command', 'command_sha256', 'image_sha256', 'label', 'pair_id', 'group_id'):
                assert raw[key] == old[key], (raw['sample_id'], key)
            r = dict(raw, training_domain=domain)
            r['training_source'] = 'robojailbench' if domain == 'robojailbench' else raw['training_source']
            rows.append(r)
            origins.append(manifest.parent/raw['image_path'])
            features.append(x[i])
        provenance.append({'domain': domain, 'manifest': str(manifest), 'manifest_sha256': digest(manifest),
                           'cache': str(cache), 'features_sha256': meta['features_sha256'], 'n': len(source_rows)})
        metas.append(meta)
    assert len(rows) == 1334 and len({r['pair_id'] for r in rows}) == 667
    assert Counter(r['label'] for r in rows) == {0: 667, 1: 667}
    assert Counter(r['training_source'] for r in rows) == {'robojailbench': 204, 'jaewoo_v2': 946, 'sungmin_v8': 184}
    assert len({r['group_id'] for r in rows}) == 5
    norm = lambda s: ' '.join(re.findall(r'\w+', s.casefold()))
    texts = [norm(r['command']) for r in rows]
    near = []
    for i, r in enumerate(rows):
        for j in range(i+1, len(rows)):
            if r['group_id'] == rows[j]['group_id']:
                continue
            a, b = texts[i], texts[j]
            if 2*min(len(a), len(b))/(len(a)+len(b)) < .9:
                continue
            matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
            if matcher.quick_ratio() >= .9 and matcher.ratio() >= .9:
                near.append({'a': r['sample_id'], 'b': rows[j]['sample_id'], 'ratio': matcher.ratio()})
    assert not near, 'Review near-duplicates across groups before training'
    out.mkdir(parents=True)
    shutil.copy2(__file__, out/'train_all_paired_f1t.py')
    dataset = out/'dataset'; (dataset/'images').mkdir(parents=True)
    for r, origin in zip(rows, origins):
        relative = 'images/'+r['image_sha256']+origin.suffix.lower()
        dest = dataset/relative
        if not dest.exists(): shutil.copy2(origin, dest)
        r['image_path'] = relative
    manifest = dataset/'selected.jsonl'
    manifest.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')
    rows = load_manifest(manifest)
    write_json(dataset/'provenance.json', provenance)
    duplicate_texts = defaultdict(list)
    for r, t in zip(rows, texts): duplicate_texts[t].append(r['sample_id'])
    audit = {'samples': len(rows), 'pairs': 667, 'labels': dict(Counter(r['label'] for r in rows)),
             'sources': dict(Counter(r['training_source'] for r in rows)),
             'groups': dict(Counter(r['group_id'] for r in rows)),
             'unique_image_hashes': len({r['image_sha256'] for r in rows}),
             'unique_normalized_commands': len(set(texts)),
             'duplicate_command_ids': [ids for ids in duplicate_texts.values() if len(ids)>1],
             'cross_group_near_text_matches_at_0_9': near,
             'near_text_method': 'word-normalized SequenceMatcher ratio >= 0.9; heuristic, not semantic proof',
             'original_camera_frame_unverified_samples': 184,
             'independent_test': False}
    write_json(dataset/'audit.json', audit)
    write_json(out/'config.json', config)
    x = np.stack(features)
    cache = out/'cache'; cache.mkdir()
    np.save(cache/'features.npy', x, allow_pickle=False)
    meta = {'format_version': 1, 'manifest_sha256': digest(manifest), 'features_sha256': digest(cache/'features.npy'),
            'config': config, 'encoder_lock': lock, 'feature_order': ['text_l2_512', 'image_l2_512'],
            'max_observed_tokens': max(m['max_observed_tokens'] for m in metas),
            'unique_commands': len({r['command'] for r in rows}), 'unique_images': audit['unique_image_hashes'],
            'rows': rows, 'complete': True, 'package_versions': metas[-1]['package_versions'],
            'reused_samples': len(rows), 'newly_encoded_commands': 0, 'source_caches': provenance}
    write_json(cache/'metadata.json', meta)
    x, _ = load_cache(cache)
    y = np.array([r['label'] for r in rows])
    domains = np.array([r['training_domain'] for r in rows])
    groups = np.array([r['group_id'] for r in rows])
    source_names = np.array([r['training_source'] for r in rows])
    print('AUDIT '+json.dumps(audit), flush=True)

    def fit(indices, balanced=False):
        xx, yy, dd = x[indices], y[indices], domains[indices]
        counts = Counter(dd)
        weights = np.array([len(indices)/(len(counts)*counts[d]) for d in dd], dtype=np.float32)
        if not balanced or len(counts) == 1:
            model, history = train_model(xx, yy, config, 'cpu')
            return model, history
        random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
        model = create_mlp(1024, config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        inputs = torch.tensor(xx, dtype=torch.float32)
        targets = torch.tensor(yy, dtype=torch.float32)
        w = torch.tensor(weights)
        assert abs(float(w.mean())-1) < 1e-6
        for d in counts: assert np.isclose(weights[dd==d].sum()/weights.sum(), 1/len(counts))
        generator = torch.Generator().manual_seed(config['seed'])
        history = []
        for _ in range(config['epochs']):
            model.train(); total = 0.
            for batch in torch.randperm(len(indices), generator=generator).split(config['batch_size']):
                optimizer.zero_grad(set_to_none=True)
                per_row = torch.nn.functional.binary_cross_entropy_with_logits(model(inputs[batch]).squeeze(-1), targets[batch], reduction='none')
                loss = (per_row*w[batch]).mean()
                assert torch.isfinite(loss)
                loss.backward(); optimizer.step()
                total += float(loss.detach())*len(batch)
            history.append(total/len(indices))
        return model, history

    def checked_split(tr, va):
        for key in ('sample_id', 'pair_id', 'image_sha256', 'group_id'):
            assert {rows[i][key] for i in tr}.isdisjoint({rows[i][key] for i in va}), key
        assert {texts[i] for i in tr}.isdisjoint({texts[i] for i in va})

    predictions = {}
    reports = {}
    # Fixed schemes, epochs, seed, and threshold. No test-driven threshold search.
    for scheme in ('rjb_only', 'combined_uniform', 'combined_domain_balanced'):
        p = np.full(len(rows), np.nan); folds = []; splits = []
        for group in sorted(set(groups)):
            va = np.flatnonzero(groups == group)
            tr = np.flatnonzero((groups != group) & ((domains == 'robojailbench') if scheme == 'rjb_only' else True))
            checked_split(tr, va)
            model, history = fit(tr, balanced=scheme.endswith('balanced'))
            p[va] = scores(model, x[va], 'cpu')
            folds.append({'held_group': str(group), 'train_n': len(tr), 'train_domain_counts': dict(Counter(domains[tr])),
                          'validation_n': len(va), 'metrics': metrics(y[va], p[va]), 'train_loss': history,
                          'optimizer_steps': int(np.ceil(len(tr)/config['batch_size']))*config['epochs']})
            splits.append({'held_group': str(group), 'train_ids': [rows[i]['sample_id'] for i in tr],
                           'validation_ids': [rows[i]['sample_id'] for i in va]})
            print(scheme+' '+str(group)+' '+json.dumps(folds[-1]['metrics']), flush=True)
        assert np.isfinite(p).all()
        by_source = {s: metrics(y[source_names==s], p[source_names==s]) for s in sorted(set(source_names))}
        report = {'scope': 'GROUP_HELD_OUT_RESEARCH_DIAGNOSTIC', 'independent_test': False,
                  'threshold': .5, 'threshold_calibrated': False, 'folds': folds,
                  'pooled_metrics': metrics(y,p), 'by_source': by_source,
                  'macro_group_metrics': {k: float(np.mean([f['metrics'][k] for f in folds])) for k in ('accuracy_at_0_5','harmful_miss_rate_at_0_5','normal_false_block_rate_at_0_5','auroc')}}
        target = out/'diagnostics'/scheme; target.mkdir(parents=True)
        write_json(target/'report.json', report); write_json(target/'splits.json', splits)
        write_json(target/'predictions.json', [{'sample_id': r['sample_id'], 'group_id': r['group_id'],
                   'training_source': r['training_source'], 'label': r['label'], 'score': float(v)} for r,v in zip(rows,p)])
        predictions[scheme] = p; reports[scheme] = report

    all_indices = np.arange(len(rows)); fits = {}
    for scheme in ('combined_uniform', 'combined_domain_balanced'):
        model, history = fit(all_indices, balanced=scheme.endswith('balanced'))
        p = scores(model,x,'cpu')
        target = out/'fits'/scheme; target.mkdir(parents=True)
        save_file({k: v.detach().cpu().contiguous() for k,v in model.state_dict().items()}, str(target/'model.safetensors'))
        restored = create_mlp(1024,config); restored.load_state_dict(load_file(str(target/'model.safetensors')))
        assert np.array_equal(p,scores(restored,x,'cpu')), 'Checkpoint reload mismatch'
        fits[scheme] = {'train_samples':len(rows), 'input_dimension':1024, 'train_loss':history,
                        'checkpoint':str((target/'model.safetensors').relative_to(out)),
                        'checkpoint_sha256':digest(target/'model.safetensors'),
                        'training_only_metrics':metrics(y,p), 'initialization':'fresh_seed_42',
                        'config':config, 'encoder_lock':lock, 'features_sha256':meta['features_sha256'],
                        'manifest_sha256':meta['manifest_sha256'], 'deployment_ready':False,
                        'operational_thresholds':None, 'checkpoint_reload_verified':True,
                        'loss_weighting':'equal total loss mass per training domain (RJB/local)' if scheme.endswith('balanced') else 'equal sample weights'}
        write_json(target/'report.json',fits[scheme])
        print('FIT '+scheme+' '+json.dumps(fits[scheme]['training_only_metrics']),flush=True)
    assert all(digest(Path(p))==sha for p,sha in protected.items()), 'Prior checkpoint changed'
    write_json(out/'prior-checkpoints.json', {'unchanged':True,'sha256':protected})
    summary = {'samples':len(rows), 'pairs':667, 'audit':audit, 'fits':fits,
               'comparison':{scheme:{'by_source':v['by_source'],'macro_group_metrics':v['macro_group_metrics']} for scheme,v in reports.items()},
               'independent_test':False, 'deployment_ready':False, 'runtime_model_replaced':False,
               'encoders_frozen':True,'features_reused':len(rows), 'device':'cpu',
               'domain_balance_preset_before_evaluation':True,
               'same_evaluation_sample_ids_across_schemes':True,
               'same_epochs_but_more_optimizer_steps_with_combined_training':True,
               'prior_checkpoints_unchanged':True,'elapsed_seconds':round(time.monotonic()-started,2)}
    write_json(out/'summary.json', summary)
    print('COMPLETE '+json.dumps(summary['comparison']),flush=True)


def local_diagnostic(repo, out):
    """Matched to the prior local held-hazard diagnostic, not independent testing."""
    import numpy as np
    import torch
    from f1.training.data import read_config, write_json, digest
    from f1.training.encoder import load_cache
    from f1.training.learner import create_mlp, train_model, scores, metrics
    x, meta = load_cache(out/'cache'); rows = meta['rows']
    config = read_config(out/'config.json'); torch.set_num_threads(config['cpu_threads'])
    y = np.array([r['label'] for r in rows]); local = np.array([r['training_domain']=='local' for r in rows])
    hazards = sorted({r['hazard_id'] for r in rows if r['training_domain']=='local'})
    prior_root = repo/'artifacts/f1t/combined-jaewoo-v2-sungmin-v8-cpu-run1'
    prior = json.loads((prior_root/'held-hazard-diagnostic/predictions.json').read_text(encoding='utf-8'))
    baseline = {r['sample_id']:r['score'] for r in prior}
    assert set(baseline)=={r['sample_id'] for r in rows if r['training_domain']=='local'}
    output = out/'same-scene-held-hazard-diagnostic'; output.mkdir(exist_ok=False)
    result = {'scope':'SAME_SCENE_HELD_HAZARD_DIAGNOSTIC_ONLY', 'independent_test':False,
              'same_local_validation_sample_ids_as_previous':True, 'same_scene_and_shared_templates':True,
              'threshold':0.5, 'threshold_calibrated':False, 'schemes':{}}
    schemes = ['prior_local_only', 'combined_uniform', 'combined_domain_balanced']
    for scheme in schemes:
        p = np.full(len(rows),np.nan); folds = []; splits=[]
        if scheme=='prior_local_only':
            for i,r in enumerate(rows):
                if local[i]:p[i]=baseline[r['sample_id']]
        else:
            for hazard in hazards:
                va=np.array([i for i,r in enumerate(rows) if local[i] and r['hazard_id']==hazard])
                tr=np.array([i for i,r in enumerate(rows) if not local[i] or r['hazard_id']!=hazard])
                assert {rows[i]['pair_id'] for i in tr}.isdisjoint({rows[i]['pair_id'] for i in va})
                # Only local hazard labels are comparable. Public hazard semantics may overlap.
                if scheme=='combined_uniform':
                    model, history=train_model(x[tr],y[tr],config,'cpu')
                else:
                    random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed'])
                    model=create_mlp(1024,config)
                    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
                    tx=torch.tensor(x[tr],dtype=torch.float32);ty=torch.tensor(y[tr],dtype=torch.float32)
                    domains=local[tr];counts=Counter(domains)
                    w=torch.tensor([len(tr)/(2*counts[d]) for d in domains],dtype=torch.float32)
                    generator=torch.Generator().manual_seed(config['seed']);history=[]
                    for _ in range(config['epochs']):
                        model.train();total=0.
                        for batch in torch.randperm(len(tr),generator=generator).split(config['batch_size']):
                            optimizer.zero_grad(set_to_none=True)
                            loss=(torch.nn.functional.binary_cross_entropy_with_logits(model(tx[batch]).squeeze(-1),ty[batch],reduction='none')*w[batch]).mean()
                            assert torch.isfinite(loss)
                            loss.backward();optimizer.step();total+=float(loss.detach())*len(batch)
                        history.append(total/len(tr))
                p[va]=scores(model,x[va],'cpu')
                folds.append({'held_local_hazard':hazard,'train_n':len(tr),'metrics':metrics(y[va],p[va]),'train_loss':history})
                splits.append({'held_local_hazard':hazard,'train_ids':[rows[i]['sample_id'] for i in tr],'validation_ids':[rows[i]['sample_id'] for i in va]})
        assert np.isfinite(p[local]).all()
        result['schemes'][scheme]={'pooled_metrics':metrics(y[local],p[local]), 'folds':folds,
             'by_source':{s:metrics(y[ii],p[ii]) for s in ('jaewoo_v2','sungmin_v8')
                          for ii in [np.array([i for i,r in enumerate(rows) if r['training_source']==s])]} }
        write_json(output/(scheme+'-predictions.json'),[{'sample_id':r['sample_id'],'label':r['label'],'score':float(p[i])} for i,r in enumerate(rows) if local[i]])
        if splits:write_json(output/(scheme+'-splits.json'),splits)
        print('LOCAL '+scheme+' '+json.dumps(result['schemes'][scheme]['pooled_metrics']),flush=True)
    write_json(output/'report.json',result)
    # The file now reproduces both phases via the explicit CLI switch.
    shutil.copy2(__file__,out/'train_all_paired_f1t.py')
    summary_path=out/'summary.json';summary=json.loads(summary_path.read_text(encoding='utf-8'))
    summary['same_scene_held_hazard_comparison']={k:v['pooled_metrics'] for k,v in result['schemes'].items()}
    write_json(summary_path,summary)


if __name__ == '__main__':
    main()
