"""CPU fit plus explicitly same-scene, held-hazard diagnostic evaluation."""
import argparse
import json
import sys
import time
import shutil
import re
from collections import Counter
from pathlib import Path

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--repo',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();repo=args.repo.resolve();out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    sys.path.insert(0,str(repo/'src'))
    import numpy as np
    import torch
    from importlib.metadata import version
    from sentence_transformers import SentenceTransformer
    from safetensors.torch import load_file
    from f1.training.data import load_manifest,read_config,write_json,digest
    from f1.training.encoder import FrozenClip,load_cache
    from f1.training.learner import run,create_mlp,scores,metrics,train_model
    sources=[repo/'data/jaewoo_f1t_pairs_english_v2/selected.jsonl',repo/'data/sungmin_v8_f1t_english_v1/selected.jsonl']
    expected_hashes=['81c11ae7c7f56942817bfd7d1e047155b68bd9e8302f6edabe1716070f527a35','097bb02a58e335cfb5b7c3861e7702bf70760bf41cb88e8c5d06f0149d1da04b']
    dataset=out/'dataset';dataset.mkdir();(dataset/'images').mkdir()
    rows=[]
    for src,expected,name in zip(sources,expected_hashes,['jaewoo_v2','sungmin_v8']):
        assert digest(src)==expected,'Source manifest changed'
        for raw in load_manifest(src):
            r=dict(raw,training_source=name)
            image=dataset/r['image_path']
            if not image.exists():shutil.copy2(src.parent/r['image_path'],image)
            assert digest(image)==r['image_sha256']
            rows.append(r)
    assert len(rows)==1130 and len({r['pair_id'] for r in rows})==565
    assert Counter(r['label'] for r in rows)=={0:565,1:565}
    assert len({' '.join(re.findall(r'\w+',r['command'].casefold())) for r in rows})==1130
    assert len({r['image_sha256'] for r in rows})==len({r['group_id'] for r in rows})==1
    manifest=dataset/'selected.jsonl'
    with manifest.open('w',encoding='utf-8') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    rows=load_manifest(manifest)
    write_json(dataset/'provenance.json',{'source_manifests':[str(p) for p in sources],'source_manifest_sha256':expected_hashes,'samples_by_source':dict(Counter(r['training_source'] for r in rows)),'substituted_image_samples':184,'single_shared_scene':True,'no_independent_test':True})
    config=read_config(repo/'configs/f1t-train.json')
    config.update(text_revision='58edf8cada9e398793dca955574a48cbb7f18be2',image_revision='327ab6726d33c0e22f920c83f2ff9e4bd38ca37f')
    config_path=out/'config.json';write_json(config_path,config)
    torch.set_num_threads(config['cpu_threads'])

    class LocalFrozenClip(FrozenClip):
        def __init__(self):
            self.config=config
            self.lock={k:{'model':config[k+'_model'],'revision':config[k+'_revision']} for k in ('text','image')}
            for k in ('text','image'):
                model=SentenceTransformer(self.lock[k]['model'],revision=self.lock[k]['revision'],device='cpu',trust_remote_code=False,local_files_only=True)
                model.eval()
                for param in model.parameters():param.requires_grad_(False)
                setattr(self,k,model)
            assert self.text.max_seq_length==128

    prior_cache=repo/'artifacts/f1t/jaewoo-v2-cpu-run1/cache'
    prior_x,prior_meta=load_cache(prior_cache)
    assert prior_meta['manifest_sha256']==expected_hashes[0]
    for key in ['text_model','image_model','text_revision','image_revision','preprocessing','embedding_dimension','max_tokens']:
        assert config[key]==prior_meta['config'][key]
    assert prior_meta['encoder_lock']=={k:{'model':config[k+'_model'],'revision':config[k+'_revision']} for k in ['text','image']}
    texts={r['command']:prior_x[i,:512] for i,r in enumerate(prior_meta['rows'])}
    cached_rows={r['sample_id']:r for r in prior_meta['rows']}
    for r in rows:
        if r['training_source']=='jaewoo_v2':
            old=cached_rows[r['sample_id']]
            assert (r['command_sha256'],r['image_sha256'],r['label'])==(old['command_sha256'],old['image_sha256'],old['label'])
    print('Reusing 946 verified cached features. Loading pinned local encoders for 184 new commands.',flush=True)
    encoder=LocalFrozenClip();commands=list(dict.fromkeys(r['command'] for r in rows));missing=[s for s in commands if s not in texts]
    images={r['image_sha256']:manifest.parent/r['image_path'] for r in rows}
    assert len(missing)==184
    print(f'Encoding {len(missing)} new commands and checking shared image on CPU.',flush=True)
    text,image,new_lengths=encoder.encode(missing,list(images.values()),config['batch_size'])
    assert np.allclose(image[0],prior_x[0,512:],atol=1e-6),'Cached image encoding changed'
    texts.update(zip(missing,text));ims=dict(zip(images,image))
    lengths=[r['token_count'] for r in rows if r['training_source']=='jaewoo_v2']+new_lengths
    x=np.stack([np.concatenate((texts[r['command']],ims[r['image_sha256']])) for r in rows])
    cache=out/'cache';cache.mkdir();np.save(cache/'features.npy',x,allow_pickle=False)
    meta={'format_version':1,'manifest_sha256':digest(manifest),'features_sha256':digest(cache/'features.npy'),
        'config':config,'encoder_lock':encoder.lock,'feature_order':['text_l2_512','image_l2_512'],
        'reused_samples':946,'newly_encoded_commands':184,'prior_cache_features_sha256':prior_meta['features_sha256'],'max_observed_tokens':max(lengths),'unique_commands':len(commands),'unique_images':len(images),'rows':rows,
        'complete':True,'package_versions':{name:version(name) for name in ['torch','numpy','Pillow','sentence-transformers','transformers']}}
    write_json(cache/'metadata.json',meta)
    del encoder,text,image,texts,ims
    x,meta=load_cache(cache);y=np.array([r['label'] for r in rows])
    print('Cache verified. Fitting both-feature MLP, 30 epochs, seed 42.',flush=True)
    fit=run(cache,config_path,out/'fit',mode='fit',features='both',device='cpu')
    model=create_mlp(1024,config);model.load_state_dict(load_file(str(out/'fit/model.safetensors')))
    p=scores(model,x,'cpu');train_metrics=metrics(y,p)
    write_json(out/'fit/training_metrics.json',{'evaluation_scope':'RESUBSTITUTION_TRAINING_DATA_ONLY','metrics':train_metrics})
    write_json(out/'fit/training_predictions.json',[{'sample_id':r['sample_id'],'label':r['label'],'score':float(v)} for r,v in zip(rows,p)])
    print('Saved final model. Training-only metrics: '+json.dumps(train_metrics),flush=True)

    # Explicitly diagnostic: scene is shared. Do not rewrite scene groups or call
    # this an independent scene holdout. Hyperparameters and epochs stay fixed.
    diag=out/'held-hazard-diagnostic';diag.mkdir()
    hazards=sorted({r['hazard_id'] for r in rows});oof=np.full(len(rows),np.nan);folds=[];splits=[]
    for h in hazards:
        va=np.array([i for i,r in enumerate(rows) if r['hazard_id']==h]);tr=np.array([i for i,r in enumerate(rows) if r['hazard_id']!=h])
        assert {rows[i]['pair_id'] for i in va}.isdisjoint({rows[i]['pair_id'] for i in tr})
        diagnostic_model,history=train_model(x[tr],y[tr],config,'cpu')
        vp=scores(diagnostic_model,x[va],'cpu');oof[va]=vp
        fold={'held_hazard':h,'train_n':len(tr),'validation_n':len(va),'metrics':metrics(y[va],vp),'train_loss':history}
        folds.append(fold);splits.append({'held_hazard':h,'train_ids':[rows[i]['sample_id'] for i in tr],'validation_ids':[rows[i]['sample_id'] for i in va]})
        print('Diagnostic '+h+': '+json.dumps(fold['metrics']),flush=True)
    assert np.isfinite(oof).all()
    report={'scope':'SAME_SCENE_LEAVE_ONE_HAZARD_OUT_DIAGNOSTIC','independent_test':False,'shared_scene':True,'shared_attack_templates':True,
        'hyperparameter_selection_performed':False,'threshold':0.5,'threshold_calibrated':False,
        'pooled_metrics':metrics(y,oof),'folds':folds,
        'macro_accuracy':float(np.mean([f['metrics']['accuracy_at_0_5'] for f in folds])),
        'manifest_sha256':meta['manifest_sha256'],'features_sha256':meta['features_sha256'],'config':config}
    write_json(diag/'report.json',report);write_json(diag/'splits.json',splits)
    write_json(diag/'predictions.json',[{'sample_id':r['sample_id'],'pair_id':r['pair_id'],'hazard_id':r['hazard_id'],'label':r['label'],'score':float(v),'predicted_label':int(v>=0.5)} for r,v in zip(rows,oof)])
    summary={'dataset':'jaewoo_v2_plus_sungmin_v8','training_samples':len(rows),'pairs':565,'device':'cpu',
        'encoder_frozen':True,'mlp':'1024 -> 128 -> 1','initialization':'fresh_seed_42','epochs':config['epochs'],
        'checkpoint':'fit/model.safetensors','checkpoint_sha256':digest(out/'fit/model.safetensors'),
        'manifest_sha256':meta['manifest_sha256'],'training_only_metrics':train_metrics,
        'same_scene_held_hazard_diagnostic_metrics':report['pooled_metrics'],'independent_test':False,'deployment_ready':False,
        'prior_checkpoints_unchanged':True,'elapsed_seconds':round(time.monotonic()-started,2)}
    source_metrics={}
    for name in ['jaewoo_v2','sungmin_v8']:
        ii=[i for i,r in enumerate(rows) if r['training_source']==name]
        source_metrics[name]=metrics(y[ii],oof[ii])
    prior_pred=json.loads((repo/'artifacts/f1t/jaewoo-v2-cpu-run1/held-hazard-diagnostic/predictions.json').read_text(encoding='utf-8'))
    pp={r['sample_id']:r for r in prior_pred};ji=[i for i,r in enumerate(rows) if r['training_source']=='jaewoo_v2']
    assert {rows[i]['sample_id'] for i in ji}==set(pp)
    baseline=metrics(y[ji],np.array([pp[rows[i]['sample_id']]['score'] for i in ji]))
    compare={'scope':'SAME_946_JAEWOO_SAMPLES_HELD_HAZARD_DIAGNOSTIC_ONLY','independent_test':False,
        'previous_jaewoo_only_training':baseline,'combined_training_jaewoo_subset':source_metrics['jaewoo_v2'],
        'combined_diagnostic_by_source':source_metrics,
        'epochs_and_seed_unchanged':True,'more_training_updates_per_epoch_due_to_more_data':True}
    write_json(out/'comparison.json',compare)
    summary['same_cohort_comparison']=compare
    write_json(out/'summary.json',summary)
    print('COMPLETE '+json.dumps(summary),flush=True)

if __name__=='__main__':main()
