"""CPU fit plus explicitly same-scene, held-hazard diagnostic evaluation."""
import argparse
import json
import sys
import time
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
    manifest=repo/'data/jaewoo_f1t_pairs_english_v2/selected.jsonl'
    rows=load_manifest(manifest)
    assert len(rows)==946 and len({r['pair_id'] for r in rows})==473
    assert len({r['image_sha256'] for r in rows})==1
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

    print('Loading pinned encoders from local cache; no downloads.',flush=True)
    encoder=LocalFrozenClip();commands=list(dict.fromkeys(r['command'] for r in rows))
    images={r['image_sha256']:manifest.parent/r['image_path'] for r in rows}
    print(f'Encoding {len(commands)} commands and {len(images)} image on CPU.',flush=True)
    text,image,lengths=encoder.encode(commands,list(images.values()),config['batch_size'])
    texts=dict(zip(commands,text));ims=dict(zip(images,image))
    x=np.stack([np.concatenate((texts[r['command']],ims[r['image_sha256']])) for r in rows])
    cache=out/'cache';cache.mkdir();np.save(cache/'features.npy',x,allow_pickle=False)
    meta={'format_version':1,'manifest_sha256':digest(manifest),'features_sha256':digest(cache/'features.npy'),
        'config':config,'encoder_lock':encoder.lock,'feature_order':['text_l2_512','image_l2_512'],
        'max_observed_tokens':max(lengths),'unique_commands':len(commands),'unique_images':len(images),'rows':rows,
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
    summary={'dataset':'jaewoo_f1t_pairs_english_v2','training_samples':len(rows),'pairs':473,'device':'cpu',
        'encoder_frozen':True,'mlp':'1024 -> 128 -> 1','initialization':'fresh_seed_42','epochs':config['epochs'],
        'checkpoint':'fit/model.safetensors','checkpoint_sha256':digest(out/'fit/model.safetensors'),
        'manifest_sha256':meta['manifest_sha256'],'training_only_metrics':train_metrics,
        'same_scene_held_hazard_diagnostic_metrics':report['pooled_metrics'],'independent_test':False,'deployment_ready':False,
        'prior_checkpoints_unchanged':True,'elapsed_seconds':round(time.monotonic()-started,2)}
    write_json(out/'summary.json',summary)
    print('COMPLETE '+json.dumps(summary),flush=True)

if __name__=='__main__':main()
