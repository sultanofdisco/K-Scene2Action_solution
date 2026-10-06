"""Controlled frozen multilingual CLIP experiment, no fabricated image inputs."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import random
import shutil
import sys
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();repo=args.repo.resolve();out=args.output.resolve()
    sys.path.insert(0,str(repo/'src'))
    import numpy as np
    import torch
    import torch.nn.functional as F
    from safetensors.torch import save_file,load_file
    from f1.training.data import digest,load_manifest,read_config,write_json
    from f1.training.encoder import load_cache
    from f1.training.learner import create_mlp,train_model,scores,metrics
    from f1.training.text_aux import load_aux,validate_rows

    started=time.monotonic()
    assert not out.exists(), 'Use a new experiment directory'
    protected={str(p):digest(p) for p in (repo/'artifacts/f1t').rglob('*.safetensors')}
    paired_root=repo/'artifacts/f1t/all-paired-rjb-local-cpu-run1'
    x,pm=load_cache(paired_root/'cache')
    rows=load_manifest(paired_root/'dataset/selected.jsonl')
    assert rows==pm['rows'] and digest(paired_root/'dataset/selected.jsonl')==pm['manifest_sha256']
    config=read_config(paired_root/'config.json');torch.set_num_threads(config['cpu_threads'])
    lock=pm['encoder_lock']['text']
    assert config['text_revision']==lock['revision']
    aux_rows=[];aux_features=[];provenance=[]
    for source,data,cache in [
        ('sab',repo/'data/safeagentbench_f1t_english_v1/splits_v1',repo/'artifacts/f1t/text-aux-safeagentbench-v1/cache'),
        ('safetext',repo/'data/public_f1_bundle_v1/text_aux',repo/'artifacts/f1t/text-aux-public-v1/cache')]:
        rr,hashes=load_aux(data)
        meta=json.loads((cache/'metadata.json').read_text(encoding='utf-8'))
        assert meta['complete'] and meta['format']=='f1t-text-aux-cache-1'
        validate_rows(meta['rows'])
        assert rr==meta['rows'] and hashes==meta['manifest_hashes']
        assert meta['revision']==lock['revision'] and meta['text_model']==lock['model']
        assert digest(cache/'features.npy')==meta['features_sha256'] and meta['max_tokens']<=128
        xx=np.load(cache/'features.npy',allow_pickle=False)
        assert xx.shape==(len(rr),512) and np.isfinite(xx).all()
        assert np.allclose(np.linalg.norm(xx,axis=1),1,atol=1e-4)
        aux_rows.extend([dict(r,aux_source=source) for r in rr]);aux_features.extend(xx)
        provenance.append({'source':source,'data':str(data),'cache':str(cache),'manifest_hashes':hashes,
                           'features_sha256':meta['features_sha256'],'split_counts':dict(Counter(r['split'] for r in rr))})
    a=np.stack(aux_features);ay=np.array([r['label'] for r in aux_rows])
    atr=np.array([i for i,r in enumerate(aux_rows) if r['split']=='train'])
    sabtr=np.array([i for i in atr if aux_rows[i]['aux_source']=='sab'])
    assert len(atr)==438 and len(sabtr)==366
    assert all(aux_rows[i]['split']=='train' for i in atr)
    # Cross-source matching was run before looking at any new scores.
    audit_path=Path(__file__).parent/'aux_cross_source_audit.json'
    cross=json.loads(audit_path.read_text(encoding='utf-8'))
    assert cross==[], 'Resolve cross-source overlap before fitting'
    # Recompute matching inside the runnable experiment, not just trust the audit file.
    import difflib,re
    normalize=lambda s:' '.join(re.findall(r'\w+',s.casefold()))
    corpora={'paired':rows,'sab':[r for r in aux_rows if r['aux_source']=='sab'],
             'safetext':[r for r in aux_rows if r['aux_source']=='safetext']}
    def forms(r):
        values=[normalize(r.get('text',r.get('command')))]
        if 'proposed_action' in r:values.append(normalize(r['proposed_action']))
        return values
    matches=[]
    for left,right in [('paired','sab'),('paired','safetext'),('sab','safetext')]:
        for ra in corpora[left]:
            for rb in corpora[right]:
                for sa in forms(ra):
                    for sb in forms(rb):
                        if 2*min(len(sa),len(sb))/(len(sa)+len(sb))<.9:continue
                        match=difflib.SequenceMatcher(None,sa,sb,autojunk=False)
                        if match.quick_ratio()>=.9 and match.ratio()>=.9:
                            matches.append((ra['sample_id'],rb['sample_id']))
    assert not matches, 'Cross-source near-text leakage'
    y=np.array([r['label'] for r in rows]);groups=np.array([r['group_id'] for r in rows])
    sources=np.array([r['training_source'] for r in rows]);local=sources!='robojailbench'
    schemes=['paired_only','joint_sab','joint_sab_safetext','pretrain_then_paired']
    seeds=[42,7,19]
    plan={'paired_samples':len(rows),'auxiliary_train_samples':438,'safeagentbench_train':366,'safetext_train':72,
          'auxiliary_validation':{'sab':31,'safetext':16},'reserved_test_not_scored':{'sab':32,'safetext':16},
          'schemes':schemes,'seeds':seeds,'paired_epochs':30,'auxiliary_pretrain_epochs':30,
          'auxiliary_loss_lambda':0.25,'joint_auxiliary_weighting':'uniform over selected auxiliary training rows',
          'paired_loss_weighting':'uniform over paired rows, identical for all schemes',
          'architecture':'1024 -> 128 ReLU Dropout(0.1) -> 1; shared text columns, bias, output weights',
          'text_only_forward':'output(dropout(relu(linear(text,W_first[:,:512],b_first))))',
          'image_feature_for_auxiliary':None,'encoders_frozen':True,'threshold':0.5,
          'hyperparameter_search':False,'paired_group_folds':5,'local_hazard_folds':12,
          'protected_test_evaluation':False,'deployment_ready':False,'config':config,
          'sequential_image_weight_initialization':'retain seeded paired-model image columns; transfer text columns, first bias, output layer',
          'pretrain_epochs_are_additional':True,'comparison_to_old_weighted_model':'not a matched ablation; primary baseline is uniform',
          'auxiliary_labels':'physical action/advice safety, not malicious intent or authorization',
          'cross_source_matching':'word-normalized SequenceMatcher >=0.9, full text and SafeText proposed_action; no semantic guarantee'}
    out.mkdir(parents=True)
    write_json(out/'experiment-plan.json',plan)
    write_json(out/'provenance.json',{'paired_cache':str(paired_root/'cache'),'paired_manifest_sha256':pm['manifest_sha256'],
                'paired_features_sha256':pm['features_sha256'],'auxiliary':provenance,'encoder_lock':pm['encoder_lock']})
    shutil.copy2(__file__,out/'train_paired_with_text_aux.py')
    shutil.copy2(audit_path,out/'aux_cross_source_audit.json')
    write_json(out/'auxiliary-training-manifest.json',[aux_rows[i] for i in atr])
    print('PLAN '+json.dumps(plan),flush=True)

    def text_logits(model,tx):
        h=F.linear(tx,model[0].weight[:,:512],model[0].bias)
        return model[3](model[2](model[1](h))).squeeze(-1)

    def text_scores(model,xx):
        model.eval()
        with torch.inference_mode():return text_logits(model,torch.tensor(xx,dtype=torch.float32)).sigmoid().numpy()

    # A meaningful mechanism test: auxiliary gradients cannot update image columns.
    torch.manual_seed(42);probe=create_mlp(1024,config)
    F.binary_cross_entropy_with_logits(text_logits(probe,torch.tensor(a[atr[:8]],dtype=torch.float32)),
                torch.tensor(ay[atr[:8]],dtype=torch.float32)).backward()
    assert torch.count_nonzero(probe[0].weight.grad[:,512:])==0
    assert torch.count_nonzero(probe[0].weight.grad[:,:512])>0
    probe.zero_grad();probe(torch.tensor(x[:8],dtype=torch.float32)).sum().backward()
    assert torch.count_nonzero(probe[0].weight.grad[:,512:])>0
    write_json(out/'mechanism-verification.json',{'auxiliary_image_gradient_zero':True,
               'auxiliary_text_gradient_nonzero':True,'paired_image_gradient_nonzero':True,'fake_image_used':False})

    def extended_metrics(yy,p):
        m=metrics(yy,p)
        m['balanced_accuracy_at_0_5']=1-(m['harmful_miss_rate_at_0_5']+m['normal_false_block_rate_at_0_5'])/2
        m.update(harmful_n=int((yy==1).sum()),normal_n=int((yy==0).sum()),
                 harmful_misses=int(((yy==1)&(p<.5)).sum()),false_blocks=int(((yy==0)&(p>=.5)).sum()))
        return m

    aux_pretrained={}
    for seed in seeds:
        cc=dict(config,seed=seed)
        model,history=train_model(a[atr],ay[atr],cc,'cpu')
        aux_pretrained[seed]=copy.deepcopy(model.state_dict())
        target=out/'auxiliary-pretraining'/f'seed-{seed}';target.mkdir(parents=True)
        save_file(aux_pretrained[seed],str(target/'text-only.safetensors'))
        report={'train_n':438,'train_loss':history,'validation':{},'test_evaluated':False}
        for source in ['sab','safetext']:
            idx=np.array([i for i,r in enumerate(aux_rows) if r['split']=='validation' and r['aux_source']==source])
            report['validation'][source]=extended_metrics(ay[idx],scores(model,a[idx],'cpu'))
        write_json(target/'report.json',report)

    def fit(indices,scheme,seed):
        cc=dict(config,seed=seed)
        if scheme=='paired_only':return train_model(x[indices],y[indices],cc,'cpu')
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        model=create_mlp(1024,cc)
        if scheme=='pretrain_then_paired':
            state=aux_pretrained[seed]
            with torch.no_grad():
                model[0].weight[:,:512].copy_(state['0.weight'])
                model[0].bias.copy_(state['0.bias'])
                model[3].weight.copy_(state['3.weight']);model[3].bias.copy_(state['3.bias'])
        optimizer=torch.optim.AdamW(model.parameters(),lr=cc['learning_rate'],weight_decay=cc['weight_decay'])
        tx=torch.tensor(x[indices],dtype=torch.float32);ty=torch.tensor(y[indices],dtype=torch.float32)
        ai=sabtr if scheme=='joint_sab' else atr
        ax=torch.tensor(a[ai],dtype=torch.float32);at=torch.tensor(ay[ai],dtype=torch.float32)
        pg=torch.Generator().manual_seed(seed);ag=torch.Generator().manual_seed(seed+10000)
        history=[]
        steps=int(np.ceil(len(indices)/cc['batch_size']))
        for _ in range(cc['epochs']):
            model.train();pair_total=0.;aux_total=0.
            paired_batches=torch.randperm(len(indices),generator=pg).split(cc['batch_size'])
            aux_batches=torch.tensor_split(torch.randperm(len(ai),generator=ag),steps)
            seen_pair=0;seen_aux=0
            for batch,ab in zip(paired_batches,aux_batches):
                optimizer.zero_grad(set_to_none=True)
                paired_loss=F.binary_cross_entropy_with_logits(model(tx[batch]).squeeze(-1),ty[batch])
                loss=paired_loss
                if scheme.startswith('joint'):
                    assert len(ab)>0
                    aloss=F.binary_cross_entropy_with_logits(text_logits(model,ax[ab]),at[ab])
                    # Exactly one auxiliary pass per epoch, despite unequal corpus sizes.
                    loss=loss+.25*(len(ab)*steps/len(ai))*aloss
                    aux_total+=float(aloss.detach())*len(ab);seen_aux+=len(ab)
                assert torch.isfinite(loss)
                loss.backward();optimizer.step()
                pair_total+=float(paired_loss.detach())*len(batch);seen_pair+=len(batch)
            assert seen_pair==len(indices)
            if scheme.startswith('joint'):assert seen_aux==len(ai)
            history.append({'paired_loss':pair_total/len(indices),
                            'auxiliary_loss':aux_total/len(ai) if scheme.startswith('joint') else None})
        return model,history

    group_folds=[]
    for group in sorted(set(groups)):
        tr=np.flatnonzero(groups!=group);va=np.flatnonzero(groups==group)
        for key in ['sample_id','pair_id','image_sha256','group_id']:
            assert {rows[i][key] for i in tr}.isdisjoint({rows[i][key] for i in va})
        assert {normalize(rows[i]['command']) for i in tr}.isdisjoint({normalize(rows[i]['command']) for i in va})
        group_folds.append((str(group),tr,va))
    hazard_folds=[]
    for hazard in sorted({r['hazard_id'] for i,r in enumerate(rows) if local[i]}):
        va=np.array([i for i,r in enumerate(rows) if local[i] and r['hazard_id']==hazard])
        tr=np.array([i for i,r in enumerate(rows) if not local[i] or r['hazard_id']!=hazard])
        assert {rows[i]['pair_id'] for i in tr}.isdisjoint({rows[i]['pair_id'] for i in va})
        hazard_folds.append((hazard,tr,va))
    write_json(out/'splits.json',{
       track:[{'held':name,'paired_train_ids':[rows[i]['sample_id'] for i in tr],
               'validation_ids':[rows[i]['sample_id'] for i in va]} for name,tr,va in folds]
       for track,folds in [('group',group_folds),('same_scene_hazard',hazard_folds)]})
    reports={};all_idx=np.arange(len(rows))
    for scheme in schemes:
        reports[scheme]={}
        for seed in seeds:
            target=out/scheme/f'seed-{seed}';target.mkdir(parents=True)
            result={'seed':seed,'scheme':scheme,'diagnostics':{}}
            for track,folds in [('group',group_folds),('same_scene_hazard',hazard_folds)]:
                p=np.full(len(rows),np.nan);fold_reports=[]
                for name,tr,va in folds:
                    model,history=fit(tr,scheme,seed);p[va]=scores(model,x[va],'cpu')
                    fold_reports.append({'held':name,'train_n':len(tr),'metrics':extended_metrics(y[va],p[va]),'train_loss':history})
                idx=all_idx if track=='group' else np.flatnonzero(local)
                assert np.isfinite(p[idx]).all()
                report={'pooled':extended_metrics(y[idx],p[idx]),'by_source':{},'folds':fold_reports,
                        'independent_test':False,'same_scene':track=='same_scene_hazard'}
                for source in sorted(set(sources[idx])):
                    ii=np.array([i for i in idx if sources[i]==source]);report['by_source'][str(source)]=extended_metrics(y[ii],p[ii])
                report['macro_fold_metrics']={k:float(np.mean([f['metrics'][k] for f in fold_reports]))
                    for k in ['accuracy_at_0_5','harmful_miss_rate_at_0_5','normal_false_block_rate_at_0_5','auroc']}
                write_json(target/(track+'-report.json'),report)
                pred=[{'sample_id':rows[i]['sample_id'],'label':rows[i]['label'],'score':float(p[i])} for i in idx]
                write_json(target/(track+'-predictions.json'),pred)
                result['diagnostics'][track]={k:v for k,v in report.items() if k!='folds'}
                if scheme=='paired_only' and seed==42:
                    oldpath=(paired_root/'diagnostics/combined_uniform/predictions.json' if track=='group' else
                             paired_root/'same-scene-held-hazard-diagnostic/combined_uniform-predictions.json')
                    old={r['sample_id']:r['score'] for r in json.loads(oldpath.read_text(encoding='utf-8'))}
                    assert all(np.isclose(p[i],old[rows[i]['sample_id']],atol=1e-7) for i in idx),'Prior baseline not reproduced'
                scope=report['by_source'].get('robojailbench',report['pooled'])
                print(f'{scheme} seed={seed} {track}: '+json.dumps(scope),flush=True)
            model,history=fit(all_idx,scheme,seed)
            save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(target/'model.safetensors'))
            restored=create_mlp(1024,config);restored.load_state_dict(load_file(str(target/'model.safetensors')))
            p=scores(model,x,'cpu');assert np.array_equal(p,scores(restored,x,'cpu'))
            result['fit']={'train_n':len(rows),'training_only_metrics':extended_metrics(y,p),'train_loss':history,
                  'checkpoint_sha256':digest(target/'model.safetensors'),'encoder_lock':pm['encoder_lock'],
                  'config':dict(config,seed=seed),'checkpoint_reload_verified':True,'operational_thresholds':None,
                  'input_dimension':1024,'architecture_compatible_with_existing_mlp':True,'deployment_ready':False}
            validation={}
            for source in ['sab','safetext']:
                idx=np.array([i for i,r in enumerate(aux_rows) if r['split']=='validation' and r['aux_source']==source])
                ap=text_scores(model,a[idx]);assert np.array_equal(ap,text_scores(restored,a[idx]))
                validation[source]=extended_metrics(ay[idx],ap)
                write_json(target/(source+'-validation-predictions.json'),[
                    {'sample_id':aux_rows[i]['sample_id'],'label':aux_rows[i]['label'],'score':float(v)} for i,v in zip(idx,ap)])
            result['text_branch_validation']=validation;result['test_evaluated']=False
            write_json(target/'fit-report.json',result['fit']);write_json(target/'summary.json',result)
            reports[scheme][str(seed)]=result
    # Summarize seed variability; each seed evaluates identical samples, not extra independent data.
    summaries={}
    for scheme in schemes:
        cohorts={
          'rjb_group_diagnostic':[reports[scheme][str(s)]['diagnostics']['group']['by_source']['robojailbench'] for s in seeds],
          'local_same_scene_hazard':[reports[scheme][str(s)]['diagnostics']['same_scene_hazard']['pooled'] for s in seeds],
          'local_scene_held_out':[next(r['metrics'] for r in json.loads((out/scheme/f'seed-{s}'/'group-report.json').read_text(encoding='utf-8'))['folds'] if r['held'].startswith('scene-')) for s in seeds],
          'sab_validation':[reports[scheme][str(s)]['text_branch_validation']['sab'] for s in seeds],
          'safetext_validation':[reports[scheme][str(s)]['text_branch_validation']['safetext'] for s in seeds],
        }
        summaries[scheme]={}
        for cohort,values in cohorts.items():
            summaries[scheme][cohort]={'n_per_seed':values[0]['n'],'per_seed':dict(zip(map(str,seeds),values)),
              'mean':{k:float(np.mean([m[k] for m in values])) for k in ['accuracy_at_0_5','balanced_accuracy_at_0_5','harmful_miss_rate_at_0_5','normal_false_block_rate_at_0_5','auroc']},
              'std_population':{k:float(np.std([m[k] for m in values])) for k in ['accuracy_at_0_5','harmful_miss_rate_at_0_5','normal_false_block_rate_at_0_5']}}
    assert all(digest(Path(p))==sha for p,sha in protected.items()),'Prior checkpoint changed'
    write_json(out/'prior-checkpoints.json',{'unchanged':True,'sha256':protected})
    summary={'plan':plan,'comparison':summaries,'test_evaluated':False,'independent_test':False,
             'runtime_model_replaced':False,'prior_checkpoints_unchanged':True,'paired_seed42_baseline_reproduced':True,
             'elapsed_seconds':round(time.monotonic()-started,2),'plan_sha256':digest(out/'experiment-plan.json')}
    write_json(out/'summary.json',summary)
    print('COMPLETE '+json.dumps({k:{c:v['mean'] for c,v in cv.items()} for k,cv in summaries.items()}),flush=True)


if __name__=='__main__':main()
