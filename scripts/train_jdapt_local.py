"""J-DAPT source-derived local adaptation, frozen multilingual CLIP, CPU only."""
import argparse
import hashlib
import importlib.util
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, obj):
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def coral_fit(source, target, eps=1e-5):
    """Symmetric covariance whitening/recoloring; fit on training rows only."""
    s = torch.as_tensor(source, dtype=torch.float64)
    t = torch.as_tensor(target, dtype=torch.float64)
    sm, tm = s.mean(0), t.mean(0)
    sc, tc = s-sm, t-tm
    cs = sc.T@sc/(len(s)-1)+eps*torch.eye(s.shape[1], dtype=s.dtype)
    ct = tc.T@tc/(len(t)-1)+eps*torch.eye(t.shape[1], dtype=t.dtype)
    vs, us = torch.linalg.eigh(cs)
    vt, ut = torch.linalg.eigh(ct)
    ws = (us*vs.clamp_min(eps).rsqrt())@us.T
    wt = (ut*vt.clamp_min(eps).sqrt())@ut.T
    a = ws@wt
    assert torch.isfinite(a).all()
    return ((s-sm)@a+tm).float().numpy(), dict(matrix=a.float(), source_mean=sm.float(), target_mean=tm.float())


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args=ap.parse_args()
    repo=args.repo.resolve();out=args.output.resolve()
    if out.exists():
        raise RuntimeError('New output directory required; prior results are never overwritten')
    sys.path.insert(0,str(repo/'src'))
    from f1.training.encoder import load_cache
    from f1.training.learner import train_model, scores, metrics
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    start=time.monotonic()
    paired=repo/'artifacts/f1t/all-paired-rjb-local-cpu-run1'
    old=repo/'artifacts/f1t/paired-text-aux-cpu-run1'
    upstream=repo/'data/jdapt_public_sources_v1/provenance/jdapt-code'
    spec=importlib.util.spec_from_file_location('jdapt_official_models',upstream/'utils/models.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    Fusion=module.CrossAttentionFusion;Classifier=module.SimpleNN
    x,meta=load_cache(paired/'cache')
    rows=[json.loads(s) for s in (paired/'dataset/selected.jsonl').read_text(encoding='utf-8').splitlines()]
    assert rows==meta['rows'] and sha(paired/'dataset/selected.jsonl')==meta['manifest_sha256']
    assert x.shape==(1334,1024) and np.isfinite(x).all()
    assert np.allclose(np.linalg.norm(x[:,:512],axis=1),1,atol=1e-4)
    assert np.allclose(np.linalg.norm(x[:,512:],axis=1),1,atol=1e-4)
    config=read(paired/'config.json')
    assert config['text_model']=='sentence-transformers/clip-ViT-B-32-multilingual-v1'
    y=np.array([r['label'] for r in rows]);local=np.array([r['training_source']!='robojailbench' for r in rows])
    indices={r['sample_id']:i for i,r in enumerate(rows)}
    splits=read(old/'splits.json')
    protected={str(p):sha(p) for p in (repo/'artifacts/f1t').rglob('*.safetensors')}
    schemes=['jdapt_concat','jdapt_fusion','jdapt_fusion_coral_weighted']
    seeds=[42,7,19]
    out.mkdir(parents=True)
    (out/'upstream').mkdir()
    for rel in ['utils/models.py','utils/helpers.py','embeddings.py']:
        shutil.copy2(upstream/rel,out/'upstream'/Path(rel).name)
    for name in ['jdapt-car_cells.py','jdapt-boat_cells.py','jdapt-robodog_cells.py']:
        shutil.copy2(upstream.parent/name,out/'upstream'/name)
    shutil.copy2(__file__,out/'train_jdapt_local.py')
    shutil.copy2(old/'splits.json',out/'splits.json')
    plan=dict(
        upstream_revision='32a2d1da5eb7feb4099728e42ae6f174ab70ba85',
        upstream_hashes={p.name:sha(p) for p in (out/'upstream').iterdir()},
        sources={'robojailbench':204,'jaewoo_v2':946,'sungmin_v8':184},
        samples=1334, label_mapping={'0':'normal','1':'harmful'}, seeds=seeds,
        encoders_frozen=True, encoder_lock=meta['encoder_lock'],config=config,
        manifest_sha256=meta['manifest_sha256'],features_sha256=meta['features_sha256'],
        splits_sha256=sha(old/'splits.json'),
        fusion=dict(architecture='official CrossAttentionFusion 512, 8 heads, output 512',epochs=20,batch=64,lr=1e-4,optimizer='AdamW',weight_decay=.01,loss='MSE to (text+image)/2',train_scope='all training paired rows only'),
        classifier=dict(architecture='official SimpleNN input -> 512 ReLU -> 2',epochs=10,batch=32,lr=.001,optimizer='Adam',shuffle=False,input_concat=1024,input_full=1536),
        adaptation=dict(source='training RoboJailBench rows',target='normal local training rows only',domain_epochs=10,domain_batch=32,domain_lr=.001,domain_balance='equal total loss mass per domain',weights='p(target|source)/(1-p(target|source)), clipped [0.1,10], mean 1',coral='symmetric covariance whitening/recoloring eps1e-5',classifier_training='aligned source plus original labeled local training rows',inference='unaltered target-space full features; source transform is training-only',missing_target='skip CORAL and weights, exactly use fusion-only model'),
        schemes=schemes,baseline='previous paired_only, same cached embeddings, exact splits and seeds; seed42 one-fold retraining checked',
        threshold=.5, hyperparameter_search=False,validation_labels_used_for_selection=False,
        limitations=['Source-derived local supervised adaptation, not paper reproduction.', 'DAQUAR and JB28K are not used; existing reviewed local and RoboJailBench paired pool is used.', 'SafeAgentBench and SafeText auxiliary tasks are not used in this controlled comparison.', 'Same-scene hazard evaluation shares the scene across training and evaluation.', 'Group track excludes complete scenes/images and includes the unsupported new-scene case.', 'Existing diagnostics have been examined before; not a new independent test.', 'Local labels are agent-reviewed, not fully human-reviewed; some frame provenance remains unverified.'],
        upstream_corrections=['Weighted cross entropy must use reduction=none.', 'Weights must stay attached to sample indices when batches change.', 'Density odds must be evaluated on each source sample, not ratios of unrelated target/source rows.', 'CORAL symmetric roots include eigenvector transposes; the public implementation omits these rotations.'],
        runtime_model_replaced=False,device='cpu',torch_version=torch.__version__,numpy_version=np.__version__,
    )
    write(out/'experiment-plan.json',plan)
    print('PLAN: 1334 pairs; 3 seeds; 5 group folds + 12 hazard folds; fusion20/classifier10; CPU4.',flush=True)

    # Mechanism checks for the modifications, before any evaluation.
    rng=np.random.default_rng(314)
    s=rng.normal(size=(80,6));t=rng.normal(size=(70,6))*np.arange(1,7)+3
    aligned,state=coral_fit(s,t)
    covs=np.cov(s,rowvar=False)+1e-5*np.eye(6);covt=np.cov(t,rowvar=False)+1e-5*np.eye(6)
    a=state['matrix'].numpy()
    assert np.allclose(a.T@covs@a,covt,atol=1e-4)
    identity,_=coral_fit(s,s);assert np.allclose(identity,s,atol=1e-5)
    logits=torch.tensor([[2.,0.],[2.,0.]],requires_grad=True)
    loss=F.cross_entropy(logits,torch.tensor([0,0]),reduction='none')
    (loss*torch.tensor([1.,3.])).mean().backward()
    assert torch.allclose(logits.grad[1],3*logits.grad[0])
    assert metrics(np.array([0,0,1,1]),np.array([.1,.9,.2,.8]))['accuracy_at_0_5']==.5
    mechanism={'coral_covariance_alignment':True,'coral_identity':True,'per_sample_weight_gradient':True,'metric_label_mapping':True}

    def evaluate(m,z):
        m.eval()
        with torch.inference_mode():
            z=torch.as_tensor(z,dtype=torch.float32)
            return torch.cat([m(b).softmax(1)[:,1] for b in z.split(256)]).numpy()

    def fuse(m,xx):
        m.eval()
        with torch.inference_mode():
            return torch.cat([m(b[:,:512],b[:,512:]) for b in torch.as_tensor(xx,dtype=torch.float32).split(256)]).numpy()

    def fit_classifier(z,labels,seed,weights=None):
        seed_all(seed);m=Classifier(z.shape[1],2)
        opt=torch.optim.Adam(m.parameters(),lr=.001)
        tx=torch.as_tensor(z,dtype=torch.float32);ty=torch.as_tensor(labels,dtype=torch.long)
        w=torch.ones(len(z)) if weights is None else torch.as_tensor(weights,dtype=torch.float32)
        history=[]
        for epoch in range(10):
            m.train();total=0.
            for batch in torch.arange(len(z)).split(32):
                opt.zero_grad(set_to_none=True)
                ll=(F.cross_entropy(m(tx[batch]),ty[batch],reduction='none')*w[batch]).mean()
                assert torch.isfinite(ll)
                ll.backward();opt.step();total+=float(ll.detach())*len(batch)
            history.append(total/len(z))
        return m,history

    def importance(zs,zt,seed):
        # Fixed order is shuffled once before fitting; same seed makes it reproducible.
        z=np.concatenate([zs,zt]);dy=np.r_[np.zeros(len(zs),dtype=int),np.ones(len(zt),dtype=int)]
        dw=np.r_[np.full(len(zs),len(z)/(2*len(zs))),np.full(len(zt),len(z)/(2*len(zt)))]
        order=np.random.default_rng(seed).permutation(len(z))
        d,h=fit_classifier(z[order],dy[order],seed,dw[order])
        p=evaluate(d,zs).clip(1e-4,1-1e-4)
        w=np.clip(p/(1-p),.1,10);w/=w.mean()
        return w,d,h

    def fit_all(tr,seed):
        xx=x[tr];yy=y[tr]
        seed_all(seed);f=Fusion(embed_dim=512,num_heads=8,output_dim=512)
        opt=torch.optim.AdamW(f.parameters(),lr=1e-4)
        gen=torch.Generator().manual_seed(seed)
        tx=torch.tensor(xx,dtype=torch.float32);fh=[]
        for epoch in range(20):
            f.train();total=0.
            for b in torch.randperm(len(tr),generator=gen).split(64):
                opt.zero_grad(set_to_none=True)
                pred=f(tx[b,:512],tx[b,512:]);target=(tx[b,:512]+tx[b,512:])*.5
                ll=F.mse_loss(pred,target);assert torch.isfinite(ll)
                ll.backward();opt.step();total+=float(ll.detach())*len(b)
            fh.append(total/len(tr))
        full=np.concatenate([xx,fuse(f,xx)],axis=1)
        c,ch=fit_classifier(xx,yy,seed)
        fm,mh=fit_classifier(full,yy,seed)
        source=np.flatnonzero(~local[tr]);target=np.flatnonzero(local[tr]&(yy==0))
        diag={'fusion_loss':fh,'concat_loss':ch,'fusion_classifier_loss':mh,'source_n':len(source),'target_reference_n':len(target),'local_supervised_n':int(local[tr].sum())}
        adaptation=None;domain=None
        if len(source)>1 and len(target)>1:
            z=full.copy();z[source],adaptation=coral_fit(full[source],full[target])
            w=np.ones(len(tr),dtype=np.float32)
            w[source],domain,dh=importance(full[source],full[target],seed)
            am,ah=fit_classifier(z,yy,seed,w)
            diag.update(adaptation_status='applied',adapted_classifier_loss=ah,domain_loss=dh,source_weights={'min':float(w[source].min()),'max':float(w[source].max()),'mean':float(w[source].mean())})
        else:
            am=fm;diag.update(adaptation_status='skipped_no_local_training_reference')
        return f,{'jdapt_concat':c,'jdapt_fusion':fm,'jdapt_fusion_coral_weighted':am},diag,adaptation,domain

    def met(idx,p):
        m=metrics(y[idx],p)
        m.update(normal_n=int((y[idx]==0).sum()),harmful_n=int((y[idx]==1).sum()),false_positives=int(((y[idx]==0)&(p>=.5)).sum()),false_negatives=int(((y[idx]==1)&(p<.5)).sum()))
        return m

    summary={scheme:{} for scheme in ['existing_mlp']+schemes}
    # Make source and target train membership auditable without repeated inference.
    all_fold_audit=[]
    for seed in seeds:
        seed_dir=out/f'seed-{seed}';seed_dir.mkdir()
        for track,fs in splits.items():
            pp={s:np.full(len(rows),np.nan) for s in schemes}
            fold_reports=[]
            for foldno,fold in enumerate(fs):
                tr=np.array([indices[k] for k in fold['paired_train_ids']]);va=np.array([indices[k] for k in fold['validation_ids']])
                assert set(tr).isdisjoint(va)
                for key in ['sample_id','pair_id']:
                    assert {rows[i][key] for i in tr}.isdisjoint({rows[i][key] for i in va})
                if track=='group':
                    for key in ['image_sha256','group_id']:
                        assert {rows[i][key] for i in tr}.isdisjoint({rows[i][key] for i in va})
                tick=time.monotonic()
                f,models,diag,adapt,domain=fit_all(tr,seed)
                z=np.concatenate([x[va],fuse(f,x[va])],axis=1)
                vals={s:evaluate(models[s],x[va] if s=='jdapt_concat' else z) for s in schemes}
                for s,p in vals.items():pp[s][va]=p
                fold_reports.append(dict(held=fold['held'],train_n=len(tr),validation_n=len(va),diagnostics=diag,metrics={s:met(va,p) for s,p in vals.items()}))
                all_fold_audit.append(dict(seed=seed,track=track,held=fold['held'],fusion_train_ids=fold['paired_train_ids'],adaptation_reference_ids=[rows[i]['sample_id'] for i in tr if local[i] and y[i]==0],validation_ids=fold['validation_ids']))
                if seed==42 and track=='group' and foldno==0:
                    baseline,_=train_model(x[tr],y[tr],dict(config,seed=42),'cpu')
                    oldpred={r['sample_id']:r['score'] for r in read(old/'paired_only/seed-42/group-predictions.json')}
                    assert np.allclose(scores(baseline,x[va],'cpu'),[oldpred[rows[i]['sample_id']] for i in va],atol=1e-7)
                    mechanism['old_baseline_one_fold_reproduced']=True
                print(f'SEED {seed} {track} {foldno+1}/{len(fs)} held={fold["held"]} n={len(va)} coral={diag["adaptation_status"]} seconds={time.monotonic()-tick:.1f}',flush=True)
                # Full fold checkpoints allow prediction replay, at modest disk cost.
                fd=seed_dir/(track+'-'+str(foldno));fd.mkdir()
                save_file({k:v.contiguous() for k,v in f.state_dict().items()},str(fd/'fusion.safetensors'))
                for s,m in models.items():save_file({k:v.contiguous() for k,v in m.state_dict().items()},str(fd/(s+'.safetensors')))
                write(fd/'fold.json',fold_reports[-1])
                del f,models,adapt,domain
            idx=np.arange(len(rows)) if track=='group' else np.flatnonzero(local)
            oldpred={r['sample_id']:r['score'] for r in read(old/'paired_only'/f'seed-{seed}'/(track+'-predictions.json'))}
            base=np.full(len(rows),np.nan)
            for i in idx:base[i]=oldpred[rows[i]['sample_id']]
            pp['existing_mlp']=base
            cohorts={'rjb_group_diagnostic':np.flatnonzero(~local),'local_scene_held_out':np.flatnonzero(local)} if track=='group' else {'local_same_scene_hazard':np.flatnonzero(local)}
            for s,p in pp.items():
                assert np.isfinite(p[idx]).all()
                write(seed_dir/(track+'-'+s+'-predictions.json'),[dict(sample_id=rows[i]['sample_id'],label=int(y[i]),score=float(p[i])) for i in idx])
                for cohort,ii in cohorts.items():summary[s].setdefault(cohort,{'per_seed':{}})['per_seed'][str(seed)]=met(ii,p[ii])
            write(seed_dir/(track+'-report.json'),fold_reports)
            print('TRACK_COMPLETE '+json.dumps({s:{c:summary[s][c]['per_seed'][str(seed)] for c in cohorts} for s in pp}),flush=True)

        # Save a local-environment candidate using all paired rows; not selected by evaluation scores.
        tr=np.arange(len(rows))
        f,models,diag,adapt,domain=fit_all(tr,seed)
        fd=seed_dir/'full-fit';fd.mkdir()
        save_file({k:v.contiguous() for k,v in f.state_dict().items()},str(fd/'fusion.safetensors'))
        if adapt is not None:save_file(adapt,str(fd/'coral-training-transform.safetensors'))
        if domain is not None:save_file({k:v.contiguous() for k,v in domain.state_dict().items()},str(fd/'domain-classifier.safetensors'))
        restored_f=Fusion();restored_f.load_state_dict(load_file(str(fd/'fusion.safetensors')))
        z=np.concatenate([x,fuse(f,x)],axis=1)
        rz=np.concatenate([x,fuse(restored_f,x)],axis=1)
        assert np.array_equal(z,rz)
        fit_metrics={}
        for s,m in models.items():
            save_file({k:v.contiguous() for k,v in m.state_dict().items()},str(fd/(s+'.safetensors')))
            restored=Classifier(1024 if s=='jdapt_concat' else 1536,2)
            restored.load_state_dict(load_file(str(fd/(s+'.safetensors'))))
            p=evaluate(m,x if s=='jdapt_concat' else z)
            assert np.array_equal(p,evaluate(restored,x if s=='jdapt_concat' else rz))
            fit_metrics[s]=met(tr,p)
        write(fd/'report.json',dict(training_only_metrics=fit_metrics,diagnostics=diag,reload_verified=True,inference='frozen multilingual CLIP -> text,image,fused concatenation -> classifier; do not CORAL-transform target inputs',deployment_ready=False))
        print(f'FULL_FIT_SAVED seed={seed}',flush=True)
    for s,cohorts in summary.items():
        for c,r in cohorts.items():
            mm=list(r['per_seed'].values());r['n_per_seed']=mm[0]['n']
            keys=['accuracy_at_0_5','harmful_miss_rate_at_0_5','normal_false_block_rate_at_0_5','auroc']
            r['mean']={k:float(np.mean([v[k] for v in mm])) for k in keys}
            r['std_population']={k:float(np.std([v[k] for v in mm])) for k in keys}
    assert all(sha(p)==h for p,h in protected.items())
    write(out/'prior-checkpoints.json',dict(unchanged=True,sha256=protected))
    write(out/'split-audit.json',all_fold_audit)
    write(out/'mechanism-verification.json',mechanism)
    write(out/'summary.json',dict(comparison=summary,elapsed_seconds=time.monotonic()-start,plan_sha256=sha(out/'experiment-plan.json'),prior_checkpoints_unchanged=True,runtime_model_replaced=False,independent_test=False,full_fit_reload_verified=True))
    print('COMPLETE '+json.dumps({s:{c:r['mean'] for c,r in v.items()} for s,v in summary.items()}),flush=True)


if __name__=='__main__':main()
