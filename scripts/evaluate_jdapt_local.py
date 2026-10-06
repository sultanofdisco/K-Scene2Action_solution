"""Replay a saved local J-DAPT classifier on a verified F1-T embedding cache."""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo',type=Path,required=True)
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--checkpoint-dir',type=Path,required=True)
    ap.add_argument('--cache',type=Path,required=True)
    ap.add_argument('--scheme',choices=['jdapt_concat','jdapt_fusion','jdapt_fusion_coral_weighted'],default='jdapt_fusion_coral_weighted')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    sys.path.insert(0,str(args.repo/'src'))
    import numpy as np
    import torch
    from safetensors.torch import load_file
    from f1.training.encoder import load_cache
    from f1.training.learner import metrics
    torch.set_num_threads(4)
    if args.output.exists():raise RuntimeError('Output exists; choose a new output file')
    plan=json.loads((args.run/'experiment-plan.json').read_text(encoding='utf-8'))
    code=args.run/'upstream/models.py'
    assert hashlib.sha256(code.read_bytes()).hexdigest()==plan['upstream_hashes']['models.py']
    spec=importlib.util.spec_from_file_location('jdapt_models_pinned',code)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    x,meta=load_cache(args.cache)
    assert meta['encoder_lock']==plan['encoder_lock'],'Encoder versions differ'
    for k in ['text_model','image_model','text_revision','image_revision','preprocessing','embedding_dimension','max_tokens']:
        assert meta['config'][k]==plan['config'][k],f'Cache config differs: {k}'
    assert x.shape[1]==1024 and np.isfinite(x).all()
    classifier=module.SimpleNN(1024 if args.scheme=='jdapt_concat' else 1536,2)
    classifier.load_state_dict(load_file(str(args.checkpoint_dir/(args.scheme+'.safetensors'))));classifier.eval()
    fusion=None
    if args.scheme!='jdapt_concat':
        fusion=module.CrossAttentionFusion()
        fusion.load_state_dict(load_file(str(args.checkpoint_dir/'fusion.safetensors')));fusion.eval()
    parts=[]
    with torch.inference_mode():
        for batch in torch.tensor(x,dtype=torch.float32).split(256):
            full=batch if fusion is None else torch.cat([batch,fusion(batch[:,:512],batch[:,512:])],dim=1)
            parts.append(classifier(full).softmax(1)[:,1])
    p=torch.cat(parts).numpy();y=np.array([r['label'] for r in meta['rows']])
    result=dict(scheme=args.scheme,checkpoint_dir=str(args.checkpoint_dir),cache=str(args.cache),metrics=metrics(y,p),
                independent_test=False,warning='Replay only. Cache may include training data; reported metrics are not an independent evaluation.',
                predictions=[dict(sample_id=r['sample_id'],label=r['label'],score=float(v),prediction=int(v>=.5)) for r,v in zip(meta['rows'],p)])
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result['metrics']))


if __name__=='__main__':main()
