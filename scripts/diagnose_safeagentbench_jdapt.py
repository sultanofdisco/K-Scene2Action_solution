import json
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from safetensors.torch import load_file
from sklearn.linear_model import LogisticRegression
from f1.training.data import digest
from f1.training.learner import create_mlp, train_model, scores, metrics
from f1.training.text_aux import load_aux

root=Path.cwd()
base=root/'artifacts/f1t/text-aux-safeagentbench-v1'
out=base/'diagnosis-jdapt-001'
out.mkdir(exist_ok=False)
report=json.loads((base/'fit-001/report.json').read_text(encoding='utf8'))
meta=json.loads((base/'cache/metadata.json').read_text(encoding='utf8'))
rows,hashes=load_aux(root/'data/safeagentbench_f1t_english_v1/splits_v1')
assert rows==meta['rows'] and hashes==report['manifest_hashes']
assert digest(base/'cache/features.npy')==report['features_sha256']
x=np.load(base/'cache/features.npy',allow_pickle=False)
tr=np.array([r['split']=='train' for r in rows]); va=np.array([r['split']=='validation' for r in rows])
xt=x[tr]; xv=x[va]; y=np.array([r['label'] for r in rows]); yt=y[tr]; yv=y[va]
rt=[r for r in rows if r['split']=='train']; rv=[r for r in rows if r['split']=='validation']
torch.set_num_threads(4)
config=report['config']
baseline=create_mlp(512,config)
baseline.load_state_dict(load_file(str(base/'fit-001/text-aux.safetensors')))
pt=scores(baseline,xt,'cpu'); pv=scores(baseline,xv,'cpu')
def measure(y,p):
 m=metrics(y,p)
 m['balanced_accuracy']=1-(m['harmful_miss_rate_at_0_5']+m['normal_false_block_rate_at_0_5'])/2
 return m
results=[{'model':'mlp','epochs':30,'seed':42,'train':measure(yt,pt),'validation':measure(yv,pv)}]
neighbors=[]
for i,r in enumerate(rv):
 if (pv[i]>=.5)==r['label']:continue
 sim=xt@xv[i]
 nearest=np.argsort(-sim)[:5]
 neighbors.append({'id':r['sample_id'],'text':r['text'],'label':r['label'],'score':float(pv[i]),
  'nearest_train':[{'id':rt[j]['sample_id'],'text':rt[j]['text'],'label':int(yt[j]),'similarity':float(sim[j]),'score':float(pt[j])} for j in nearest]})
for seed,epochs in [(42,300),(7,30),(7,300)]:
 cfg=dict(config,seed=seed,epochs=epochs)
 model,history=train_model(xt,yt,cfg,'cpu')
 p=scores(model,xv,'cpu')
 results.append({'model':'mlp','epochs':epochs,'seed':seed,'train':measure(yt,scores(model,xt,'cpu')),
                 'validation':measure(yv,p),'validation_scores':p.tolist()})
linear=LogisticRegression(C=1.0,max_iter=2000,random_state=42).fit(xt,yt)
results.append({'model':'logistic_regression_C1','train':measure(yt,linear.predict_proba(xt)[:,1]),
                'validation':measure(yv,linear.predict_proba(xv)[:,1])})
all_results={'purpose':'diagnosis_only_not_independent_model_selection','test_evaluated':False,
 'deployed_checkpoint_changed':False,'experiments':results,'error_neighbors':neighbors,
 'training_label_counts':dict(Counter(int(z) for z in yt))}
(out/'diagnostics.json').write_text(json.dumps(all_results,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(all_results,ensure_ascii=False,indent=2))
