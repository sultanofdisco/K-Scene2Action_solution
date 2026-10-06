"""Cache frozen encoder outputs before the trainable native decision layers."""
import json,os,runpy,time
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import torch
from safetensors.torch import save_file
from f1.laya.backend import LayaBackend,check_budget
from f1.laya.pipeline import make_state
from f1.laya.questions_v2 import QUESTIONS
from f1.laya.finetuned import lock_hash

ROOT=Path('artifacts/laya/ripa-decision-cpu-v1');DATA=Path('data/laya_ripa_domain_v1')
def main():
    ROOT.mkdir(exist_ok=True);cache=ROOT/'encoder-cache';cache.mkdir(exist_ok=True)
    h=runpy.run_path('scripts/train_laya_scorer.py');read,dump,sha=h['read'],h['dump'],h['sha'];helper=runpy.run_path('scripts/compare_laya_prompts.py')
    lock=json.loads((DATA/'data-lock.json').read_text());assert lock['questions']==QUESTIONS
    for p,v in lock['protected_files'].items():assert sha(Path(p))==v
    backend=LayaBackend(cpu_threads=4);backend._load();model=backend.agent.model;model.eval()
    for p in model.parameters():p.requires_grad_(False)
    manifest={'data_lock_sha256':sha(DATA/'data-lock.json'),'base_model_lock_sha256':lock_hash(backend.metadata['model_lock']),'splits':{},'dtype':'float32','before_trainable_head':True}
    old=Path('artifacts/laya/deepset-decision-cpu-v1');old_lock=json.loads((old/'decision-lock.json').read_text());assert old_lock['base_model_lock_sha256']==manifest['base_model_lock_sha256']
    for name,split in [('legacy_train','train'),('legacy_validation','validation')]:
        fp=old/(split+'-encoder-features.safetensors');ip=old/(split+'-encoder-index.json')
        manifest['splits'][name]={'shards':[{'features':str(fp),'index':str(ip),'features_sha256':sha(fp),'index_sha256':sha(ip)}],'cases':58 if split=='train' else 22,'reused_original_encoder_cache':True}
    capture={}
    def pre(m,a):capture.update(mask=a[1].detach().clone(),markers=a[2].detach().clone(),qtype=a[4].detach().clone())
    def enc(m,a,o):capture['h']=o.last_hidden_state.detach().clone()
    hooks=[model.register_forward_pre_hook(pre),model.encoder.register_forward_hook(enc)];started=time.perf_counter()
    for split in ['ripa_train','ripa_validation']:
        cases=read(DATA/(split+'.jsonl'));assert sha(DATA/(split+'.jsonl'))==lock['sha256'][split];shards=[];budgets=[]
        for offset in range(0,len(cases),20):
            fp=cache/f'{split}-{offset:04d}.safetensors';ip=cache/f'{split}-{offset:04d}.json';items={};indices=[]
            # Resume only fully committed shards whose local hashes have been recorded.
            record=cache/f'{split}-{offset:04d}-lock.json'
            if record.exists():
                shard=json.loads(record.read_text());assert sha(fp)==shard['features_sha256'] and sha(ip)==shard['index_sha256'];shards.append(shard);continue
            for start in range(offset,min(offset+20,len(cases)),2):
                chunk=cases[start:min(start+2,offset+20)];states=[]
                for c in chunk:
                    req,ctx,binding,ocr=helper['inputs'](c);state=make_state(req,ctx,ocr);budgets.append({'case_id':c['case_id'],'budget':check_budget(backend.agent.tok,state,QUESTIONS)});states.append(state)
                capture.clear();result=backend.agent.predict_batch(states,QUESTIONS,batch_size=2,max_len=1024,head_max_len=256,sort_by_length=False)
                assert capture['h'].shape[0]==len(chunk)*4
                for i,c in enumerate(chunk):
                    assert result[i]['usage']['truncated'] is False
                    for q,key in enumerate(QUESTIONS):
                        j=i*4+q;length=int(capture['mask'][j].sum());tname=f'h{len(indices)}'
                        items[tname]=capture['h'][j,:length].contiguous().clone()
                        indices.append({'tensor':tname,'case_id':c['case_id'],'q':q,'qtype':int(capture['qtype'][j]),'markers':capture['markers'][j].tolist(),'label':['YES','NO','UNKNOWN'].index(c['expected_questions'][key])})
            save_file(items,str(fp));dump(ip,indices);shard={'features':str(fp),'index':str(ip),'features_sha256':sha(fp),'index_sha256':sha(ip)};dump(record,shard);shards.append(shard)
            print(json.dumps({'stage':'encoder_cache','split':split,'done':min(offset+20,len(cases)),'total':len(cases),'seconds':time.perf_counter()-started}),flush=True)
        manifest['splits'][split]={'cases':len(cases),'shards':shards,'all_untruncated':True};dump(cache/(split+'-budgets.json'),budgets);dump(ROOT/'cache-manifest.json',manifest)
    for hook in hooks:hook.remove()
    manifest['completed']=True;manifest['seconds']=time.perf_counter()-started;dump(ROOT/'cache-manifest.json',manifest);print(json.dumps({'cache_completed':True,'seconds':manifest['seconds']}),flush=True)
if __name__=='__main__':main()
