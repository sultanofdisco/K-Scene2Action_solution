"""Fixed-epoch diagnostic cross validation and research-only MLP checkpoints."""
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn

from .data import make_folds, read_config, write_json, digest
from .encoder import load_cache

def create_mlp(input_dimension,config):
    return nn.Sequential(nn.Linear(input_dimension,config["hidden_dimension"]),nn.ReLU(),
                         nn.Dropout(config["dropout"]),nn.Linear(config["hidden_dimension"],1))

def select_features(x,kind):
    if kind=="both":return x
    if kind=="text":return x[:,:512]
    if kind=="image":return x[:,512:]
    raise ValueError("Unknown feature kind")

def train_model(x,y,config,device="cpu"):
    seed=config["seed"];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    model=create_mlp(x.shape[1],config).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config["learning_rate"],weight_decay=config["weight_decay"])
    loss_fn=nn.BCEWithLogitsLoss()
    inputs=torch.tensor(x,dtype=torch.float32);targets=torch.tensor(y,dtype=torch.float32)
    generator=torch.Generator().manual_seed(seed);history=[]
    for _ in range(config["epochs"]):
        model.train();total=0.0
        for batch in torch.randperm(len(x),generator=generator).split(config["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            loss=loss_fn(model(inputs[batch].to(device)).squeeze(-1),targets[batch].to(device))
            if not torch.isfinite(loss):raise ValueError("Nonfinite training loss")
            loss.backward();optimizer.step();total+=float(loss.detach().cpu())*len(batch)
        history.append(total/len(x))
    return model,history

def scores(model,x,device):
    model.eval()
    with torch.inference_mode():
        return model(torch.tensor(x,dtype=torch.float32,device=device)).squeeze(-1).sigmoid().cpu().numpy()

def metrics(y,p):
    from sklearn.metrics import roc_auc_score
    y=np.asarray(y);p=np.asarray(p);pred=p>=0.5
    if not np.isfinite(p).all():raise ValueError("Nonfinite scores")
    return {"n":len(y),"accuracy_at_0_5":float(np.mean(pred==y)),
            "harmful_miss_rate_at_0_5":float(np.mean(~pred[y==1])) if (y==1).any() else None,
            "normal_false_block_rate_at_0_5":float(np.mean(pred[y==0])) if (y==0).any() else None,
            "auroc":float(roc_auc_score(y,p)) if len(set(y))==2 else None}

def run(cache,config_path,output,mode="cv",features="both",device="cpu"):
    x,meta=load_cache(cache);config=read_config(config_path)
    if device=="cpu":torch.set_num_threads(config["cpu_threads"])
    for key in ["text_model","image_model","text_revision","image_revision","preprocessing","embedding_dimension","max_tokens"]:
        if config[key]!=meta["config"][key]:raise ValueError(f"Cache/config mismatch: {key}")
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    x=select_features(x,features);rows=meta["rows"];y=np.array([r["label"] for r in rows])
    result={"mode":mode,"features":features,"deployment_ready":False,
            "threshold_status":"UNCALIBRATED_0_5_DIAGNOSTIC_ONLY", "config":config,
            "manifest_sha256":meta["manifest_sha256"],"encoder_lock":meta["encoder_lock"]}
    result["features_sha256"]=meta["features_sha256"]
    result["runtime"]={"device":device,"torch":torch.__version__,"numpy":np.__version__}
    if mode=="cv":
        indices={r["sample_id"]:i for i,r in enumerate(rows)};fold_results=[];oof=np.full(len(rows),np.nan)
        for fold in make_folds(rows):
            tr=[indices[k] for k in fold["train_ids"]];va=[indices[k] for k in fold["validation_ids"]]
            model,history=train_model(x[tr],y[tr],config,device);p=scores(model,x[va],device);oof[va]=p
            fold_results.append({"held_group":fold["held_group"],"train_n":len(tr),"metrics":metrics(y[va],p),"train_loss":history})
        result.update(folds=fold_results,pooled_diagnostic_metrics=metrics(y,oof),
                      macro_fold_metrics={k:float(np.mean([f["metrics"][k] for f in fold_results])) for k in ["accuracy_at_0_5","harmful_miss_rate_at_0_5","normal_false_block_rate_at_0_5","auroc"]},
                      independent_test=False)
        write_json(out/"predictions.json",[{"sample_id":r["sample_id"],"group_id":r["group_id"],"label":r["label"],"uncalibrated_score":float(p)} for r,p in zip(rows,oof)])
    elif mode=="fit":
        from safetensors.torch import save_file
        model,history=train_model(x,y,config,device)
        save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(out/"model.safetensors"))
        result.update(input_dimension=x.shape[1],train_samples=len(rows),train_loss=history,
                      checkpoint_sha256=digest(out/"model.safetensors"),operational_thresholds=None)
    else:raise ValueError("Unknown training mode")
    write_json(out/"report.json",result)
    return result
