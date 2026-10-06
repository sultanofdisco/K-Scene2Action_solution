"""Within-family control using original train only; keep duplicate advice together."""
from collections import defaultdict
import json
from pathlib import Path
import random
import numpy as np
import torch
from f1.training.data import write_json
from f1.training.learner import train_model, scores, metrics

root=Path("artifacts/f1t/text-aux-public-v1"); out=root/"diagnosis-001"
meta=json.loads((root/"cache/metadata.json").read_text(encoding="utf8"))
config=json.loads((root/"fit-001/report.json").read_text(encoding="utf8"))["config"]
raw=np.load(root/"cache/features.npy",allow_pickle=False)
ids=[i for i,r in enumerate(meta["rows"]) if r["split"]=="train"]
rows=[meta["rows"][i] for i in ids]; x=raw[ids].copy(); del raw
y=np.array([r["label"] for r in rows]); torch.set_num_threads(4)
parents={r["scenario_id"]:r["scenario_id"] for r in rows}
def find(k):
    while parents[k]!=k:
        parents[k]=parents[parents[k]]; k=parents[k]
    return k
seen={}
for r in rows:
    advice=" ".join(r["proposed_action"].casefold().split())
    if advice in seen: parents[find(r["scenario_id"])]=find(seen[advice])
    seen[advice]=r["scenario_id"]
families=defaultdict(set)
for r in rows: families[r["hazard_family"]].add(find(r["scenario_id"]))
results=[]
for seed in (42,7,19):
    rng=random.Random(seed)
    held={rng.choice(sorted(groups)) for family,groups in sorted(families.items()) if len(groups)>=2}
    va=np.array([find(r["scenario_id"]) in held for r in rows]); tr=~va
    advices=lambda mask:{" ".join(r["proposed_action"].casefold().split()) for r,k in zip(rows,mask) if k}
    assert not advices(tr)&advices(va)
    assert {r["hazard_family"] for r,k in zip(rows,va) if k}<={r["hazard_family"] for r,k in zip(rows,tr) if k}
    for epochs in (30,300):
        model,_=train_model(x[tr],y[tr],dict(config,seed=seed,epochs=epochs),"cpu")
        result={"seed":seed,"epochs":epochs,"train_n":int(tr.sum()),"validation_n":int(va.sum()),
                "duplicate_advice_crosses_split":False,
                "held_scenarios":sorted({r["scenario_id"] for r,k in zip(rows,va) if k}),
                "train":metrics(y[tr],scores(model,x[tr],"cpu")),
                "validation":metrics(y[va],scores(model,x[va],"cpu"))}
        results.append(result);print(json.dumps(result),flush=True)
report=json.loads((out/"diagnostics.json").read_text(encoding="utf8"))
if "within_known_family_controls" in report:
    report["superseded_within_family_control_without_advice_dedup"]=report.pop("within_known_family_controls")
report["within_known_family_controls_deduplicated"]=results
report["within_known_family_note"]="Original train only; intact scenario pairs and repeated advice held together. Different evaluation targets and sample counts: not a replacement for original validation."
write_json(out/"diagnostics.json",report)
