"""Bounded train/validation diagnostics. Never fits or scores the held-out test set."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from sklearn.linear_model import LogisticRegression
from f1.training.data import digest, write_json
from f1.training.learner import create_mlp, train_model, scores, metrics

ROOT = Path("artifacts/f1t/text-aux-public-v1")
OUT = ROOT/"diagnosis-001"
OUT.mkdir(parents=True, exist_ok=False)
meta = json.loads((ROOT/"cache/metadata.json").read_text(encoding="utf8"))
original = json.loads((ROOT/"fit-001/report.json").read_text(encoding="utf8"))
assert digest(ROOT/"cache/features.npy") == meta["features_sha256"]
assert digest(ROOT/"fit-001/text-aux.safetensors") == original["checkpoint_sha256"]
for split, expected in meta["manifest_hashes"].items():
    assert digest(Path("data/public_f1_bundle_v1/text_aux")/f"{split}.jsonl") == expected
all_x = np.load(ROOT/"cache/features.npy", allow_pickle=False)
keep = [i for i, r in enumerate(meta["rows"]) if r["split"] in {"train", "validation"}]
rows = [meta["rows"][i] for i in keep]
x = all_x[keep].copy()
del all_x
y = np.array([r["label"] for r in rows])
tr = np.array([r["split"] == "train" for r in rows]); va = ~tr
torch.set_num_threads(4)
config = original["config"]
baseline = create_mlp(512, config)
baseline.load_state_dict(load_file(str(ROOT/"fit-001/text-aux.safetensors")), strict=True)
p = scores(baseline, x, "cpu")


def pair_results(part_rows, features, pred):
    pairs = defaultdict(list)
    for i, r in enumerate(part_rows): pairs[r["scenario_id"]].append(i)
    result = []
    for ids in pairs.values():
        safe = next(i for i in ids if part_rows[i]["label"] == 0)
        unsafe = next(i for i in ids if part_rows[i]["label"] == 1)
        result.append({"scenario": part_rows[safe]["scenario"], "scenario_id": part_rows[safe]["scenario_id"],
                       "family": part_rows[safe]["hazard_family"], "safe_action": part_rows[safe]["proposed_action"],
                       "unsafe_action": part_rows[unsafe]["proposed_action"],
                       "safe_score": float(pred[safe]), "unsafe_score": float(pred[unsafe]),
                       "unsafe_minus_safe": float(pred[unsafe]-pred[safe]),
                       "cosine_similarity": float(features[safe]@features[unsafe]),
                       "same_predicted_label": bool((pred[safe]>=.5)==(pred[unsafe]>=.5))})
    return result


report = {"test_used": False, "baseline_checkpoint_sha256": original["checkpoint_sha256"],
          "train_families": dict(Counter(r["hazard_family"] for r in rows if r["split"] == "train")),
          "validation_families": dict(Counter(r["hazard_family"] for r in rows if r["split"] == "validation")),
          "baseline": {"train": metrics(y[tr],p[tr]), "validation": metrics(y[va],p[va])},
          "train_pairs": pair_results([r for r in rows if r["split"] == "train"], x[tr], p[tr]),
          "validation_pairs": pair_results([r for r in rows if r["split"] == "validation"], x[va], p[va])}
thresholds = [0.0, 1.0] + [(a+b)/2 for a,b in zip(sorted(set(p[va]))[:-1],sorted(set(p[va]))[1:])]
best = max(thresholds, key=lambda t: np.mean((p[va]>=t)==y[va]))
report["validation_oracle_threshold"] = {"threshold": float(best), "accuracy": float(np.mean((p[va]>=best)==y[va])),
                                          "diagnostic_only_optimistic_not_an_evaluation": True}
experiments = []
for epochs in (30, 100, 300):
    for seed in (42, 7, 19):
        cfg = dict(config, epochs=epochs, seed=seed)
        model, history = train_model(x[tr], y[tr], cfg, "cpu")
        result = {"epochs": epochs, "seed": seed, "train": metrics(y[tr],scores(model,x[tr],"cpu")),
                  "validation": metrics(y[va],scores(model,x[va],"cpu")), "final_train_loss": history[-1]}
        experiments.append(result)
        print(json.dumps(result), flush=True)
report["epoch_seed_controls"] = experiments
linear = []
for c in (1., 100.):
    model = LogisticRegression(C=c, max_iter=2000, random_state=42).fit(x[tr],y[tr])
    linear.append({"C":c,"train":metrics(y[tr],model.predict_proba(x[tr])[:,1]),
                   "validation":metrics(y[va],model.predict_proba(x[va])[:,1])})
report["logistic_controls"] = linear
write_json(OUT/"diagnostics.json",report)
print(json.dumps({k:v for k,v in report.items() if k not in {"train_pairs","epoch_seed_controls"}}, ensure_ascii=False,indent=2), flush=True)

# Fixed encoder input ablation: context + action versus action alone.
# This is diagnosis on already-examined validation, not model selection or final testing.
from sentence_transformers import SentenceTransformer
from f1.training.encoder import require_token_budget
encoder=SentenceTransformer(meta["text_model"],revision=meta["revision"],device="cpu",local_files_only=True)
encoder.eval()
for param in encoder.parameters(): param.requires_grad_(False)
inputs=[r["proposed_action"] for r in rows]
require_token_budget(encoder.tokenizer,inputs,128)
with torch.inference_mode():
    action_x=encoder.encode(inputs,batch_size=32,normalize_embeddings=True,convert_to_numpy=True)
ablation=[]
for seed in (42,7,19):
    cfg=dict(config,seed=seed)
    model,_=train_model(action_x[tr],y[tr],cfg,"cpu")
    ablation.append({"seed":seed,"train":metrics(y[tr],scores(model,action_x[tr],"cpu")),
                     "validation":metrics(y[va],scores(model,action_x[va],"cpu"))})
report["action_only_30_epoch_controls"]=ablation
report["action_only_validation_pairs"]=pair_results([r for r in rows if r["split"]=="validation"],action_x[va],scores(model,action_x[va],"cpu"))
write_json(OUT/"diagnostics.json",report)
assert digest(ROOT/"fit-001/text-aux.safetensors")==original["checkpoint_sha256"]
print('ACTION_ONLY',json.dumps(ablation),flush=True)
