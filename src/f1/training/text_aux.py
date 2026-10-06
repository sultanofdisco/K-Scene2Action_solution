"""Separate reviewed text auxiliary experiments. Never fabricate an image feature."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from .data import digest, read_config, write_json


def load_aux(directory):
    directory = Path(directory)
    rows = []
    hashes = {}
    for split in ("train", "validation", "test"):
        path = directory/f"{split}.jsonl"
        part = [json.loads(s) for s in path.read_text(encoding="utf8").splitlines() if s.strip()]
        if not part or any(r["split"] != split for r in part):
            raise ValueError("Missing or mislabeled auxiliary split")
        hashes[split] = digest(path)
        rows.extend(part)
    validate_rows(rows)
    local_summary=directory/"summary.json"
    if local_summary.exists():
        summary=json.loads(local_summary.read_text(encoding="utf8"))
        expected=summary.get("manifest_hashes")
        if expected is not None and hashes != expected:
            raise ValueError("Auxiliary split manifest checksum mismatch")
    return rows, hashes


def validate_rows(rows):
    if rows and any(r.get("schema_version") == "f1t-text-aux-sab-1" for r in rows):
        from .safeagent_aux import validate_sab_rows
        validate_sab_rows(rows)
        return
    from hashlib import sha256
    ids = set(); links = defaultdict(set); scenarios = defaultdict(list)
    for r in rows:
        if r["schema_version"] != "f1t-text-aux-1" or r["modality"] != "text_only" or r["image_path"] is not None:
            raise ValueError("Not a text-only auxiliary manifest")
        if r["source"] != "sharonlevy/SafeText" or r["usage"] != "auxiliary_text_experiment_only":
            raise ValueError("Evaluation or unsupported source cannot enter auxiliary training")
        if r["review_status"] != "SELECTED_TEXT_AUX_AGENT_REVIEWED" or r["language"] != "en":
            raise ValueError("Unreviewed or non-English auxiliary row")
        if type(r["label"]) is not int or r["label"] not in (0, 1) or r["source_label"] != ("unsafe" if r["label"] else "safe"):
            raise ValueError("Invalid auxiliary label")
        expected = f"Scenario: {r['scenario']}\nProposed action: {r['proposed_action']}"
        if r["text"] != expected or sha256(r["text"].encode()).hexdigest() != r["text_sha256"]:
            raise ValueError("Auxiliary text changed")
        if r["sample_id"] in ids or r["split"] not in {"train", "validation", "test"}:
            raise ValueError("Duplicate ID or invalid split")
        ids.add(r["sample_id"])
        for key in [("group", r["group_id"]), ("family", r["hazard_family"]),
                    ("scenario", r["scenario_id"]), ("text", " ".join(r["text"].casefold().split())),
                    ("advice", " ".join(r["proposed_action"].casefold().split()))]:
            links[key].add(r["split"])
        scenarios[r["scenario_id"]].append(r)
    if any(len(v) != 1 for v in links.values()):
        raise ValueError("Auxiliary split leakage")
    if any(len(v) != 2 or {r["label"] for r in v} != {0, 1} for v in scenarios.values()):
        raise ValueError("Incomplete reviewed scenario contrast")
    for split in ("train", "validation", "test"):
        if {r["label"] for r in rows if r["split"] == split} != {0, 1}:
            raise ValueError("Each split needs both labels")


def cache(data, output):
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    from .encoder import require_token_budget
    rows, hashes = load_aux(data)
    summary_path=Path(data)/"summary.json"
    if not summary_path.exists(): summary_path=Path(data).parent/"summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf8"))
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    model = SentenceTransformer(summary["tokenizer_model"], revision=summary["tokenizer_revision"],
                                device="cpu", trust_remote_code=False, local_files_only=True)
    model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    lengths = require_token_budget(model.tokenizer, [r["text"] for r in rows], 128)
    with torch.inference_mode():
        features = model.encode([r["text"] for r in rows], batch_size=32, normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=True).astype("float32")
    if features.shape != (len(rows), 512) or not np.isfinite(features).all():
        raise ValueError("Invalid auxiliary embeddings")
    np.save(output/"features.npy", features, allow_pickle=False)
    from importlib.metadata import version
    metadata = {"format": "f1t-text-aux-cache-1", "rows": rows, "input_dimension": 512,
                "manifest_hashes": hashes, "features_sha256": digest(output/"features.npy"),
                "text_model": summary["tokenizer_model"], "revision": summary["tokenizer_revision"],
                "max_tokens": max(lengths), "complete": True, "training_run": False,
                "packages": {k: version(k) for k in ["torch", "numpy", "sentence-transformers", "transformers"]}}
    metadata["data_protocol"]={k:summary[k] for k in ("split_policy","track","rare_categories_train_only") if k in summary}
    write_json(output/"metadata.json", metadata)
    return {"samples": len(rows), "dimensions": 512, "max_tokens": max(lengths), "training_run": False}


def fit(cache_dir, output, config_path):
    import numpy as np
    import torch
    from safetensors.torch import save_file
    from .learner import train_model, scores, metrics
    cache_dir = Path(cache_dir)
    meta = json.loads((cache_dir/"metadata.json").read_text(encoding="utf8"))
    if meta["format"] != "f1t-text-aux-cache-1" or meta["complete"] is not True:
        raise ValueError("Wrong/incomplete cache")
    validate_rows(meta["rows"])
    if digest(cache_dir/"features.npy") != meta["features_sha256"]:
        raise ValueError("Cache checksum mismatch")
    x = np.load(cache_dir/"features.npy", allow_pickle=False)
    rows = meta["rows"]
    if x.shape != (len(rows), 512) or not np.isfinite(x).all() or not np.allclose(np.linalg.norm(x, axis=1), 1, atol=1e-4):
        raise ValueError("Invalid text auxiliary features")
    config = read_config(config_path)
    if config["text_model"] != meta["text_model"]:
        raise ValueError("Text model mismatch")
    torch.set_num_threads(config["cpu_threads"])
    y = np.array([r["label"] for r in rows])
    tr = np.array([r["split"] == "train" for r in rows])
    va = np.array([r["split"] == "validation" for r in rows])
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    model, history = train_model(x[tr], y[tr], config, "cpu")
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(output/"text-aux.safetensors"))
    validation_scores=scores(model,x[va],"cpu")
    validation_metrics=metrics(y[va],validation_scores)
    validation_metrics["balanced_accuracy_at_0_5"]=1-(validation_metrics["harmful_miss_rate_at_0_5"]+validation_metrics["normal_false_block_rate_at_0_5"])/2
    report = {"mode": "text_auxiliary_only", "input_dimension": 512, "train_n": int(tr.sum()),
              "validation_n": int(va.sum()), "validation_metrics": validation_metrics,
              "validation_baselines":{"always_safe_accuracy":float(np.mean(y[va]==0)),
                                      "always_unsafe_accuracy":float(np.mean(y[va]==1)),
                                      "balanced_accuracy_constant_predictor":0.5},
              "test_evaluated": False, "deployment_ready": False, "compatible_with_current_1024_mlp": False,
              "threshold_status": "UNCALIBRATED_0_5_DIAGNOSTIC_ONLY", "train_loss": history,
              "config": config, "text_model": meta["text_model"], "revision": meta["revision"],
              "manifest_hashes": meta["manifest_hashes"], "features_sha256": meta["features_sha256"],
              "checkpoint_sha256": digest(output/"text-aux.safetensors")}
    report["data_protocol"]=meta.get("data_protocol",{})
    write_json(output/"validation-predictions.json",[
        {"sample_id":r["sample_id"],"label":r["label"],"uncalibrated_score":float(p)}
        for r,p in zip([r for r in rows if r['split']=='validation'],validation_scores)])
    write_json(output/"report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description="Separate text auxiliary experiment; no image feature fabrication")
    sub = parser.add_subparsers(dest="action", required=True)
    c = sub.add_parser("cache"); c.add_argument("--data", required=True)
    s = sub.add_parser("prepare-sab"); s.add_argument("--data", required=True)
    s.add_argument("--output", required=True)
    f = sub.add_parser("fit"); f.add_argument("--cache", required=True)
    f.add_argument("--config", default="configs/f1t-train.json")
    for cmd in (c, f): cmd.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        if args.action == "prepare-sab":
            from .safeagent_aux import prepare
            result=prepare(args.data,args.output)
        elif args.action == "cache": result=cache(args.data,args.output)
        else: result=fit(args.cache,args.output,args.config)
        print(json.dumps(result, indent=2))
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Auxiliary preparation failed: {exc}\n")


if __name__ == "__main__": main()
