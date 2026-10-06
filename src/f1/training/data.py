"""Validate reviewed pairs and construct conservative diagnostic folds."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

EXCLUDED = {"Ethical_Social_Offenses", "Permission_Related_Offense"}

def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")

def read_config(path):
    c=json.loads(Path(path).read_text(encoding="utf-8"))
    for k in ["embedding_dimension", "max_tokens", "hidden_dimension", "batch_size", "cpu_threads", "epochs", "seed"]:
        if type(c[k]) is not int or c[k] < (0 if k=="seed" else 1):
            raise ValueError(f"Invalid {k}")
    if c["embedding_dimension"]!=512 or c["max_tokens"]!=128:
        raise ValueError("This encoder pair requires 512 dimensions and a 128-token limit")
    if not 0 <= c["dropout"] < 1 or not 0 < c["learning_rate"] < 1 or not 0 <= c["weight_decay"] < 1:
        raise ValueError("Invalid optimizer or dropout configuration")
    if c["preprocessing"]!="exif-rgb-pad224-clipmean-v1":
        raise ValueError("Unknown image preprocessing")
    return c

def load_manifest(path):
    path=Path(path).resolve(); root=path.parent
    rows=[json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]
    if not rows: raise ValueError("Empty manifest")
    from PIL import Image
    ids=set(); checked=set(); pairs=defaultdict(list); links=defaultdict(set)
    for r in rows:
        if r["sample_id"] in ids: raise ValueError("Duplicate sample_id")
        ids.add(r["sample_id"])
        if type(r["label"]) is not int or r["label"] not in (0,1): raise ValueError("Only binary reviewed labels are allowed")
        if r["review_status"]!="SELECTED_AGENT_REVIEWED": raise ValueError("Unselected sample")
        if r["language"]!="en": raise ValueError("This experiment uses English commands")
        if r["source"].split("/")[-1] in EXCLUDED: raise ValueError("User-excluded category")
        if not r["command"].strip() or not r["group_id"].strip(): raise ValueError("Missing command/group")
        if hashlib.sha256(r["command"].encode("utf-8")).hexdigest()!=r["command_sha256"]:
            raise ValueError("Command hash mismatch")
        image=(root/r["image_path"]).resolve()
        if not image.is_relative_to(root): raise ValueError("Image path escapes dataset root")
        if digest(image)!=r["image_sha256"]: raise ValueError("Image hash mismatch")
        if image not in checked:
            with Image.open(image) as im:
                if im.size!=(r["image_width"],r["image_height"]): raise ValueError("Image dimensions mismatch")
                im.verify()
            checked.add(image)
        else:
            with Image.open(image) as im:
                if im.size!=(r["image_width"],r["image_height"]): raise ValueError("Image dimensions mismatch")
        pairs[r["pair_id"]].append(r)
        links[("image",r["image_sha256"])].add(r["group_id"])
        links[("command"," ".join(r["command"].casefold().split()))].add(r["group_id"])
    if any(len(g)!=1 for g in links.values()): raise ValueError("Duplicate image/command crosses groups")
    for pair in pairs.values():
        if len(pair)!=2 or {r["label"] for r in pair}!={0,1}: raise ValueError("Incomplete contrast pair")
        if len({(r["group_id"],r["image_sha256"]) for r in pair})!=1: raise ValueError("Pair identity mismatch")
    return rows

def make_folds(rows):
    groups=sorted({r["group_id"] for r in rows})
    if len(groups)<2: raise ValueError("At least two independent groups required")
    return [{"held_group":g,
             "train_ids":[r["sample_id"] for r in rows if r["group_id"]!=g],
             "validation_ids":[r["sample_id"] for r in rows if r["group_id"]==g]}
            for g in groups]

def prepare(manifest, output):
    rows=load_manifest(manifest); output=Path(output); output.mkdir(parents=True,exist_ok=False)
    report={"manifest_sha256":digest(manifest),"samples":len(rows),
            "pairs":len(rows)//2,"label_counts":dict(Counter(r["label"] for r in rows)),
            "group_sizes":dict(Counter(r["group_id"] for r in rows)),
            "validation_method":"leave_one_group_out_diagnostic_only",
            "deployment_ready":False,"token_check":"pending_encoder_load"}
    write_json(output/"data-audit.json",report)
    write_json(output/"folds.json",make_folds(rows))
    return report
