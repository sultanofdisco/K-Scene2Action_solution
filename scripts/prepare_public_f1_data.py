"""Audit the supplied archive and prepare reviewed, text-only SafeText auxiliary data."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import random
from itertools import combinations
import zipfile

from f1.training.data import digest, write_json

# Explicit review: one clear safe advice and the original unsafe advice per scenario.
# All other advice remains available in the untouched source and review ledger.
SELECTION = {
    "pressurized_containers": {9:1,116:1,237:0,341:0},
    "cleaning_chemicals": {17:0,22:0,24:2,28:1,31:0,173:1,179:1,183:0,364:0},
    "microwave_misuse": {8:2,32:1,43:2,162:1,242:0},
    "hot_oil": {34:0,72:2,178:1,225:0},
    "indoor_combustion": {50:1,79:2,151:1,259:0,288:0},
    "gas_and_alarms": {96:0,159:1,211:0,203:2,244:0},
    "electrical_contact": {153:2,196:2,202:0,284:0,289:2,336:1,365:1,214:0,282:2,290:2},
    "sharp_objects": {213:0,227:1,233:0,347:1},
    "batteries": {253:0,359:0},
    "hot_surfaces": {190:0,295:2},
    "uncontrolled_fire": {353:2},
    "unidentified_liquid": {281:1},
}
MODEL = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
REVISION = "58edf8cada9e398793dca955574a48cbb7f18be2"


def norm(text):
    return " ".join(text.casefold().split())


def read_lines(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8-sig").splitlines() if s.strip()]


def write_lines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows), encoding="utf-8")


def assign_splits(records):
    groups = sorted({r["group_id"] for r in records})
    random.Random(42).shuffle(groups)
    counts = Counter(r["group_id"] for r in records)
    assignments = {}
    target = round(len(records)*.15)
    for split in ("validation", "test"):
        # Whole hazard families only. Select by size, never by model performance.
        choices = [c for size in range(2, len(groups)-1) for c in combinations(groups, size)]
        chosen = min(choices, key=lambda c: (abs(sum(counts[g] for g in c)-target), len(c)))
        assignments.update({g: split for g in chosen})
        groups = [g for g in groups if g not in chosen]
    assignments.update({g: "train" for g in groups})
    return assignments


def prepare(archive, output, baseline):
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    source.mkdir()
    inventory = []
    with zipfile.ZipFile(archive) as z:
        if len(z.namelist()) != len(set(z.namelist())):
            raise ValueError("Duplicate archive member")
        if sum(i.file_size for i in z.infolist()) > 500_000_000:
            raise ValueError("Unexpected expanded archive size")
        for info in z.infolist():
            name = info.filename.replace("\\", "/")
            rel = PurePosixPath(name)
            target = (source / name).resolve()
            if rel.is_absolute() or ".." in rel.parts or ":" in name or not target.is_relative_to(source.resolve()):
                raise ValueError("Unsafe archive path")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            raw = z.read(info)
            target.write_bytes(raw)
            inventory.append({"path": name, "bytes": len(raw), "sha256": sha256(raw).hexdigest()})
    manifest = json.loads((source/"download_manifest.json").read_text(encoding="utf-8-sig"))
    for entry in manifest["downloads"]:
        path = (source/entry["file"]).resolve()
        if not path.is_relative_to(source.resolve()) or digest(path) != entry["sha256"] or path.stat().st_size != entry["bytes"]:
            raise ValueError("Source download checksum mismatch")
    from PIL import Image
    baseline_rows = read_lines(baseline)
    baseline_images = {r["image_sha256"] for r in baseline_rows}
    image_overlap = []
    images = []
    for item in inventory:
        if item["path"].lower().endswith((".png", ".jpg", ".jpeg")):
            with Image.open(source/item["path"]) as im:
                size = im.size; im.verify()
            images.append({**item, "width": size[0], "height": size[1]})
            if item["sha256"] in baseline_images:
                image_overlap.append(item["path"])
    raw_rows = read_lines(source/"safetext/advice_pairs.jsonl")
    original = list(json.loads((source/"safetext/SafeText.json").read_text(encoding="utf-8-sig")).items())
    for row in raw_rows:
        _, number, role, index = row["case_id"].split("-")
        scenario, advice = original[int(number)]
        if scenario != row["scenario"] or advice[role][int(index)] != row["advice"] or role != row["source_label"]:
            raise ValueError("SafeText export differs from original")
    chosen = {}
    for family, ids in SELECTION.items():
        for number, safe_index in ids.items():
            chosen[f"safetext-{number}-safe-{safe_index}"] = family
            chosen[f"safetext-{number}-unsafe-0"] = family
    if not chosen.keys() <= {r["case_id"] for r in raw_rows}:
        raise ValueError("Missing reviewed ID")
    records, ledger = [], []
    baseline_texts = {norm(r["command"]) for r in baseline_rows}
    for row in raw_rows:
        selected = row["case_id"] in chosen
        ledger.append({"sample_id": row["case_id"], "source_label": row["source_label"],
                       "status": "SELECTED_TEXT_AUX_AGENT_REVIEWED" if selected else "HELD_NOT_SELECTED",
                       "reason": "Explicitly reviewed household physical-risk contrast; image absent" if selected else
                       "Outside initial household subset, ambiguous/context-dependent advice, or redundant alternate advice; no new label assigned"})
        if not selected:
            continue
        text = f"Scenario: {row['scenario']}\nProposed action: {row['advice']}"
        if norm(text) in baseline_texts or norm(row["advice"]) in baseline_texts:
            raise ValueError("Selected auxiliary text duplicates baseline; review before continuing")
        records.append({"schema_version": "f1t-text-aux-1", "sample_id": row["case_id"],
                        "scenario_id": row["scenario_group"], "hazard_family": chosen[row["case_id"]],
                        "scenario": row["scenario"], "proposed_action": row["advice"], "text": text,
                        "text_sha256": sha256(text.encode()).hexdigest(),
                        "label": int(row["source_label"] == "unsafe"), "source_label": row["source_label"],
                        "source": "sharonlevy/SafeText", "source_revision": "25bb7a8c4a00634bc0ea1b5e85bec85cf32f36aa",
                        "language": "en", "modality": "text_only", "image_path": None,
                        "review_status": "SELECTED_TEXT_AUX_AGENT_REVIEWED", "human_reviewed": False,
                        "label_basis": "SOURCE_ADVICE_SAFETY_PLUS_AGENT_TEXT_REVIEW",
                        "usage": "auxiliary_text_experiment_only"})
    # Link hazard families and repeated advice, not merely rows within one scenario.
    parents = {r["hazard_family"]: r["hazard_family"] for r in records}
    def find(x):
        while parents[x] != x:
            parents[x] = parents[parents[x]]; x = parents[x]
        return x
    seen = {}
    for row in records:
        key = norm(row["proposed_action"])
        if key in seen:
            a, b = sorted((find(row["hazard_family"]), find(seen[key])))
            parents[b] = a
        seen[key] = row["hazard_family"]
    for row in records:
        row["group_id"] = "safetext-family-"+find(row["hazard_family"])
    counts = Counter(r["group_id"] for r in records)
    assignments = assign_splits(records)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    lengths = tokenizer([r["text"] for r in records], add_special_tokens=True, truncation=False)["input_ids"]
    for row, tokens in zip(records, lengths):
        if len(tokens) > 128:
            raise ValueError("Reviewed input exceeds encoder budget; no silent truncation")
        row["token_count"] = len(tokens); row["split"] = assignments[row["group_id"]]
    for split in ("train", "validation", "test"):
        part = [r for r in records if r["split"] == split]
        if {r["label"] for r in part} != {0, 1}:
            raise ValueError("Each auxiliary split needs both labels")
        write_lines(output/f"text_aux/{split}.jsonl", part)
    write_lines(output/"text_aux/selected.jsonl", records)
    write_lines(output/"text_aux/review_ledger.jsonl", ledger)
    routes = [
        {"source": "SafeText", "rows": len(raw_rows), "selected_text_aux": len(records), "multimodal_train": 0, "role": "text_auxiliary"},
        {"source": "JBB-Behaviors", "rows": 200, "multimodal_train": 0, "role": "held_general_text_safety_not_robot_action_gold"},
        {"source": "deepset/prompt-injections", "rows": 662, "original_train": 546, "original_test": 116, "multimodal_train": 0, "role": "reserved_F1O_candidates_no_question_gold"},
        {"source": "CyberSecEval3", "metadata_rows": 1000, "available_images": 12, "multimodal_train": 0, "role": "evaluation_only_source_card_excludes_training"},
        {"source": "SIUO", "metadata_rows": 167, "available_images": 10, "multimodal_train": 0, "role": "reserved_external_evaluation_project_labels_pending"},
    ]
    write_json(output/"routing.json", routes)
    write_json(output/"provenance.json", {"archive": str(archive), "archive_sha256": digest(archive),
                                         "files": inventory, "download_checks_verified": len(manifest["downloads"]),
                                         "images": images, "baseline_manifest_sha256": digest(baseline)})
    for source_name, filename in [("siuo", "evaluation_subset_10.jsonl"), ("cyberseceval3", "evaluation_subset_12.jsonl")]:
        evaluation = []
        for r in read_lines(source/source_name/filename):
            image_path = Path("source")/source_name/r["local_image"]
            evaluation.append({**r, "image_path": image_path.as_posix(), "image_sha256": digest(output/image_path),
                               "project_label": None, "training_allowed_in_this_preparation": False})
        write_lines(output/f"evaluation/{source_name}.jsonl", evaluation)
    summary = {"multimodal_train_added": 0, "text_aux_samples": len(records), "scenarios": len(records)//2,
               "label_counts": dict(Counter(r["label"] for r in records)), "groups": len(counts),
               "splits": {s: {"rows": sum(r["split"] == s for r in records),
                                 "groups": sum(v == s for v in assignments.values()),
                                 "labels": dict(Counter(r["label"] for r in records if r["split"] == s))} for s in ("train", "validation", "test")},
               "tokenizer_model": MODEL, "tokenizer_revision": REVISION, "max_tokens": max(r["token_count"] for r in records),
               "baseline_exact_text_overlap": 0, "baseline_exact_image_overlap": image_overlap,
               "no_image_fabrication": True, "baseline_modified": False, "training_run": False}
    write_json(output/"summary.json", summary)
    (output/".gitignore").write_text("source/\n__pycache__/\n", encoding="utf8")
    safe_source = source/"safetext"
    (output/"text_aux/LICENSE").write_bytes((safe_source/"LICENSE").read_bytes())
    (output/"text_aux/SOURCE_README.md").write_bytes((safe_source/"README.md").read_bytes())
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/public_f1_bundle_v1"))
    parser.add_argument("--baseline", type=Path, default=Path("data/robojailbench_f1t_english_v1/selected.jsonl"))
    args = parser.parse_args()
    print(json.dumps(prepare(args.archive, args.output, args.baseline), indent=2))
