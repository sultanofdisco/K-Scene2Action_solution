from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import pytest

from f1.training.text_aux import validate_rows, load_aux


@pytest.fixture
def aux_rows():
    rows = []
    for index, split in enumerate(("train", "validation", "test")):
        for label in (0, 1):
            scenario = f"Test scenario {index}"
            action = f"Test action {index} {label}"
            text = f"Scenario: {scenario}\nProposed action: {action}"
            rows.append(dict(schema_version="f1t-text-aux-1", modality="text_only", image_path=None,
                             source="sharonlevy/SafeText", usage="auxiliary_text_experiment_only",
                             review_status="SELECTED_TEXT_AUX_AGENT_REVIEWED", language="en",
                             sample_id=f"{index}-{label}", scenario_id=str(index), hazard_family=str(index),
                             group_id=str(index), scenario=scenario, proposed_action=action, text=text,
                             text_sha256=sha256(text.encode()).hexdigest(), label=label,
                             source_label="unsafe" if label else "safe", split=split))
    return rows


@pytest.mark.parametrize("change", ["image", "evaluation", "label", "text", "group", "family", "duplicate"])
def test_aux_rejects_wrong_modality_source_labels_and_leakage(aux_rows, change):
    rows = deepcopy(aux_rows)
    if change == "image": rows[0]["image_path"] = "fake.png"
    elif change == "evaluation": rows[0]["source"] = "CyberSecEval3"
    elif change == "label": rows[0]["label"] = True
    elif change == "text": rows[0]["text"] += " modified"
    elif change == "group": rows[2]["group_id"] = rows[0]["group_id"]
    elif change == "family": rows[2]["hazard_family"] = rows[0]["hazard_family"]
    else: rows[2]["sample_id"] = rows[0]["sample_id"]
    with pytest.raises(ValueError): validate_rows(rows)


@pytest.fixture
def sab_rows():
    from f1.training.safeagent_aux import SCHEMA, SOURCE, REVISION, POLICY
    rows=[]
    for i,split in enumerate(('train','validation','test')):
        for label in (0,1):
            text=f'Instruction {i} {label}'
            rows.append(dict(schema_version=SCHEMA,source=SOURCE,source_revision=REVISION,
                modality='text_only',image_path=None,language='en',review_status='SELECTED',
                usage='auxiliary_text_experiment_only',split_policy=POLICY,split=split,
                label=label,source_label='unsafe' if label else 'safe',text=text,command=text,
                text_sha256=sha256(text.encode()).hexdigest(),sample_id=f'sab-{i}-{label}',
                group_id=f'g-{i}',source_scene_id=f'scene-{i}',contrast_group_id=f'contrast-{i}',
                text_group_id=f'text-{i}-{label}',category='same_category_across_splits'))
    return rows


def test_sab_allows_known_categories_but_requires_separate_scenes(sab_rows):
    validate_rows(sab_rows)
    sab_rows[2]['source_scene_id']=sab_rows[0]['source_scene_id']
    with pytest.raises(ValueError,match='leakage'): validate_rows(sab_rows)


@pytest.mark.parametrize('change',['held','label','metadata_in_input','group','contrast','text_group','unknown_source','revision','unassigned','normalized_duplicate'])
def test_sab_rejects_unreviewed_input_and_split_leakage(sab_rows,change):
    rows=deepcopy(sab_rows)
    if change=='held': rows[0]['review_status']='HELD'
    elif change=='label': rows[0]['label']=True
    elif change=='metadata_in_input':
        rows[0]['text']='Safe. '+rows[0]['text']
        rows[0]['text_sha256']=sha256(rows[0]['text'].encode()).hexdigest()
    elif change=='group': rows[2]['group_id']=rows[0]['group_id']
    elif change=='contrast': rows[2]['contrast_group_id']=rows[0]['contrast_group_id']
    elif change=='text_group': rows[2]['text_group_id']=rows[0]['text_group_id']
    elif change=='unknown_source': rows[0]['source']='evaluation-only'
    elif change=='revision': rows[0]['source_revision']='main'
    elif change=='unassigned': rows[0]['split']='UNASSIGNED'
    else:
        rows[2]['text']=rows[2]['command']=rows[0]['text'].upper()+'.'
        rows[2]['text_sha256']=sha256(rows[2]['text'].encode()).hexdigest()
    with pytest.raises(ValueError):validate_rows(rows)


def test_sab_load_rejects_manifest_changed_after_split(sab_rows,tmp_path):
    for split in ('train','validation','test'):
        (tmp_path/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in sab_rows if r['split']==split),encoding='utf8')
    (tmp_path/'summary.json').write_text(json.dumps({'manifest_hashes':{'train':'wrong','validation':'wrong','test':'wrong'}}),encoding='utf8')
    with pytest.raises(ValueError,match='checksum'):load_aux(tmp_path)


@pytest.mark.parametrize('source_kind',['safetext','safeagent'])
def test_aux_manifest_and_fit_exclude_test_from_learning_and_scoring(aux_rows, sab_rows, source_kind, tmp_path, monkeypatch):
    if source_kind=='safeagent': aux_rows=sab_rows
    np = pytest.importorskip("numpy"); pytest.importorskip("torch")
    from f1.training import learner
    from f1.training.text_aux import fit
    from f1.training.data import write_json, digest
    for split in ("train", "validation", "test"):
        (tmp_path/f"{split}.jsonl").write_text("".join(json.dumps(r)+"\n" for r in aux_rows if r["split"] == split))
    rows, hashes = load_aux(tmp_path)
    config = json.loads((Path(__file__).resolve().parents[1]/"configs/f1t-train.json").read_text())
    config["epochs"] = 1
    write_json(tmp_path/"config.json", config)
    x = np.random.default_rng(42).normal(size=(6, 512)).astype("float32")
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    np.save(tmp_path/"features.npy", x, allow_pickle=False)
    write_json(tmp_path/"metadata.json", dict(format="f1t-text-aux-cache-1", complete=True, rows=rows,
               features_sha256=digest(tmp_path/"features.npy"), manifest_hashes=hashes,
               text_model=config["text_model"], revision="synthetic-test-only"))
    train = learner.train_model; score = learner.scores; calls = []
    def train_checked(features, labels, config, device):
        assert np.array_equal(features, x[:2]); calls.append("train")
        return train(features, labels, config, device)
    def score_checked(model, features, device):
        assert np.array_equal(features, x[2:4]); calls.append("validation")
        return score(model, features, device)
    monkeypatch.setattr(learner, "train_model", train_checked)
    monkeypatch.setattr(learner, "scores", score_checked)
    report = fit(tmp_path, tmp_path/"output", tmp_path/"config.json")
    assert calls == ["train", "validation"]
    assert report["test_evaluated"] is False
    assert report["compatible_with_current_1024_mlp"] is False
    assert report['validation_baselines']['always_safe_accuracy']==0.5
    assert 'balanced_accuracy_at_0_5' in report['validation_metrics']
    predictions=json.loads((tmp_path/'output/validation-predictions.json').read_text())
    assert {r['sample_id'] for r in predictions}=={r['sample_id'] for r in aux_rows if r['split']=='validation'}
