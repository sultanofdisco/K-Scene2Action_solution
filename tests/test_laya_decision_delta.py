import json
from types import SimpleNamespace
import pytest
import torch
from safetensors.torch import save_file
from f1.laya.backend import digest
from f1.laya.finetuned import apply_decision_delta, lock_hash, DecisionHeadTunedLayaBackend
from f1.laya.questions import QUESTIONS as V1


def fixture(tmp_path, forbidden=False):
    model = torch.nn.Module()
    model.head = torch.nn.Linear(3, 3)
    model.scorer = torch.nn.Linear(3, 1)
    model.encoder = torch.nn.Linear(3, 3)
    backend = SimpleNamespace(agent=SimpleNamespace(model=model), metadata={'model_lock': {'revision': 'fixed'}}, version='fixture')
    weights = {k: torch.full_like(v, 2.0) for k, v in model.state_dict().items() if k.startswith(('head.', 'scorer.'))}
    if forbidden:
        weights['encoder.weight'] = model.encoder.weight.detach().clone()
    save_file(weights, str(tmp_path / 'decision.safetensors'))
    manifest = {'base_model_lock_sha256': lock_hash(backend.metadata['model_lock']), 'weights_sha256': digest(tmp_path / 'decision.safetensors')}
    (tmp_path / 'decision-lock.json').write_text(json.dumps(manifest))
    return backend


def test_decision_delta_updates_head_and_scorer_only(tmp_path):
    backend = fixture(tmp_path)
    encoder = backend.agent.model.encoder.weight.detach().clone()
    apply_decision_delta(backend, tmp_path)
    for module in (backend.agent.model.head, backend.agent.model.scorer):
        assert torch.equal(module.weight, torch.full_like(module.weight, 2.0))
    assert torch.equal(encoder, backend.agent.model.encoder.weight)
    assert 'decision_finetuning' in backend.metadata


def test_decision_delta_rejects_wrong_base(tmp_path):
    backend = fixture(tmp_path)
    backend.metadata['model_lock']['revision'] = 'other'
    with pytest.raises(ValueError, match='exact frozen base'):
        apply_decision_delta(backend, tmp_path)


def test_decision_delta_rejects_tampering(tmp_path):
    backend = fixture(tmp_path)
    with (tmp_path / 'decision.safetensors').open('ab') as f:
        f.write(b'bad')
    with pytest.raises(ValueError, match='hash mismatch'):
        apply_decision_delta(backend, tmp_path)


def test_decision_delta_rejects_encoder_weights(tmp_path):
    backend = fixture(tmp_path, forbidden=True)
    with pytest.raises(ValueError, match='Only native head and scorer'):
        apply_decision_delta(backend, tmp_path)


def test_decision_backend_rejects_v1_without_loading():
    backend = DecisionHeadTunedLayaBackend('does-not-exist')
    with pytest.raises(ValueError, match='V2 question'):
        backend.predict({}, V1)
    assert backend.agent is None
