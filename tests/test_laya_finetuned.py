import json
from types import SimpleNamespace
import pytest
import torch
from safetensors.torch import save_file
from f1.laya.backend import digest
from f1.laya.finetuned import apply_scorer_delta,lock_hash,ScorerTunedLayaBackend
from f1.laya.questions import QUESTIONS as V1


def fixture(tmp_path):
    model=torch.nn.Module()
    model.scorer=torch.nn.Linear(3,1)
    model.encoder=torch.nn.Linear(3,3)
    backend=SimpleNamespace(agent=SimpleNamespace(model=model),metadata={'model_lock':{'repo':'fixture','revision':'fixed'}},version='fixture')
    weights={'weight':torch.full((1,3),2.0),'bias':torch.ones(1)}
    save_file(weights,str(tmp_path/'scorer.safetensors'))
    manifest={'base_model_lock_sha256':lock_hash(backend.metadata['model_lock']),'scorer_sha256':digest(tmp_path/'scorer.safetensors'),'trainable_module':'scorer'}
    (tmp_path/'scorer-lock.json').write_text(json.dumps(manifest))
    return backend,manifest


def test_delta_updates_only_scorer(tmp_path):
    backend,_=fixture(tmp_path)
    encoder=backend.agent.model.encoder.weight.detach().clone()
    apply_scorer_delta(backend,tmp_path)
    assert torch.equal(backend.agent.model.scorer.weight,torch.full((1,3),2.0))
    assert torch.equal(backend.agent.model.encoder.weight,encoder)
    assert 'scorer_finetuning' in backend.metadata


def test_delta_rejects_wrong_base(tmp_path):
    backend,_=fixture(tmp_path)
    backend.metadata['model_lock']['revision']='changed'
    with pytest.raises(ValueError,match='exact frozen base'):apply_scorer_delta(backend,tmp_path)


def test_delta_rejects_tampered_weights(tmp_path):
    backend,_=fixture(tmp_path)
    with (tmp_path/'scorer.safetensors').open('ab') as f:f.write(b'bad')
    with pytest.raises(ValueError,match='hash mismatch'):apply_scorer_delta(backend,tmp_path)


def test_finetuned_backend_rejects_wrong_prompt_before_loading():
    backend=ScorerTunedLayaBackend('does-not-exist')
    with pytest.raises(ValueError,match='V2 question'):backend.predict({},V1)
    assert backend.agent is None
