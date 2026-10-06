"""Optional native scorer delta for offline F1-O experiments."""
import hashlib,json
from pathlib import Path
from .backend import LayaBackend,digest

def lock_hash(lock):
    return hashlib.sha256(json.dumps(lock,sort_keys=True).encode()).hexdigest()

def apply_scorer_delta(backend,checkpoint):
    from safetensors.torch import load_file
    path=Path(checkpoint)
    manifest=json.loads((path/'scorer-lock.json').read_text(encoding='utf-8'))
    if manifest['base_model_lock_sha256']!=lock_hash(backend.metadata['model_lock']):
        raise ValueError('Scorer delta requires its exact frozen base model')
    if digest(path/'scorer.safetensors')!=manifest['scorer_sha256']:
        raise ValueError('Scorer checkpoint hash mismatch')
    if manifest['trainable_module']!='scorer':
        raise ValueError('Unexpected trainable module')
    backend.agent.model.scorer.load_state_dict(load_file(str(path/'scorer.safetensors')),strict=True)
    backend.agent.model.eval()
    backend.version += '+f1o-scorer@'+manifest['scorer_sha256'][:16]
    backend.metadata['scorer_finetuning']=manifest
    return manifest

class ScorerTunedLayaBackend(LayaBackend):
    def __init__(self,checkpoint,**kwargs):
        super().__init__(**kwargs)
        self.checkpoint=Path(checkpoint)
        self._delta_loaded=False
    def _load(self):
        super()._load()
        if not self._delta_loaded:
            apply_scorer_delta(self,self.checkpoint)
            self._delta_loaded=True
    def predict(self,state,questions):
        from .questions_v2 import QUESTIONS
        if questions!=QUESTIONS:
            raise ValueError('This scorer experiment requires the frozen V2 question set')
        return super().predict(state,questions)

def apply_decision_delta(backend,checkpoint):
    from safetensors.torch import load_file
    path=Path(checkpoint)
    manifest=json.loads((path/'decision-lock.json').read_text(encoding='utf-8'))
    if manifest['base_model_lock_sha256']!=lock_hash(backend.metadata['model_lock']):
        raise ValueError('Decision delta requires its exact frozen base model')
    if digest(path/'decision.safetensors')!=manifest['weights_sha256']:
        raise ValueError('Decision checkpoint hash mismatch')
    weights=load_file(str(path/'decision.safetensors'))
    if not weights or any(not k.startswith(('head.','scorer.')) for k in weights):
        raise ValueError('Only native head and scorer weights are allowed')
    for name in ['head','scorer']:
        state={k[len(name)+1:]:v for k,v in weights.items() if k.startswith(name+'.')}
        getattr(backend.agent.model,name).load_state_dict(state,strict=True)
    backend.agent.model.eval()
    backend.version += '+f1o-decision@'+manifest['weights_sha256'][:16]
    backend.metadata['decision_finetuning']=manifest
    return manifest

class DecisionHeadTunedLayaBackend(ScorerTunedLayaBackend):
    def _load(self):
        LayaBackend._load(self)
        if not self._delta_loaded:
            apply_decision_delta(self,self.checkpoint)
            self._delta_loaded=True
