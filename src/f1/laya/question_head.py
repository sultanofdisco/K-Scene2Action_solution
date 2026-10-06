"""Opt-in question-specific readout over frozen native Laya features."""
import json
import hashlib
from pathlib import Path
from .backend import LayaBackend,InvalidOutput,digest
from .finetuned import lock_hash
from .questions_v2 import QUESTIONS,PROMPT_VERSION

def context_profile(context):
    if set(context)!={'command','target','explicit_goal','constraints'}:
        raise ValueError('Unexpected trusted context fields')
    return hashlib.sha256(json.dumps(context,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode('utf-8')).hexdigest()

def native_status(answers):
    from .pipeline import parse_answer
    d=[str(parse_answer(answers[q]).decision) for q in QUESTIONS]
    return 'ATTACK' if 'YES' in d[:3] else 'CLEAR' if d==['NO','NO','NO','YES'] else 'UNKNOWN'


class QuestionReadout:
    def __init__(self,checkpoint,base_lock):
        import torch
        from safetensors.torch import load_file
        path=Path(checkpoint)
        self.lock=json.loads((path/'question-head-lock.json').read_text(encoding='utf-8'))
        if self.lock['base_model_lock_sha256']!=lock_hash(base_lock):
            raise ValueError('Question head requires its exact original frozen base')
        if self.lock['prompt_version']!=PROMPT_VERSION or self.lock['questions']!=list(QUESTIONS):
            raise ValueError('Question head requires exact V2 questions and order')
        if self.lock['architecture']!='question-specific-linear-v1' or self.lock['classes']!=['YES','NO','UNKNOWN']:
            raise ValueError('Unexpected question head architecture or labels')
        if digest(path/'question-head.safetensors')!=self.lock['weights_sha256']:
            raise ValueError('Question head hash mismatch')
        self.tensors=load_file(str(path/'question-head.safetensors'))
        shapes={'weight':(4,3,2304),'bias':(4,3),'mean':(4,2304),'scale':(4,2304)}
        if set(self.tensors)!=set(shapes):raise ValueError('Unexpected question head tensors')
        for name,shape in shapes.items():
            t=self.tensors[name]
            if tuple(t.shape)!=shape or t.dtype!=torch.float64 or not torch.isfinite(t).all():
                raise ValueError('Invalid question head parameters')
        if not (self.tensors['scale']>0).all():raise ValueError('Invalid feature scale')

    def logits(self,features):
        import torch
        if features.ndim!=3 or tuple(features.shape[1:])!=(3,768) or len(features)%4:
            raise InvalidOutput('Unexpected Laya feature shape or question count')
        x=features.to(torch.float64).reshape(-1,4,2304);t=self.tensors
        if not torch.isfinite(x).all():raise InvalidOutput('Nonfinite Laya features')
        return torch.einsum('nqd,qcd->nqc',(x-t['mean'])/t['scale'],t['weight'])+t['bias']

    def answers(self,features):
        probs=self.logits(features).softmax(-1)
        if len(probs)!=1:raise InvalidOutput('Expected one request')
        return {q:{'type':'choice','choice':'ABC'[int(p.argmax())],
                   'probabilities':dict(zip('ABC',[round(float(x),4) for x in p]))}
                for q,p in zip(QUESTIONS,probs[0])}


class QuestionHeadLayaBackend(LayaBackend):
    """Diagnostic opt-in only; existing default backend is unchanged."""
    def __init__(self,checkpoint,**kwargs):
        super().__init__(**kwargs);self.checkpoint=Path(checkpoint);self.readout=None

    def _load(self):
        super()._load()
        if self.readout is None:
            self.readout=QuestionReadout(self.checkpoint,self.metadata['model_lock'])
            self.version+='+question-head@'+self.readout.lock['weights_sha256'][:16]
            self.metadata['question_head']=self.readout.lock

    def predict(self,state,questions):
        if questions!=QUESTIONS or list(questions)!=list(QUESTIONS):
            raise ValueError('Question head requires exact V2 questions and order')
        self._load();captured=[]
        hook=self.agent.model.scorer.register_forward_pre_hook(lambda m,a:captured.append(a[0].detach().clone()))
        try:
            result=super().predict(state,questions)
        finally:
            hook.remove()
        if len(captured)!=1 or tuple(captured[0].shape)!=(4,3,768):
            raise InvalidOutput('Native Laya did not return four question features')
        result['answers']=self.readout.answers(captured[0])
        return result


class ProfileRepairLayaBackend(QuestionHeadLayaBackend):
    """Diagnostic candidate; failed fresh attack stress tests. Not a safe default."""
    def __init__(self,checkpoint,**kwargs):
        super().__init__(checkpoint,**kwargs);self.policy=None

    def _load(self):
        super()._load()
        if self.policy is None:
            import math
            p=json.loads((self.checkpoint/'fp-policy.json').read_text(encoding='utf-8'))
            if p['policy_version']!='f1o-profile-normal-repair-v1' or p['question_head_sha256']!=self.readout.lock['weights_sha256'] or p['base_model_lock_sha256']!=lock_hash(self.metadata['model_lock']) or p['prompt_version']!=PROMPT_VERSION:
                raise ValueError('Profile policy does not match model and prompt')
            for item in p['profiles'].values():
                t=item['threshold']
                if type(t) not in (int,float) or not math.isfinite(t) or not .5<=t<=1:
                    raise ValueError('Invalid conservative profile threshold')
            self.policy=p;self.version+='+'+p['policy_version'];self.metadata['fp_policy']=p

    def predict(self,state,questions):
        if questions!=QUESTIONS or list(questions)!=list(QUESTIONS):raise ValueError('Profile repair requires exact V2 questions and order')
        self._load();self.metadata.pop('normal_repair',None)
        profile=context_profile(state['trusted_context']);policy=self.policy['profiles'].get(profile)
        if policy is None:
            result=LayaBackend.predict(self,state,questions)
            self.metadata['normal_repair']={'profile_supported':False,'applied':False,'profile':profile,'native_status':native_status(result['answers'])}
            return result
        captured=[];hook=self.agent.model.scorer.register_forward_pre_hook(lambda m,a:captured.append(a[0].detach().clone()))
        try:result=LayaBackend.predict(self,state,questions)
        finally:hook.remove()
        if len(captured)!=1 or tuple(captured[0].shape)!=(4,3,768):raise InvalidOutput('Unexpected native features')
        probs=self.readout.logits(captured[0]).softmax(-1)[0]
        score=float(probs[[0,1,2,3],[1,1,1,0]].min())
        answers=self.readout.answers(captured[0])
        from .pipeline import parse_answer
        clear=[str(parse_answer(answers[q]).decision) for q in QUESTIONS]==['NO','NO','NO','YES']
        accepted=clear and score>policy['threshold']
        original_status=native_status(result['answers'])
        support={}
        if accepted:accepted,support=self._support_accept(captured[0])
        if accepted:result['answers']=answers
        self.metadata['normal_repair']={'profile_supported':True,'profile':profile,'applied':accepted,'score':score,'threshold':policy['threshold'],'support':support,'native_status':original_status}
        return result

    def _support_accept(self,features):return True,{}


class SupportRepairLayaBackend(ProfileRepairLayaBackend):
    """Additional empirical training-neighborhood guard, still diagnostic only."""
    def __init__(self,checkpoint,**kwargs):
        super().__init__(checkpoint,**kwargs);self.support_tensors=None;self.support_lock=None

    def _load(self):
        super()._load()
        if self.support_tensors is None:
            import torch,math
            from safetensors.torch import load_file
            p=json.loads((self.checkpoint/'support-lock.json').read_text(encoding='utf8'))
            if p['policy_version']!='f1o-neighborhood-support-v1' or p['question_head_sha256']!=self.readout.lock['weights_sha256'] or p['profile_policy_sha256']!=digest(self.checkpoint/'fp-policy.json') or p['weights_sha256']!=digest(self.checkpoint/'support.safetensors'):
                raise ValueError('Support guard does not match the frozen checkpoint')
            if not .5<=p['threshold']<=1 or not math.isfinite(p['normal_radius']) or p['normal_radius']<0:raise ValueError('Invalid support bounds')
            t=load_file(str(self.checkpoint/'support.safetensors'))
            if set(t)!={'mean','scale','normal','nonclear'}:raise ValueError('Invalid support tensors')
            if any(v.dtype!=torch.float64 or not torch.isfinite(v).all() for v in t.values()):raise ValueError('Nonfinite support tensors')
            if t['mean'].shape!=(9216,) or t['scale'].shape!=(9216,) or not (t['scale']>0).all():raise ValueError('Invalid support normalization')
            if any(t[k].ndim!=2 or t[k].shape[1]!=9216 or len(t[k])==0 for k in ['normal','nonclear']):raise ValueError('Invalid support examples')
            self.support_tensors=t;self.support_lock=p;self.metadata['support_guard']=p;self.version+='+'+p['policy_version']

    def _support_accept(self,features):
        import torch
        t=self.support_tensors;p=self.support_lock
        x=(features.to(torch.float64).reshape(1,9216)-t['mean'])/t['scale'];x=x/x.norm(dim=-1,keepdim=True).clamp_min(1e-12)
        dn=float(torch.cdist(x,t['normal']).min());da=float(torch.cdist(x,t['nonclear']).min());score=da/max(da+dn,1e-12)
        accepted=score>p['threshold'] and dn<=p['normal_radius']
        return accepted,{'score':score,'normal_distance':dn,'threshold':p['threshold'],'normal_radius':p['normal_radius'],'accepted':accepted}


def assess_profile_repair(request,context,binding,ocr,backend,trace=None):
    """Explicit V2 entry point preserving the existing binding, OCR and gate checks."""
    from .pipeline import assess
    class Adapter:
        @property
        def metadata(self):return backend.metadata
        @property
        def version(self):return backend.version
        def predict(self,state,questions):
            if list(questions)!=list(QUESTIONS):raise ValueError('Unexpected pipeline question keys')
            return backend.predict(state,QUESTIONS)
    trace=trace if trace is not None else {}
    result=assess(request,context,binding,ocr,Adapter(),trace)
    trace['prompt_version']=PROMPT_VERSION
    trace['normal_repair_policy']='f1o-profile-normal-repair-v1'
    return result
