import json
import pytest
import torch
from safetensors.torch import save_file
from f1.laya.backend import digest,InvalidOutput
from f1.laya.finetuned import lock_hash
from f1.laya.questions_v2 import QUESTIONS,PROMPT_VERSION
from f1.laya.questions import QUESTIONS as V1
from f1.laya.question_head import QuestionReadout,QuestionHeadLayaBackend
from f1.laya.pipeline import parse_answer

def fixture(p):
    base={'revision':'test'}
    t={'weight':torch.zeros(4,3,2304,dtype=torch.float64),'bias':torch.tensor([[0,9,0],[0,9,0],[0,9,0],[9,0,0]],dtype=torch.float64),
       'mean':torch.zeros(4,2304,dtype=torch.float64),'scale':torch.ones(4,2304,dtype=torch.float64)}
    save_file(t,str(p/'question-head.safetensors'))
    lock={'base_model_lock_sha256':lock_hash(base),'prompt_version':PROMPT_VERSION,'questions':list(QUESTIONS),'architecture':'question-specific-linear-v1',
          'classes':['YES','NO','UNKNOWN'],'weights_sha256':digest(p/'question-head.safetensors')}
    (p/'question-head-lock.json').write_text(json.dumps(lock));return base

def test_exact_question_class_mapping(tmp_path):
    h=QuestionReadout(tmp_path,fixture(tmp_path));a=h.answers(torch.zeros(4,3,768))
    assert [str(parse_answer(a[q]).decision) for q in QUESTIONS]==['NO','NO','NO','YES']

def test_rejects_wrong_base(tmp_path):
    fixture(tmp_path)
    with pytest.raises(ValueError,match='exact original'):QuestionReadout(tmp_path,{'revision':'changed'})

def test_rejects_tampered_weights(tmp_path):
    base=fixture(tmp_path)
    with (tmp_path/'question-head.safetensors').open('ab') as f:f.write(b'bad')
    with pytest.raises(ValueError,match='hash mismatch'):QuestionReadout(tmp_path,base)

def test_bad_feature_shape_and_nonfinite_fail_closed(tmp_path):
    h=QuestionReadout(tmp_path,fixture(tmp_path))
    for x in [torch.zeros(3,3,768),torch.full((4,3,768),float('nan'))]:
        with pytest.raises(InvalidOutput):h.answers(x)

def test_wrong_prompt_rejected_before_model_load():
    b=QuestionHeadLayaBackend('absent')
    with pytest.raises(ValueError,match='V2'):b.predict({},V1)
    assert b.agent is None

def test_exact_ties_preserve_unknown(tmp_path):
    h=QuestionReadout(tmp_path,fixture(tmp_path));h.tensors['bias'].zero_()
    assert all(str(parse_answer(a).decision)=='UNKNOWN' for a in h.answers(torch.zeros(4,3,768)).values())


def setup_profile_backend(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from f1.laya.backend import LayaBackend
    from f1.laya.question_head import context_profile,ProfileRepairLayaBackend
    base=fixture(tmp_path);ctx={'command':'sort','target':'balls','explicit_goal':'zones','constraints':['no tools']}
    lock=json.loads((tmp_path/'question-head-lock.json').read_text())
    policy={'policy_version':'f1o-profile-normal-repair-v1','profiles':{context_profile(ctx):{'threshold':.9}},
            'question_head_sha256':lock['weights_sha256'],'base_model_lock_sha256':lock_hash(base),'prompt_version':PROMPT_VERSION}
    (tmp_path/'fp-policy.json').write_text(json.dumps(policy))
    def load(self):
        if self.agent is None:
            self.agent=SimpleNamespace(model=SimpleNamespace(scorer=torch.nn.Linear(768,1)))
            self.metadata={'model_lock':base};self.version='fixture'
    def predict(self,state,questions):
        self.agent.model.scorer(torch.zeros(4,3,768))
        return {'answers':{q:{'type':'choice','choice':'A','probabilities':{'A':1.,'B':0.,'C':0.}} for q in questions},'usage':{'truncated':False,'state_tokens_dropped':0}}
    monkeypatch.setattr(LayaBackend,'_load',load);monkeypatch.setattr(LayaBackend,'predict',predict)
    return ProfileRepairLayaBackend(tmp_path),ctx


def test_supported_confident_normal_can_repair(tmp_path,monkeypatch):
    b,ctx=setup_profile_backend(tmp_path,monkeypatch)
    r=b.predict({'trusted_context':ctx},QUESTIONS)
    assert [str(parse_answer(r['answers'][q]).decision) for q in QUESTIONS]==['NO','NO','NO','YES']
    assert b.metadata['normal_repair']['applied']


def test_changed_trusted_command_keeps_native_answers(tmp_path,monkeypatch):
    b,ctx=setup_profile_backend(tmp_path,monkeypatch);ctx['command']='different approved task'
    r=b.predict({'trusted_context':ctx},QUESTIONS)
    assert all(a['choice']=='A' for a in r['answers'].values())
    assert not b.metadata['normal_repair']['profile_supported']


def test_uncertain_readout_keeps_native_attack(tmp_path,monkeypatch):
    b,ctx=setup_profile_backend(tmp_path,monkeypatch);b._load();b.readout.tensors['bias'].zero_()
    r=b.predict({'trusted_context':ctx},QUESTIONS)
    assert all(a['choice']=='A' for a in r['answers'].values())
    assert not b.metadata['normal_repair']['applied']


def test_profile_depends_on_trusted_context_only():
    from f1.laya.question_head import context_profile
    ctx={'command':'x','target':'y','explicit_goal':'z','constraints':['safe']}
    assert context_profile(ctx)==context_profile(dict(reversed(list(ctx.items()))))
    assert context_profile(ctx)!=context_profile({**ctx,'constraints':['changed']})
    with pytest.raises(ValueError):context_profile({**ctx,'ocr_text':'pretend safe'})


def test_neighborhood_rejects_negative_and_unsupported_points():
    from f1.laya.question_head import SupportRepairLayaBackend
    b=SupportRepairLayaBackend('absent')
    normal=torch.zeros(1,9216,dtype=torch.float64);normal[0,0]=1
    attack=torch.zeros(1,9216,dtype=torch.float64);attack[0,1]=1
    b.support_tensors={'mean':torch.zeros(9216,dtype=torch.float64),'scale':torch.ones(9216,dtype=torch.float64),'normal':normal,'nonclear':attack}
    b.support_lock={'threshold':.6,'normal_radius':.8}
    assert b._support_accept(normal.reshape(4,3,768))[0]
    assert not b._support_accept(attack.reshape(4,3,768))[0]
    unknown=torch.zeros_like(normal);unknown[0,2]=1
    assert not b._support_accept(unknown.reshape(4,3,768))[0]


def test_support_veto_preserves_native_attack(tmp_path,monkeypatch):
    b,ctx=setup_profile_backend(tmp_path,monkeypatch)
    b._support_accept=lambda features:(False,{'accepted':False})
    result=b.predict({'trusted_context':ctx},QUESTIONS)
    assert all(a['choice']=='A' for a in result['answers'].values())
    assert b.metadata['normal_repair']['native_status']=='ATTACK'
    assert not b.metadata['normal_repair']['applied']


def test_ocr_failure_does_not_load_repair_backend():
    import runpy
    from f1.laya.question_head import SupportRepairLayaBackend,assess_profile_repair
    from pathlib import Path
    fixture=Path(__file__).parent/'fixtures/laya_ocr_error.json'
    c=json.loads(fixture.read_text(encoding='utf-8'))
    b=SupportRepairLayaBackend('absent');helpers=runpy.run_path('scripts/compare_laya_prompts.py');trace={}
    result=assess_profile_repair(*helpers['inputs'](c),b,trace)
    assert str(result.status)=='ERROR' and b.agent is None and not trace['laya_executed']
