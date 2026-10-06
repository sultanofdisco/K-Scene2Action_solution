import copy
import runpy
from pathlib import Path

from f1.laya.questions import QUESTIONS as V1
from f1.laya.questions_v2 import QUESTIONS as V2


def test_comparison_adapter_sends_candidate_to_model_without_mutating_default():
    module=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/compare_laya_prompts.py'))
    class Backend:
        version='fake'
        metadata={}
        def predict(self,state,questions):
            self.received=copy.deepcopy(questions)
            return {'sent':True}
    base=Backend()
    original=copy.deepcopy(V1)
    adapter=module['CandidateBackend'](base,V2)
    assert adapter.predict({'observation':'label'},V1)=={'sent':True}
    assert base.received==V2 and base.received!=V1
    assert V1==original


def test_paired_comparison_rejects_different_case_sets():
    import pytest
    module=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/compare_laya_prompts.py'))
    with pytest.raises(AssertionError):
        module['compare']([{'case_id':'a','suite':'semantic'}],[{'case_id':'b','suite':'semantic'}],lambda rows:{})


def test_paired_comparison_preserves_regressions():
    module=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/compare_laya_prompts.py'))
    before=[{'case_id':'a','suite':'semantic','status_correct':True,'actual_status':'ATTACK'}]
    after=[{'case_id':'a','suite':'semantic','status_correct':False,'actual_status':'CLEAR'}]
    result=module['compare'](before,after,lambda rows:{'n':len(rows)})
    assert result['regressed_status']==['a'] and result['improved_status']==[]
