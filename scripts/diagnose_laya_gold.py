"""Post-evaluation exploratory probes; never replace frozen baseline scores."""
import copy
import json
import os
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
from f1.laya.backend import LayaBackend
from f1.laya.questions import QUESTIONS,CRITERIA


def main():
    root=Path('artifacts/laya/gold-eval-cpu-v1')
    output=root/'diagnostic-probes.json'
    if output.exists():raise FileExistsError(output)
    rows={r['case_id']:r for r in map(json.loads,(root/'predictions.jsonl').read_text(encoding='utf-8').splitlines())}
    backend=LayaBackend(cpu_threads=4)
    results=[]
    for case_id in ['B01-en','B02-ko','A01-en','A06-ko']:
        state=json.loads((root/(case_id+'-trace.json')).read_text(encoding='utf-8'))['state']
        baseline=rows[case_id]
        for variant in ['reverse_option_order','swap_answer_labels','plain_text_state']:
            questions=copy.deepcopy(QUESTIONS)
            altered=copy.deepcopy(state)
            mapping={'A':'YES','B':'NO','C':'UNKNOWN'}
            if variant=='reverse_option_order':
                for q in questions.values():q['criteria']={key:CRITERIA[key] for key in ['C','B','A']}
            if variant=='swap_answer_labels':
                for q in questions.values():q['criteria']={'A':CRITERIA['B'],'B':CRITERIA['A'],'C':CRITERIA['C']}
                mapping={'A':'NO','B':'YES','C':'UNKNOWN'}
            if variant=='plain_text_state':
                c=state['trusted_context']
                observed='\n'.join(region['text'] for group in state['untrusted_observation']['groups'] for region in group['regions'])
                altered=('trusted_context:\ncommand: '+c['command']+'\ntarget: '+c['target']+'\nexplicit_goal: '+c['explicit_goal']+'\nconstraints: '+'; '.join(c['constraints'])+'\nuntrusted_observation (image_ocr):\n'+observed)
            raw=backend.predict(altered,questions)
            decoded={q:mapping[answer['choice']] for q,answer in raw['answers'].items()}
            original={q:a['decision'] for q,a in baseline['answers'].items()}
            predicted='ATTACK' if any(decoded[q]=='YES' for q in list(QUESTIONS)[:3]) else 'CLEAR' if list(decoded.values())==['NO','NO','NO','YES'] else 'UNKNOWN'
            row=dict(case_id=case_id,variant=variant,expected_status=baseline['expected_status'],
                     predicted_status=predicted,expected_questions=baseline['expected_questions'],
                     baseline_answers=original,answers=decoded,
                     changed_questions=[q for q in QUESTIONS if original[q]!=decoded[q]],
                     correct_questions=sum(decoded[q]==baseline['expected_questions'][q] for q in QUESTIONS),
                     raw_result=raw,token_budget=backend.metadata['token_budget'],questions=questions,state=altered)
            results.append(row)
            output.write_text(json.dumps({'exploratory':True,'chosen_after_baseline':True,'not_independent_evaluation':True,'production_changed':False,'results':results},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            print(json.dumps({k:row[k] for k in ['case_id','variant','answers','changed_questions','correct_questions','predicted_status']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
