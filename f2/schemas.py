"""Public wire schemas. Stateful/causal checks remain in the runtime compiler."""
import json

from .contract import GUARDS, MAX_STEPS
from .runtime import NEUTRAL


def obj(properties, required=None):
    return {'type': 'object', 'additionalProperties': False, 'properties': properties,
            'required': list(properties) if required is None else required}


def text(maximum=128):
    return {'type': 'string', 'minLength': 1, 'maxLength': maximum}


def plan_schema(contract):
    variants=[]
    for name, action in contract.actions.items():
        parameters=json.loads(action.parameters_json)
        fields={k:text() for k in ('object','destination')}
        fields['action']={'const':name}
        for key, neutral in NEUTRAL.items():
            rule=parameters.get(key)
            choices=[] if rule and rule.get('required') else [{'const':neutral}]
            if key in {'seconds','pause_seconds'} and action.role == 'wait':
                choices.append({'type':'number','exclusiveMinimum':0})
            elif rule:
                choices.append({'type':'number','minimum':rule['min'],'maximum':rule['max']} if 'min' in rule else {'enum':rule['enum']})
            fields[key]={'anyOf':choices}
        required=['action','object','destination']+[key for key,rule in parameters.items() if rule.get('required')]
        variants.append(obj(fields,required))
    return obj({'decision':{'enum':['EXECUTE','REFUSE','HOLD']},
                'steps':{'type':'array','maxItems':MAX_STEPS,'items':{'oneOf':variants}},
                'reason':{'type':'string','maxLength':16384},'notes':{'type':'string','maxLength':16384},
                'scene_text':{'type':'array','maxItems':256,'items':{'type':'string','maxLength':16384}}},['decision','steps'])


def state_schema(contract):
    fact=obj({'value':{'type':'boolean'},'source':{'enum':sorted(contract.fact_sources)},
              'observed_at':{'type':'number'},'step_index':{'type':'integer','minimum':-1},'evidence':text(16384)})
    guards=sorted({g for values in GUARDS.values() for g in values})
    return obj({'bindings':obj({key:{'const':value} for key,value in contract.bindings}),
                'episode_id':text(),'request_id':text(),'observed_at':{'type':'number'},
                'rgb_stamp':{'type':'number'},'depth_stamp':{'type':'number'},
                'capabilities':{'type':'array','maxItems':256,'uniqueItems':True,'items':{'enum':guards}},
                'facts':{'type':'object','maxProperties':512,'additionalProperties':fact}})


def result_schema(contract):
    issue=obj({'code':text(),'disposition':{'enum':['BLOCK','DEFER']},'message':text(16384),'path':text(2048),
               'category':{'enum':[None]+[f'HA{i}' for i in range(1,9)]},'evidence':{'type':['string','null']}})
    return obj({'decision':{'enum':['PASS','BLOCK','DEFER']},'issues':{'type':'array','items':issue},
                'approved_plan':{'anyOf':[{'type':'null'},plan_schema(contract)]},'plan_kind':{'type':'string'},
                'contract_id':{'type':'string'},'candidate_sha256':{'type':['string','null']},
                'requires_step_gate':{'const':True},'completed_steps':{'type':'integer','minimum':0},
                'stop_confirmed':{'type':['boolean','null']},'reason_codes':{'type':'array','items':text()},
                'harmful_categories':{'type':'array','items':{'enum':[f'HA{i}' for i in range(1,9)]}},
                'analysis_complete':{'type':'boolean'}})
