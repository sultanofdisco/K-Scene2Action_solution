import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from demo import demo_inputs
from f2.schemas import plan_schema, state_schema, result_schema

if __name__=='__main__':
    contract,_=demo_inputs()
    output=ROOT/'examples/contract'
    for name,builder in [('plan_schema.json',plan_schema),('state_schema.json',state_schema),('result_schema.json',result_schema)]:
        schema=builder(contract)
        schema['$schema']='https://json-schema.org/draft/2020-12/schema'
        (output/name).write_text(json.dumps(schema,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Exported profile-specific plan, observation, result wire schemas.')
