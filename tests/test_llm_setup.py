import copy
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from contract_fixtures import draft, environment, plan, state, task
from f2.contract import ContractError, ContractManager, compile_contract, load_contract
from f2.json_input import parse_json
from f2.llm_setup import JsonSchemaMapper
from f2.runtime import RuntimeFilter

ROOT=Path(__file__).resolve().parents[1]


class SetupTests(unittest.TestCase):
    def test_json_schema_http_once_then_only_cpu(self):
        requests=[]
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
                body=json.dumps({'choices':[{'finish_reason':'stop','message':{'content':json.dumps(draft())}}]}).encode()
                self.send_response(200); self.send_header('Content-Length',str(len(body))); self.end_headers(); self.wfile.write(body)
            def log_message(self,*args): pass
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
        try:
            client=JsonSchemaMapper(f'http://127.0.0.1:{server.server_port}/v1')
            manager=ContractManager(client)
            c=manager.prepare(task(),environment())
            f=RuntimeFilter(c,clock=lambda:100)
            for _ in range(30): self.assertEqual(f.inspect(plan(),state(c),step_gated=True).decision,'PASS')
            manager.prepare(task(),environment())
            self.assertEqual(client.calls,1)
            self.assertEqual(len(requests),1)
            self.assertEqual(requests[0]['response_format']['type'],'json_schema')
            self.assertFalse(requests[0]['response_format']['json_schema']['schema']['additionalProperties'])
        finally:
            server.shutdown(); server.server_close(); thread.join(timeout=2)

    def test_bad_incomplete_or_tool_llm_response_never_activates(self):
        class Opener:
            def __init__(self,content): self.content=content
            def open(self,request,timeout): return io.BytesIO(self.content)
        bodies=[b'{"choices":[]}',b'not JSON',b'{"choices":[{"finish_reason":"length","message":{"content":"{}"}}]}',
                json.dumps({'choices':[{'finish_reason':'stop','message':{'content':'{}','tool_calls':[{}]}}]}).encode(),
                json.dumps({'choices':[{'finish_reason':'stop','message':{'content':'{"goals":[],"goals":[]}'}}]}).encode()]
        for body in bodies:
            m=ContractManager(JsonSchemaMapper(opener=Opener(body)))
            with self.assertRaises(ContractError): m.prepare(task(),environment())
            self.assertIsNone(m.active)

    def test_mapping_without_expected_goals_requires_normal_and_unsafe_contrasts(self):
        t=task(); t['expected_goals']=None
        e=environment(); d=draft()
        with self.assertRaises(ContractError): compile_contract(d,t,e)
        normal=plan(); wrong=copy.deepcopy(normal); wrong['steps'][4]['destination']='edge'
        baseline=compile_contract(draft(),task(),e)
        contrasts=[dict(plan=normal,state=state(baseline),expected='PASS'),dict(plan=wrong,state=state(baseline),expected='BLOCK')]
        m=ContractManager(lambda payload,schema:d)
        c=m.prepare(t,e,contrasts)
        self.assertIs(m.active,c)
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'contract.json'; c.save(p)
            with self.assertRaises(ContractError): load_contract(p,t,e)
            self.assertEqual(load_contract(p,t,e,contrasts=contrasts).contract_id,c.contract_id)
        contrasts[0]['expected']='BLOCK'
        with self.assertRaises(ContractError): m.prepare(t,e,contrasts)
        self.assertIsNone(m.active)

    def test_cli_offline_setup_and_known_attack_check(self):
        folder=ROOT/'examples/contract'
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'contract.json'
            command=[sys.executable,'-m','f2','setup','--task',str(folder/'task.json'),'--environment',str(folder/'environment.json'),
                     '--mapping',str(folder/'mapping_fixture.json'),'--output',str(output)]
            proc=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(proc.returncode,0,proc.stderr)
            self.assertEqual(json.loads(proc.stdout)['llm_calls'],0)
            # CLI checks must not accidentally treat serialized demo timestamps as fresh.
            state_path=Path(directory)/'state.json'; state_path.write_text('{}')
            proc=subprocess.run([sys.executable,'-m','f2','check','--task',str(folder/'task.json'),'--environment',str(folder/'environment.json'),
                                 '--contract',str(output),'--plan',str(folder/'normal_plan.json'),'--state',str(state_path)],cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(proc.returncode,3,proc.stderr)
            self.assertEqual(json.loads(proc.stdout)['decision'],'DEFER')

    def test_unavailable_observation_provider_blocks_before_any_command(self):
        from f2.runtime import GatedExecutor
        c=compile_contract(draft(),task(),environment()); f=RuntimeFilter(c,clock=lambda:100)
        s=state(c); s['capabilities'].remove('holding_confirmed:{object}')
        commands=[]
        r=GatedExecutor(f).run(plan(),lambda i:s,commands.append,lambda:True)
        self.assertEqual(r.decision,'DEFER'); self.assertEqual(commands,[])
        self.assertIn('OBSERVATION_PROVIDER_UNAVAILABLE',r.reason_codes)

    def test_ha5_zone_entry_is_not_an_object_manipulation_harm(self):
        e=environment(); e['hazard_rules']=[dict(id='zone',category='HA5',actions=['approach'],object='*',destination='*',fact='entered_zone',
            protected_target='area',protected_condition='no entry',harm_path='entry alone',exposure_condition='entry alone')]
        with self.assertRaises(ContractError): compile_contract(draft(),task(),e)

    def test_exponent_overflow_is_not_valid_json_number(self):
        with self.assertRaises(ValueError): parse_json('{"seconds":1e999}')

    def test_only_the_started_setup_process_is_terminated_even_on_error(self):
        from f2.owned_setup import OwnedSetupServer
        server = OwnedSetupServer([sys.executable, '-c', 'import time; time.sleep(30)'])
        with self.assertRaises(RuntimeError):
            with server:
                self.assertIsNone(server.process.poll())
                raise RuntimeError('setup failed')
        self.assertIsNotNone(server.process.poll())
        # An existing/shared PID is never accepted by this API.
        with self.assertRaises(ContractError): OwnedSetupServer(12345)


if __name__=='__main__': unittest.main()
