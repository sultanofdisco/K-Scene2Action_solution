import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from contract_fixtures import draft, environment, plan, state, task
from f2.contract import ContractError, ContractManager, compile_contract, load_contract
from f2.episode import validate_episode
from f2.json_input import load_json
from f2.runtime import GatedExecutor, RuntimeFilter, extract_plan

ROOT = Path(__file__).resolve().parents[1]


class SecurityBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.e, self.t = environment(), task()
        self.c = compile_contract(draft(), self.t, self.e)
        self.f = RuntimeFilter(self.c, clock=lambda: 100.0)

    def check(self, p=None, s=None):
        return self.f.inspect(p or plan(), s or state(self.c), step_gated=True)

    def test_block_precedes_unknown_and_never_returns_executable_plan(self):
        p, s = plan(), state(self.c)
        p['steps'][4]['destination'] = 'edge'
        del s['facts']['motion_safety_clear']
        r = self.check(p, s)
        self.assertEqual(r.decision, 'BLOCK')
        self.assertFalse(r.analysis_complete)
        self.assertIsNone(r.approved_plan)

    def test_model_refuse_or_hold_with_steps_does_not_execute(self):
        for decision in ('REFUSE', 'HOLD', 'PASS', None):
            p = plan(); p['decision'] = decision
            self.assertEqual(self.check(p).decision, 'DEFER')

    def test_unknown_action_and_unknown_fields_defer(self):
        for action in ('pull', 'PUSH', 'move_to_xyz', 'ignore_filters'):
            p = plan(); p['steps'][0]['action'] = action
            self.assertEqual(self.check(p).decision, 'DEFER')
        p = plan(); p['contract'] = {'approved': True}
        self.assertEqual(self.check(p).decision, 'DEFER')

    def test_neutral_extended_runpod_fields_preserved(self):
        from f2.runtime import NEUTRAL
        p = plan()
        for step in p['steps']:
            step.update(copy.deepcopy(NEUTRAL))
        r = self.check(p)
        self.assertEqual(r.decision, 'PASS')
        self.assertEqual(r.approved_plan, p)

    def test_non_neutral_attack_flags_do_not_get_ignored(self):
        for key, value in [('at_fraction', 0.5), ('motion_phase', 'midway'), ('resume_without_reset', True), ('turns', 20)]:
            p = plan(); p['steps'][4][key] = value
            self.assertEqual(self.check(p).decision, 'DEFER')

    def test_depth_steps_and_python_objects_bounded(self):
        p = plan(); p['steps'] *= 17
        self.assertEqual(self.check(p).decision, 'DEFER')
        p = plan(); p['reason'] = 'x' * 16385
        self.assertEqual(self.check(p).decision, 'DEFER')
        for value in (object(), {'source': []}, ['nested']):
            s = state(self.c); s['facts']['motion_safety_clear']['source'] = value
            self.assertEqual(self.check(s=s).decision, 'DEFER')
        nested = []
        for _ in range(18): nested = [nested]
        p = plan(); p['notes'] = nested
        self.assertEqual(self.check(p).decision, 'DEFER')

    def test_contract_malformed_reference_never_activates(self):
        for key in ('object', 'destination', 'outcome', 'count'):
            d = draft(); d['goals'][0][key] = {}
            with self.assertRaises(ContractError): compile_contract(d, self.t, self.e)

    def test_missing_safety_configuration_does_not_activate(self):
        for key in ('max_observation_age_s', 'max_rgbd_skew_s'):
            e = copy.deepcopy(self.e); del e['limits'][key]
            with self.assertRaises(ContractError): compile_contract(draft(), self.t, e)
        d = draft(); d['unresolved'] = ['unknown object fragility']
        with self.assertRaises(ContractError): compile_contract(d, self.t, self.e)

    def test_failed_setup_invalidates_active_and_is_not_cached(self):
        calls = []
        def mapper(payload, schema):
            calls.append(True)
            if len(calls) == 2: raise ContractError('offline')
            return draft()
        m = ContractManager(mapper)
        m.prepare(self.t, self.e)
        t = copy.deepcopy(self.t); t['task_id'] = 'changed'
        with self.assertRaises(ContractError): m.prepare(t, self.e)
        self.assertIsNone(m.active)
        m.prepare(t, self.e)
        self.assertEqual(len(calls), 3)

    def test_immutable_compiled_settings(self):
        with self.assertRaises(TypeError): self.c.goals['red_ball'] = ('edge', 'supported_placement', None)
        with self.assertRaises(TypeError): self.c.limits['max_observation_age_s'] = 10000
        self.e['actions']['release']['before'] = ['*']
        self.assertIn('INVALID_TRANSITION', self.check({'decision': 'EXECUTE', 'steps': [{'action': 'release', 'object': 'red_ball', 'destination': 'zone1'}]}).reason_codes)

    def test_stored_contract_revalidated_against_live_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'contract.json'
            self.c.save(path)
            self.assertEqual(load_contract(path, self.t, self.e).contract_id, self.c.contract_id)
            tampered = load_json(path); tampered['mapping']['goals'][0]['destination'] = 'zone2'
            path.write_text(json.dumps(tampered), encoding='utf-8')
            with self.assertRaises(ContractError): load_contract(path, self.t, self.e)

    def test_consistent_envelopes_only(self):
        p = plan()
        self.assertEqual(extract_plan({'plan': p, 'vlm': {'parsed_plan': p}}), p)
        q = copy.deepcopy(p); q['steps'][0]['destination'] = 'edge'
        with self.assertRaises(ContractError): extract_plan({'plan': p, 'model_plan': q})

    def test_guard_facts_must_follow_the_previous_completed_step(self):
        commands = []
        gate = GatedExecutor(self.f)
        def observe(index):
            return state(self.c, step_index=-1)  # old frame masquerading as latest
        r = gate.run(plan(), observe, lambda s: commands.append(s) or True, lambda: True)
        self.assertEqual(r.decision, 'DEFER')
        self.assertEqual(len(commands), 4)
        self.assertIn('EVIDENCE_MISSING_OR_STALE', r.reason_codes)

    def test_release_missing_support_prevents_open_and_latches(self):
        commands = []
        gate = GatedExecutor(self.f)
        def observe(index):
            s = state(self.c, step_index=index-1)
            if index == 6: del s['facts']['support_confirmed:red_ball:zone1']
            return s
        r = gate.run(plan(), observe, lambda s: commands.append(s) or True, lambda: False)
        self.assertEqual(r.decision, 'DEFER')
        self.assertEqual(len(commands), 6)
        self.assertFalse(r.stop_confirmed)
        self.assertIn('STOP_NOT_CONFIRMED', r.reason_codes)
        r = gate.run(plan(), observe, lambda s: commands.append(s) or True, lambda: True)
        self.assertIn('RECOVERY_REQUIRED', r.reason_codes)

    def test_episode_change_mid_plan_and_callback_failure_stop_followup(self):
        for fail in ('episode', 'result', 'exception'):
            commands, stops = [], []
            def observe(index):
                s = state(self.c, step_index=index-1)
                if fail == 'episode' and index == 2: s['episode_id'] = 'other'
                return s
            def execute(s):
                commands.append(s)
                if fail == 'exception': raise RuntimeError('controller timeout')
                return fail != 'result'
            r = GatedExecutor(self.f).run(plan(), observe, execute, lambda: stops.append(True) or True)
            self.assertEqual(r.decision, 'DEFER')
            self.assertLessEqual(len(commands), 2)
            self.assertEqual(stops, [True])

    def test_candidate_mutation_does_not_replace_approved_snapshot(self):
        p, commands = plan(), []
        def observe(index):
            if index == 1: p['steps'][4]['destination'] = 'off_table'
            obj, dest = ('red_ball', 'zone1') if index < 8 else ('blue_ball', 'zone2')
            return state(self.c, step_index=index-1, obj=obj, dest=dest)
        r = GatedExecutor(self.f).run(p, observe, lambda s: commands.append(s) or True, lambda: True)
        self.assertEqual(r.decision, 'PASS')
        self.assertEqual(commands[4]['destination'], 'zone1')

    def test_unauthorized_repeat_and_wait_budgets(self):
        t = copy.deepcopy(self.t); t['expected_goals'][0]['count'] = 1
        c = compile_contract(draft(t), t, self.e)
        p = plan(); p['steps'][8:8] = copy.deepcopy(p['steps'][:8])
        self.assertIn('INCOMPLETE_OR_REPEATED_GOAL', RuntimeFilter(c, clock=lambda:100).inspect(p, state(c), step_gated=True).reason_codes)
        t = copy.deepcopy(self.t); t['allowed_actions'].append('hold')
        c = compile_contract(draft(t), t, self.e)
        p = plan(); p['steps'].insert(4, {'action':'hold','object':'red_ball','destination':'none','seconds':10})
        self.assertIn('WAIT_BUDGET_EXCEEDED', RuntimeFilter(c, clock=lambda:100).inspect(p, state(c), step_gated=True).reason_codes)

    def test_reported_drop_plans_block_without_fabricating_actual_damage(self):
        cases = load_json(ROOT / 'examples/recorded_drop_plans.json')
        self.assertEqual(len(cases), 6)
        for case in cases:
            r = self.check(case['plan'])
            self.assertEqual(r.decision, 'BLOCK', case['case_id'])
            self.assertEqual(r.harmful_categories, [])

    def test_current_runpod_execute_plans_are_blocked_with_known_violations(self):
        cases = load_json(ROOT / 'examples/contract/recorded_runpod_plans.json')
        self.assertEqual(len(cases), 7)
        for case in cases:
            expected = 'BLOCK' if case['plan']['decision'] == 'EXECUTE' else 'DEFER'
            self.assertEqual(self.check(case['plan']).decision, expected, case['case_id'])

    def test_ha_categories_require_causal_rule_and_trusted_condition(self):
        for category in ('HA1', 'HA2', 'HA3', 'HA4', 'HA5', 'HA6', 'HA7', 'HA8'):
            e = copy.deepcopy(self.e)
            e['hazard_rules'] = [dict(id='risk', category=category, actions=['grasp'], object='red_ball', destination='*',
                                     fact='causal_risk', protected_target='person/object/organization',
                                     protected_condition='no injury/damage/loss', harm_path='test-specific cause -> exposure -> damage',
                                     exposure_condition='trusted test exposure')]
            c = compile_contract(draft(), self.t, e)
            f = RuntimeFilter(c, clock=lambda:100)
            s = state(c)
            self.assertEqual(f.inspect(plan(), s, step_gated=True).decision, 'DEFER')
            s['facts']['causal_risk'] = dict(value=True, source='safety_monitor', observed_at=100, step_index=-1, evidence='test concrete path')
            r = f.inspect(plan(), s, step_gated=True)
            self.assertEqual(r.decision, 'BLOCK')
            self.assertEqual(r.harmful_categories, [category])
            self.assertIn('harm_path', r.issues[-1].evidence)
            s['facts']['causal_risk']['value'] = False
            self.assertEqual(f.inspect(plan(), s, step_gated=True).decision, 'PASS')

    def test_third_environment_information_delivery(self):
        e = environment(('report',), ('customer',))
        e['environment_id'] = 'delivery'
        e['actions']['deliver'] = dict(before=['idle'], after='idle', effect='complete', role='publish',
                                       object_mode='new', destination_mode='goal', parameters={})
        t = task(('report',), ('customer',)); t['instruction'] = 'Deliver the approved report to the authorized customer'
        t['expected_goals'][0]['outcome'] = 'authorized_delivery'; t['allowed_actions'] = ['deliver']
        c = compile_contract(draft(t), t, e)
        p = dict(decision='EXECUTE', steps=[dict(action='deliver', object='report', destination='customer')])
        s = state(c)
        f = RuntimeFilter(c, clock=lambda:100)
        self.assertEqual(f.inspect(p, s, step_gated=True).decision, 'PASS')
        self.assertEqual(f.inspect_step(p['steps'][0], s, 0, binding=('episode-1','request-1')).decision, 'DEFER')
        s['facts']['disclosure_authorized:report:customer'] = dict(value=True,source='controller',observed_at=100,step_index=-1,evidence='access policy allows recipient')
        self.assertEqual(f.inspect_step(p['steps'][0], s, 0, binding=('episode-1','request-1')).decision, 'PASS')

    def test_protective_stop_is_not_task_completion_or_financial_harm(self):
        p = dict(decision='EXECUTE', steps=[dict(action='stop', object='none', destination='none')])
        s = state(self.c); s['facts'] = {}
        commands, stops = [], []
        r = GatedExecutor(self.f).run(p, lambda i:s, commands.append, lambda:stops.append(True) or True)
        self.assertEqual(r.decision, 'PASS')
        self.assertEqual(commands, [])
        self.assertEqual(stops, [True])
        self.assertEqual(r.harmful_categories, [])
        self.assertEqual(r.plan_kind, 'protective_stop')
        self.assertTrue(r.stop_confirmed)

    def test_controller_success_does_not_replace_final_object_outcome(self):
        commands=[]
        def observe(index):
            obj,dest=('red_ball','zone1') if index<8 else ('blue_ball','zone2')
            s=state(self.c,step_index=index-1,obj=obj,dest=dest)
            if index==16: s['facts']['outcome_confirmed:red_ball:zone1']['value']=False
            return s
        r=GatedExecutor(self.f).run(plan(),observe,lambda s:commands.append(s) or True,lambda:True)
        self.assertEqual(len(commands),16)
        self.assertEqual(r.decision,'BLOCK')
        self.assertIn('FINAL_OUTCOME_NOT_CONFIRMED',r.reason_codes)

    def test_registered_numeric_parameters_cannot_use_string_enums(self):
        e=copy.deepcopy(self.e)
        e['actions']['transfer']['parameters']={'at_fraction':{'enum':['halfway']}}
        with self.assertRaises(ContractError): compile_contract(draft(),self.t,e)

    def test_required_registered_parameter_must_be_present_and_non_neutral(self):
        e=copy.deepcopy(self.e)
        e['actions']['transfer']['parameters']={'at_fraction':{'min':1.0,'max':1.0,'required':True}}
        c=compile_contract(draft(),self.t,e)
        f=RuntimeFilter(c,clock=lambda:100)
        self.assertEqual(f.inspect(plan(),state(c),step_gated=True).decision,'DEFER')
        p=plan()
        for s in p['steps']:
            if s['action']=='transfer': s['at_fraction']=1.0
        self.assertEqual(f.inspect(p,state(c),step_gated=True).decision,'PASS')

    def test_excessive_condition_complexity_is_deferred_without_skipping_checks(self):
        e=copy.deepcopy(self.e)
        e['hazard_rules']=[dict(id=f'risk{i}',category='HA1',actions=self.t['allowed_actions'],object='*',destination='*',fact='risk_clear',
            protected_target='worker',protected_condition='no injury',harm_path='conditional collision exposure',exposure_condition='measured collision path') for i in range(256)]
        c=compile_contract(draft(),self.t,e)
        p=plan(); p['steps']*=2
        r=RuntimeFilter(c,clock=lambda:100).inspect(p,state(c),step_gated=True)
        self.assertEqual(r.decision,'DEFER')
        self.assertIn('CONDITION_PROCESSING_BUDGET_EXCEEDED',r.reason_codes)

    def test_new_request_id_cannot_repeat_an_already_consumed_episode(self):
        gate=GatedExecutor(self.f)
        request=['first']
        def observe(index):
            obj,dest=('red_ball','zone1') if index<8 else ('blue_ball','zone2')
            s=state(self.c,step_index=index-1,obj=obj,dest=dest)
            s['request_id']=request[0]
            return s
        self.assertEqual(gate.run(plan(),observe,lambda s:True,lambda:True).decision,'PASS')
        request[0]='second'
        r=gate.run(plan(),observe,lambda s:True,lambda:True)
        self.assertEqual(r.decision,'DEFER')
        self.assertIn('EPISODE_ALREADY_CONSUMED',r.reason_codes)

    def test_repetition_without_explicit_authority_is_deferred_not_claimed_as_damage(self):
        p=plan(); p['steps']*=2
        r=self.check(p)
        self.assertEqual(r.decision,'DEFER')
        self.assertIn('REPETITION_AUTHORITY_UNSPECIFIED',r.reason_codes)
        self.assertEqual(r.harmful_categories,[])


class EpisodeBindingTests(unittest.TestCase):
    def test_original_and_rotated_image_digests_are_distinct_and_verified(self):
        artifacts = {'rgb':b'original JPEG','depth':b'NPY bytes','camera_status':b'status JSON'}
        evidence = dict(schema='scene2action-fresh-episode-evidence-v1', case_id='case', model='qwen', reset_count=1,
                        reset_completed_at_unix=998, captured_at_unix=999, ground_truth_used_for_control=False,
                        **{k+'_sha256':hashlib.sha256(v).hexdigest() for k,v in artifacts.items()})
        expected = dict(case_id='case',model='qwen',reset_count=1,orientation_transform='ROTATE_CCW_90',
                        sent_image_sha256='rotated-image-digest',request_sha256='trusted-request-digest')
        record = dict(case_id='case',model_label='qwen',vlm={'parsed_plan':plan()},input=dict(
            episode_evidence=evidence,same_episode_camera_for_vlm=True,raw_image_sha256=evidence['rgb_sha256'],
            image_sha256=expected['sent_image_sha256'],request_sha256=expected['request_sha256'],camera_orientation_transform=expected['orientation_transform']))
        self.assertEqual(validate_episode(record,evidence,artifacts,expected,now_unix=1000,max_age_s=2), plan())
        for name in artifacts:
            wrong = dict(artifacts); wrong[name] = b'tampered'
            with self.assertRaises(ContractError): validate_episode(record,evidence,wrong,expected,now_unix=1000,max_age_s=2)
        with self.assertRaises(ContractError): validate_episode(record,evidence,artifacts,expected,now_unix=1002,max_age_s=2)


if __name__ == '__main__': unittest.main()
