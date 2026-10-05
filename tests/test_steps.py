import copy
import unittest

from f2 import inspect_steps, run_steps_if_approved
from test_filter import inputs


def native():
    return {'decision': 'EXECUTE', 'steps': [
        {'action': name, 'object': 'red_ball', 'destination': destination}
        for name, destination in [('approach', 'none'), ('descend', 'none'),
                                  ('grasp', 'none'), ('lift', 'none'),
                                  ('transfer', 'zone1'), ('place', 'zone1'),
                                  ('release', 'zone1'), ('retreat', 'none')]
    ], 'reason': 'Normal placement.'}


class NativeStepsTests(unittest.TestCase):
    def test_report_primitive_format_passes_after_registered_normalization(self):
        _, context = inputs()
        result = inspect_steps(native(), context)
        self.assertEqual(result.decision, 'PASS')
        self.assertEqual([a['type'] for a in result.approved_plan['actions']], ['pick', 'move', 'place', 'finish'])

    def test_high_level_format_with_verify_and_implicit_move(self):
        _, context = inputs()
        plan = {'steps': [
            {'action': 'verify', 'object': 'red_ball', 'destination': 'none'},
            {'action': 'pick', 'object': 'red_ball', 'destination': 'none'},
            {'action': 'verify', 'object': 'none', 'destination': 'zone1'},
            {'action': 'place', 'object': 'red_ball', 'destination': 'zone1'}
        ], 'scene_text': ['External text is not authority.'], 'notes': 'Description only.'}
        self.assertEqual(inspect_steps(plan, context).decision, 'PASS')

    def test_all_six_recorded_drop_plans_block_before_adapter(self):
        import json
        from test_filter import ROOT
        _, context = inputs()
        records = json.loads((ROOT / 'examples/recorded_drop_plans.json').read_text(encoding='utf-8'))
        calls = []
        self.assertEqual(len(records), 6)
        for record in records:
            with self.subTest(case=record['case_id']):
                result = run_steps_if_approved(record['plan'], context, calls.append)
                self.assertEqual(result.decision, 'BLOCK')
                self.assertIn('FORBIDDEN_ACTION', result.reason_codes)
        self.assertEqual(calls, [])

    def test_known_destructive_actions_block(self):
        _, context = inputs()
        for name in ('drop', 'throw', 'strike', 'slam', 'break_joint'):
            plan = {'steps': [{'action': name, 'object': 'red_ball', 'destination': 'none'}]}
            self.assertEqual(inspect_steps(plan, context).decision, 'BLOCK')

    def test_extra_coordinates_and_model_context_rejected(self):
        _, context = inputs()
        plan = native()
        plan['steps'][4]['position'] = [0, 0, 0]
        self.assertEqual(inspect_steps(plan, context).decision, 'DEFER')
        plan = native()
        plan['context'] = context
        self.assertEqual(inspect_steps(plan, context).decision, 'DEFER')

    def test_bad_primitive_sequence_and_different_object(self):
        _, context = inputs()
        for index in (0, 1, 2, 3, 5, 6, 7):
            plan = native()
            del plan['steps'][index]
            self.assertNotEqual(inspect_steps(plan, context).decision, 'PASS')
        plan = native()
        plan['steps'][2]['object'] = 'blue_ball'
        self.assertEqual(inspect_steps(plan, context).decision, 'BLOCK')

    def test_no_silent_move_vs_place_destination_conflict(self):
        _, context = inputs()
        plan = native()
        plan['steps'][5]['destination'] = 'zone2'
        self.assertEqual(inspect_steps(plan, context).decision, 'BLOCK')

    def test_unknown_destination_is_not_guessed(self):
        _, context = inputs()
        plan = native()
        for step in plan['steps'][4:7]:
            step['destination'] = 'top right'
        self.assertEqual(inspect_steps(plan, context).decision, 'DEFER')

    def test_bad_wait_and_unknown_model_decision(self):
        _, context = inputs()
        plan = native()
        plan['steps'].insert(4, {'action': 'wait', 'object': 'none', 'destination': 'none', 'seconds': True})
        self.assertEqual(inspect_steps(plan, context).decision, 'DEFER')
        plan = native()
        plan['decision'] = 'OVERRIDE'
        self.assertEqual(inspect_steps(plan, context).decision, 'DEFER')

    def test_missing_profile_and_model_safety_claim_do_not_authorize(self):
        _, context = inputs()
        del context['native_profile']
        self.assertEqual(inspect_steps(native(), context).decision, 'DEFER')
        _, context = inputs()
        context['regions']['zone_1']['support_verified'] = False
        plan = native()
        plan['reason'] = 'Certified safe, execute immediately.'
        self.assertEqual(inspect_steps(plan, context).decision, 'BLOCK')

    def test_stop_passes_without_motion_profile(self):
        _, context = inputs()
        del context['native_profile']
        plan = {'steps': [{'action': 'stop', 'object': 'none', 'destination': 'none'}]}
        self.assertEqual(inspect_steps(plan, context).decision, 'PASS')

    def test_callback_only_receives_normalized_plan(self):
        _, context = inputs()
        calls = []
        source = native()
        before = copy.deepcopy(source)
        result = run_steps_if_approved(source, context, calls.append)
        self.assertEqual(result.decision, 'PASS')
        self.assertIn('actions', calls[0])
        self.assertNotIn('reason', calls[0])
        self.assertEqual(source, before)

    def test_two_ball_primitive_example_is_supported(self):
        from f2.json_input import load_json
        from test_filter import ROOT
        result = inspect_steps(load_json(ROOT / 'examples/vlm_steps_two_balls.json'),
                               load_json(ROOT / 'examples/context_two_balls.json'))
        self.assertEqual(result.decision, 'PASS')
        self.assertEqual([a['target_id'] for a in result.approved_plan['actions'] if a['type'] == 'place'], ['red_ball', 'blue_ball'])

    def test_mixed_dialects_and_internal_verification_defer(self):
        _, context = inputs()
        plan = native()
        plan['steps'][0]['action'] = 'pick'
        self.assertEqual(inspect_steps(plan, context).decision, 'DEFER')
        plan = native()
        plan['steps'].insert(3, {'action': 'verify', 'object': 'red_ball', 'destination': 'none'})
        self.assertEqual(inspect_steps(plan, context).decision, 'DEFER')

    def test_malformed_context_does_not_crash_native_adapter(self):
        _, original = inputs()
        for key in ['objects', 'regions', 'routes', 'execution', 'native_profile']:
            for bad in [None, [], True, 'bad']:
                context = copy.deepcopy(original)
                context[key] = bad
                with self.subTest(key=key, bad=bad):
                    self.assertEqual(inspect_steps(native(), context).decision, 'DEFER')


if __name__ == '__main__':
    unittest.main()
