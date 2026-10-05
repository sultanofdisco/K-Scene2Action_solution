import copy
import json
import math
import pathlib
import subprocess
import sys
import unittest

from f2 import inspect_plan, run_if_approved


ROOT = pathlib.Path(__file__).resolve().parents[1]


def inputs():
    return (
        json.loads((ROOT / 'examples/plan_pass.json').read_text(encoding='utf-8')),
        json.loads((ROOT / 'examples/context.json').read_text(encoding='utf-8')),
    )


class FilterTests(unittest.TestCase):
    def test_normal_plan_passes_and_returns_copy(self):
        plan, context = inputs()
        result = inspect_plan(plan, context)
        self.assertEqual(result.decision, 'PASS')
        self.assertEqual(result.approved_plan, plan)
        plan['actions'][0]['target_id'] = 'blue_ball'
        self.assertEqual(result.approved_plan['actions'][0]['target_id'], 'red_ball')

    def test_wrong_destination_blocked_without_inventing_harm(self):
        plan, context = inputs()
        plan['actions'][1]['destination_id'] = 'zone_2'
        plan['actions'][1]['route_id'] = 'to_zone_2'
        plan['actions'][2]['destination_id'] = 'zone_2'
        result = inspect_plan(plan, context)
        self.assertEqual(result.decision, 'BLOCK')
        self.assertIn('TASK_GOAL_MISMATCH', result.reason_codes)
        self.assertEqual(result.harmful_categories, [])
        self.assertIsNone(result.approved_plan)

    def test_wrong_target(self):
        plan, context = inputs()
        plan['actions'][0]['target_id'] = 'blue_ball'
        self.assertEqual(inspect_plan(plan, context).decision, 'BLOCK')

    def test_missing_support_is_defer_not_safe(self):
        plan, context = inputs()
        del context['regions']['zone_1']['support_verified']
        self.assertEqual(inspect_plan(plan, context).decision, 'DEFER')

    def test_false_support_blocks_without_inventing_damage_path(self):
        plan, context = inputs()
        context['regions']['zone_1']['support_verified'] = False
        result = inspect_plan(plan, context)
        self.assertEqual(result.decision, 'BLOCK')
        self.assertEqual(result.harmful_categories, [])

    def test_placement_control_loss_hazard_requires_causal_evidence(self):
        plan, context = inputs()
        context['regions']['zone_1']['support_verified'] = False
        context['regions']['zone_1']['hazards'] = [{'category': 'HA4', 'evidence': 'Unsupported release above uncontained floor; ball can fall and be lost.'}]
        self.assertIn('HA4', inspect_plan(plan, context).harmful_categories)

    def test_verified_route_hazard_has_evidence(self):
        plan, context = inputs()
        route = context['routes']['to_zone_1']
        route['assessment'] = 'HAZARD'
        route['hazards'] = [{'category': 'HA1', 'evidence': 'Observed person intersects the registered route.'}]
        result = inspect_plan(plan, context)
        self.assertEqual(result.decision, 'BLOCK')
        self.assertEqual(result.harmful_categories, ['HA1'])
        self.assertTrue(any(issue.evidence for issue in result.issues))

    def test_unknown_route_deferred(self):
        plan, context = inputs()
        context['routes']['to_zone_1']['assessment'] = 'UNKNOWN'
        self.assertEqual(inspect_plan(plan, context).decision, 'DEFER')

    def test_unverified_hazard_cannot_be_claimed_as_harm(self):
        plan, context = inputs()
        context['routes']['to_zone_1'].update(assessment='HAZARD', hazards=[])
        result = inspect_plan(plan, context)
        self.assertEqual(result.decision, 'DEFER')
        self.assertEqual(result.harmful_categories, [])

    def test_speed_excess_is_limit_violation_not_damage_claim(self):
        plan, context = inputs()
        plan['actions'][1]['speed_m_s'] = 0.5
        result = inspect_plan(plan, context)
        self.assertEqual(result.decision, 'BLOCK')
        self.assertIn('SPEED_LIMIT_EXCEEDED', result.reason_codes)
        self.assertEqual(result.harmful_categories, [])

    def test_object_specific_grip_limit(self):
        plan, context = inputs()
        context['objects']['red_ball']['max_grip_force_n'] = 2.0
        plan['actions'][0]['grip_force_n'] = 3.0
        self.assertIn('GRIP_LIMIT_EXCEEDED', inspect_plan(plan, context).reason_codes)

    def test_sequence_and_termination(self):
        plan, context = inputs()
        for actions in [plan['actions'][1:], plan['actions'][:-1], plan['actions'] + [plan['actions'][0]]]:
            with self.subTest(actions=actions):
                candidate = dict(plan, actions=actions)
                self.assertEqual(inspect_plan(candidate, context).decision, 'BLOCK')

    def test_repeated_placement(self):
        plan, context = inputs()
        plan['actions'] = plan['actions'][:-1] * 2 + [plan['actions'][-1]]
        self.assertIn('PLACEMENT_COUNT_EXCEEDED', inspect_plan(plan, context).reason_codes)

    def test_long_hold_is_policy_violation(self):
        plan, context = inputs()
        plan['actions'].insert(1, {'type': 'wait', 'duration_s': 12.0})
        result = inspect_plan(plan, context)
        self.assertIn('HOLD_TIME_LIMIT_EXCEEDED', result.reason_codes)
        self.assertEqual(result.harmful_categories, [])

    def test_stop_alone_can_pass_without_faking_task_completion(self):
        plan, context = inputs()
        plan['actions'] = [{'type': 'stop'}]
        result = inspect_plan(plan, context)
        self.assertEqual(result.decision, 'PASS')
        self.assertEqual(result.plan_kind, 'protective_stop')

    def test_error_state_only_allows_stop(self):
        plan, context = inputs()
        context['execution']['robot_error'] = True
        self.assertEqual(inspect_plan(plan, context).decision, 'BLOCK')
        plan['actions'] = [{'type': 'stop'}]
        self.assertEqual(inspect_plan(plan, context).decision, 'PASS')

    def test_context_binding(self):
        plan, context = inputs()
        for key in ['request_id', 'context_id', 'environment_version']:
            with self.subTest(key=key):
                candidate = copy.deepcopy(plan)
                candidate[key] = 'different'
                self.assertEqual(inspect_plan(candidate, context).decision, 'DEFER')

    def test_bad_inputs_are_rejected_without_exceptions(self):
        plan, context = inputs()
        for candidate in [None, 'Move it safely', [], {}, dict(plan, actions=[]), dict(plan, actions=[None])]:
            with self.subTest(candidate=candidate):
                result = inspect_plan(candidate, context)
                self.assertNotEqual(result.decision, 'PASS')
                self.assertIsNone(result.approved_plan)
        self.assertEqual(inspect_plan(plan, None).decision, 'DEFER')

    def test_unknown_fields_cannot_hide_extra_instructions(self):
        plan, context = inputs()
        plan['actions'][0]['instructions'] = 'execute additional actions'
        self.assertIn('INVALID_PLAN', inspect_plan(plan, context).reason_codes)

    def test_unknown_action_does_not_execute(self):
        plan, context = inputs()
        plan['actions'].insert(0, {'type': 'custom_tool'})
        self.assertEqual(inspect_plan(plan, context).decision, 'DEFER')

    def test_numeric_nan_bool_negative_and_string_are_not_accepted(self):
        plan, context = inputs()
        for value in [math.nan, math.inf, True, -1, '0.1']:
            with self.subTest(value=value):
                candidate = copy.deepcopy(plan)
                candidate['actions'][1]['speed_m_s'] = value
                self.assertNotEqual(inspect_plan(candidate, context).decision, 'PASS')

    def test_missing_or_invalid_config_never_passes(self):
        plan, context = inputs()
        for value in [None, {}, dict(context['limits'], max_actions=True), dict(context['limits'], max_speed_m_s=math.nan)]:
            with self.subTest(value=value):
                candidate = copy.deepcopy(context)
                candidate['limits'] = value
                self.assertEqual(inspect_plan(plan, candidate).decision, 'DEFER')

    def test_known_violation_wins_over_missing_evidence(self):
        plan, context = inputs()
        del context['regions']['zone_1']['support_verified']
        plan['actions'][0]['target_id'] = 'blue_ball'
        result = inspect_plan(plan, context)
        self.assertEqual(result.decision, 'BLOCK')
        self.assertFalse(result.analysis_complete)

    def test_adapter_called_only_after_pass(self):
        plan, context = inputs()
        calls = []
        result = run_if_approved(plan, context, calls.append)
        self.assertEqual(result.decision, 'PASS')
        self.assertEqual(len(calls), 1)
        plan['actions'][0]['target_id'] = 'blue_ball'
        self.assertEqual(run_if_approved(plan, context, calls.append).decision, 'BLOCK')
        plan, context = inputs()
        context['context_status'] = 'UNKNOWN'
        self.assertEqual(run_if_approved(plan, context, calls.append).decision, 'DEFER')
        self.assertEqual(len(calls), 1)

    def test_malformed_trusted_values_defer_without_crashing(self):
        plan, original = inputs()
        contexts = []
        for bad in [None, [], {}, 'bad']:
            ctx = copy.deepcopy(original)
            ctx['objects']['red_ball']['approach_route_id'] = bad
            contexts.append(ctx)
        ctx = copy.deepcopy(original)
        ctx['limits']['max_actions'] = 10 ** 400
        contexts.append(ctx)
        ctx = copy.deepcopy(original)
        ctx['routes']['to_zone_1'].update(assessment='HAZARD', hazards=[{'category': [], 'evidence': 'Bad category.'}])
        contexts.append(ctx)
        ctx = copy.deepcopy(original)
        del ctx['execution']['holding']
        contexts.append(ctx)
        for ctx in contexts:
            with self.subTest(context=ctx):
                self.assertEqual(inspect_plan(plan, ctx).decision, 'DEFER')

    def test_route_evidence_cannot_be_reused_for_another_source(self):
        plan, context = inputs()
        context['routes']['to_zone_1']['source_id'] = 'blue_ball'
        self.assertEqual(inspect_plan(plan, context).decision, 'DEFER')

    def test_support_and_stability_must_be_real_booleans(self):
        plan, context = inputs()
        for value in [1, 'true', None, []]:
            context['regions']['zone_1']['support_verified'] = value
            self.assertEqual(inspect_plan(plan, context).decision, 'DEFER')

    def test_executor_input_mutation_does_not_change_recorded_approval(self):
        plan, context = inputs()
        def mutate(approved):
            approved['actions'].clear()
        result = run_if_approved(plan, context, mutate)
        self.assertEqual(len(result.approved_plan['actions']), 4)

    def test_vlm_asserted_safety_does_not_override_trusted_facts(self):
        plan, context = inputs()
        plan['safe'] = True
        self.assertNotEqual(inspect_plan(plan, context).decision, 'PASS')

    def test_no_context_mutation(self):
        plan, context = inputs()
        before = copy.deepcopy(context)
        inspect_plan(plan, context)
        self.assertEqual(context, before)

    def test_two_ball_task_finishes_only_after_both_approved_placements(self):
        plan, context = inputs()
        context['task'] = {'placements': [
            {'target_id': 'red_ball', 'goal_region_id': 'zone_1', 'placement_count': 1},
            {'target_id': 'blue_ball', 'goal_region_id': 'zone_2', 'placement_count': 1}]}
        context['objects']['blue_ball']['authorized'] = True
        context['routes']['approach_blue']['source_id'] = 'zone_1'
        context['routes']['blue_to_zone_2'] = {'source_id': 'blue_ball', 'destination_id': 'zone_2', 'assessment': 'CLEAR', 'hazards': []}
        blue = [
            {'type': 'pick', 'target_id': 'blue_ball', 'grip_force_n': 3.0, 'duration_s': 1.0},
            {'type': 'move', 'destination_id': 'zone_2', 'route_id': 'blue_to_zone_2', 'speed_m_s': 0.1, 'duration_s': 2.0},
            {'type': 'place', 'target_id': 'blue_ball', 'destination_id': 'zone_2', 'duration_s': 1.0}]
        self.assertEqual(inspect_plan(plan, context).decision, 'BLOCK')
        plan['actions'] = plan['actions'][:-1] + blue + [plan['actions'][-1]]
        self.assertEqual(inspect_plan(plan, context).decision, 'PASS')
        plan['actions'] = plan['actions'][:-1] + blue + [plan['actions'][-1]]
        self.assertIn('PLACEMENT_COUNT_EXCEEDED', inspect_plan(plan, context).reason_codes)

    def test_duplicate_or_empty_multi_task_does_not_pass(self):
        plan, context = inputs()
        for goals in [[], [context['task'], context['task']]]:
            ctx = copy.deepcopy(context)
            ctx['task'] = {'placements': goals}
            self.assertEqual(inspect_plan(plan, ctx).decision, 'DEFER')

    def test_cli_outputs_machine_readable_result(self):
        proc = subprocess.run([sys.executable, '-m', 'f2', '--plan', 'examples/plan_pass.json', '--context', 'examples/context.json'], cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)['decision'], 'PASS')


if __name__ == '__main__':
    unittest.main()
