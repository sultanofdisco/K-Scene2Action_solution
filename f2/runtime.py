"""Bounded CPU-only plan checks and mandatory, fail-closed staged execution."""
from __future__ import annotations

import copy
import json
import threading
import time
from collections import Counter
from dataclasses import dataclass

from .contract import ContractError, GUARDS, MAX_STEPS, MAX_PREDICATE_CHECKS, finite, fingerprint, snapshot
from .filter import Issue, Result, conclude

NEUTRAL = dict(seconds=None, turns=None, direction='none', motion_phase='none', at_fraction=None,
               pause_seconds=None, resume_without_reset=False, force_mode='none', limit_relation='none')


@dataclass(frozen=True)
class RuntimeResult(Result):
    contract_id: str = ''
    candidate_sha256: str | None = None
    requires_step_gate: bool = True
    completed_steps: int = 0
    stop_confirmed: bool | None = None


def extract_plan(record):
    """Select consistent recorded plans, never infer actions from prose."""
    record = snapshot(record)
    if not isinstance(record, dict):
        raise ContractError('INVALID_PLAN_RECORD')
    if 'steps' in record:
        return record
    plans = [record[k] for k in ('plan', 'model_plan') if k in record]
    if isinstance(record.get('vlm'), dict) and 'parsed_plan' in record['vlm']:
        plans.append(record['vlm']['parsed_plan'])
    if not plans or any(p != plans[0] for p in plans[1:]):
        raise ContractError('MISSING_OR_CONFLICTING_PLANS')
    return plans[0]


class RuntimeFilter:
    def __init__(self, contract, clock=time.monotonic):
        self.contract, self.clock = contract, clock
        self._hazards = tuple(json.loads(r) for r in contract.hazard_rules)
        self._hazards_by_action = {}
        for rule in self._hazards:
            causal = json.dumps({k: rule[k] for k in ('protected_target', 'protected_condition', 'harm_path', 'exposure_condition')}, ensure_ascii=False)
            for action in rule['actions']:
                self._hazards_by_action.setdefault(action, []).append((rule, causal))
        self._parameters = {name: json.loads(spec.parameters_json) for name, spec in contract.actions.items()}

    def result(self, issues, plan=None, completed=0, stop=None):
        kind = 'contract_plan'
        if (isinstance(plan, dict) and isinstance(plan.get('steps'), list) and len(plan['steps']) == 1
                and isinstance(plan['steps'][0], dict) and type(plan['steps'][0].get('action')) is str
                and plan['steps'][0]['action'] in self.contract.actions
                and self.contract.actions[plan['steps'][0]['action']].role == 'stop'):
            kind = 'protective_stop'
        r = conclude(issues, plan, kind)
        return RuntimeResult(r.decision, r.issues, r.approved_plan, r.plan_kind, self.contract.contract_id,
                             fingerprint(plan) if plan is not None else None, True, completed, stop)

    @staticmethod
    def add(issues, code, path='$', disposition='DEFER', category=None, evidence=None):
        issues.append(Issue(code, disposition, code.replace('_', ' ').lower(), path, category, evidence))

    def _state(self, state, issues, binding=None):
        if not isinstance(state, dict):
            self.add(issues, 'STATE_REQUIRED')
            return False
        allowed = {'bindings', 'episode_id', 'request_id', 'observed_at', 'rgb_stamp', 'depth_stamp', 'facts', 'capabilities'}
        if set(state) != allowed:
            self.add(issues, 'INVALID_STATE_FIELDS')
        if state.get('bindings') != self.contract.bindings_dict():
            self.add(issues, 'CONTEXT_MISMATCH')
        for key in ('episode_id', 'request_id'):
            if type(state.get(key)) is not str or not 0 < len(state[key]) <= 128:
                self.add(issues, 'EXECUTION_BINDING_REQUIRED', '$.' + key)
        if binding is not None and (state.get('episode_id'), state.get('request_id')) != binding:
            self.add(issues, 'EPISODE_OR_REQUEST_MISMATCH')
        now, observed = self.clock(), state.get('observed_at')
        age = self.contract.limits['max_observation_age_s']
        if not finite(now) or not finite(observed) or not 0 <= now - observed <= age:
            self.add(issues, 'STATE_STALE_OR_FUTURE')
        rgb, depth = state.get('rgb_stamp'), state.get('depth_stamp')
        if not finite(rgb) or not finite(depth) or abs(rgb - depth) > self.contract.limits['max_rgbd_skew_s']:
            self.add(issues, 'RGBD_UNSYNCHRONIZED')
        if not isinstance(state.get('facts'), dict):
            self.add(issues, 'FACTS_REQUIRED')
        elif len(state['facts']) > 512:
            self.add(issues, 'FACT_COUNT_BUDGET_EXCEEDED')
        else:
            for key, fact in state['facts'].items():
                if (not isinstance(fact, dict) or set(fact) != {'value', 'source', 'observed_at', 'step_index', 'evidence'}
                        or type(fact.get('value')) is not bool or type(fact.get('source')) is not str
                        or fact.get('source') not in self.contract.fact_sources or not finite(fact.get('observed_at'))
                        or type(fact.get('step_index')) is not int or fact['step_index'] < -1
                        or type(fact.get('evidence')) is not str or not fact['evidence'].strip()):
                    self.add(issues, 'INVALID_FACT_FIELDS', '$.facts.' + key)
        capabilities = state.get('capabilities')
        registered = {g for values in GUARDS.values() for g in values}
        if (not isinstance(capabilities, list) or len(capabilities) > 256
                or any(type(p) is not str or p not in registered for p in capabilities)
                or len(set(capabilities)) != len(capabilities)):
            self.add(issues, 'OBSERVATION_CAPABILITIES_REQUIRED')
        return not issues

    def _fact(self, state, key, issues, required_value=True, index=None, code='GUARD_FAILED', category=None, cause=None):
        fact = state.get('facts', {}).get(key)
        fields = {'value', 'source', 'observed_at', 'step_index', 'evidence'}
        now = self.clock()
        if (not isinstance(fact, dict) or set(fact) != fields or type(fact.get('value')) is not bool
                or type(fact.get('source')) is not str
                or fact.get('source') not in self.contract.fact_sources
                or not finite(fact.get('observed_at'))
                or not 0 <= now - fact['observed_at'] <= self.contract.limits['max_observation_age_s']
                or type(fact.get('step_index')) is not int
                or index is not None and fact['step_index'] != index - 1
                or type(fact.get('evidence')) is not str or not fact['evidence'].strip()):
            self.add(issues, 'EVIDENCE_MISSING_OR_STALE', '$.facts.' + key)
        elif fact['value'] != required_value:
            self.add(issues, code, '$.facts.' + key, 'BLOCK', category,
                     (cause + ' | ' if cause else '') + fact['evidence'])

    def _hazard(self, step, state, issues):
        for rule, causal in self._hazards_by_action.get(step['action'], ()):
            if (rule['object'] not in ('*', step['object'])
                    or rule['destination'] not in ('*', step['destination'])):
                continue
            # A true hazardous condition blocks; unknown does not mean clear.
            self._fact(state, rule['fact'], issues, False, code='HAZARD_' + rule['id'], category=rule['category'], cause=causal)

    def _format(self, plan, issues):
        if (not isinstance(plan, dict) or not {'steps', 'decision'} <= set(plan)
                or not set(plan) <= {'steps', 'decision', 'reason', 'notes', 'scene_text'}):
            self.add(issues, 'INVALID_PLAN_FIELDS')
            return None
        if plan['decision'] != 'EXECUTE':
            self.add(issues, 'MODEL_NOT_EXECUTABLE')
            return None
        for key in ('reason', 'notes'):
            if key in plan and type(plan[key]) is not str:
                self.add(issues, 'INVALID_EXPLANATION')
        if 'scene_text' in plan and (not isinstance(plan['scene_text'], list) or any(type(s) is not str for s in plan['scene_text'])):
            self.add(issues, 'INVALID_SCENE_TEXT')
        steps = plan['steps']
        if not isinstance(steps, list) or not 1 <= len(steps) <= MAX_STEPS:
            self.add(issues, 'PLAN_STEP_BUDGET_EXCEEDED')
            return None
        invalid = False
        for i, s in enumerate(steps):
            path = f'$.steps[{i}]'
            if (not isinstance(s, dict) or not {'action', 'object', 'destination'} <= set(s)
                    or any(type(s.get(k)) is not str for k in ('action', 'object', 'destination'))):
                self.add(issues, 'INVALID_STEP', path)
                invalid = True
                continue
            action = s['action']
            if action not in self.contract.actions:
                self.add(issues, 'UNSUPPORTED_ACTION', path)
                invalid = True
                continue
            if action not in self.contract.allowed_actions and self.contract.actions[action].role != 'stop':
                self.add(issues, 'ACTION_NOT_AUTHORIZED', path, 'BLOCK')
            goal = self.contract.goals.get(s['object'])
            if goal and s['destination'] != 'none' and s['destination'] != goal[0]:
                self.add(issues, 'GOAL_MISMATCH', path, 'BLOCK')
            if s['object'] != 'none' and s['object'] not in self.contract.objects:
                self.add(issues, 'OBJECT_NOT_AUTHORIZED', path, 'BLOCK')
            if not set(s) <= {'action', 'object', 'destination'} | set(NEUTRAL):
                self.add(issues, 'UNKNOWN_PARAMETER', path)
                invalid = True
            for key, rule in self._parameters[action].items():
                if rule.get('required') and (key not in s or type(s[key]) is type(NEUTRAL[key]) and s[key] == NEUTRAL[key]):
                    self.add(issues, 'REQUIRED_PARAMETER_MISSING', path + '.' + key)
                    invalid = True
            for key in set(s) & set(NEUTRAL):
                v, neutral = s[key], NEUTRAL[key]
                if type(v) is type(neutral) and v == neutral:
                    continue
                rule = self._parameters[action].get(key)
                if key in ('seconds', 'pause_seconds') and self.contract.actions[action].role == 'wait':
                    if not finite(v) or v <= 0:
                        self.add(issues, 'INVALID_WAIT_PARAMETER', path)
                        invalid = True
                    if rule is None:
                        continue
                if rule is None:
                    self.add(issues, 'UNSUPPORTED_NONNEUTRAL_PARAMETER', path + '.' + key)
                    invalid = True
                elif 'min' in rule:
                    if not finite(v):
                        self.add(issues, 'INVALID_NUMERIC_PARAMETER', path)
                        invalid = True
                    elif not rule['min'] <= v <= rule['max']:
                        self.add(issues, 'PARAMETER_LIMIT_EXCEEDED', path, 'BLOCK')
                elif not any(type(v) is type(item) and v == item for item in rule['enum']):
                    self.add(issues, 'PARAMETER_ENUM_NOT_AUTHORIZED', path, 'BLOCK')
        return None if invalid else steps

    def inspect(self, candidate, observation, *, step_gated=False):
        issues = []
        try:
            plan, state = snapshot(candidate), snapshot(observation)
        except (ValueError, TypeError, OverflowError, RecursionError) as exc:
            self.add(issues, 'INPUT_INVALID_OR_BUDGET_EXCEEDED', evidence=str(exc))
            return self.result(issues)
        valid_state = self._state(state, issues)
        steps = self._format(plan, issues)
        if steps is None:
            return self.result(issues, plan)
        checks = sum(len(self._hazards_by_action.get(s['action'], ())) + len(GUARDS[self.contract.actions[s['action']].role]) for s in steps)
        if checks > MAX_PREDICATE_CHECKS:
            self.add(issues, 'CONDITION_PROCESSING_BUDGET_EXCEEDED')
            return self.result(issues, plan)
        stop_only = len(steps) == 1 and self.contract.actions[steps[0]['action']].role == 'stop'
        if not step_gated:
            self.add(issues, 'STAGE_GATE_REQUIRED')
        if stop_only:
            if steps[0]['object'] != 'none' or steps[0]['destination'] != 'none':
                self.add(issues, 'INVALID_PROTECTIVE_STOP')
            return self.result(issues, plan)
        if valid_state:
            capabilities = set(state['capabilities'])
            if 'outcome_confirmed:{object}:{destination}' not in capabilities:
                self.add(issues, 'OBSERVATION_PROVIDER_UNAVAILABLE', '$.capabilities', evidence='outcome_confirmed:{object}:{destination}')
            for action in {s['action'] for s in steps}:
                for guard in GUARDS[self.contract.actions[action].role]:
                    if guard not in capabilities:
                        self.add(issues, 'OBSERVATION_PROVIDER_UNAVAILABLE', '$.capabilities', evidence=guard)
            physical = any(self.contract.actions[s['action']].role in {'motion', 'carry', 'manipulate', 'release', 'controlled_release', 'wait', 'resume'} for s in steps)
            if physical:
                self._fact(state, 'controller_ready', issues, code='CONTROLLER_NOT_READY')
                self._fact(state, 'initial_gripper_empty', issues, code='INITIAL_GRIPPER_NOT_EMPTY')
            if any(self.contract.actions[s['action']].role in {'motion', 'carry', 'manipulate'} for s in steps):
                self._fact(state, 'motion_safety_clear', issues, code='MOTION_NOT_CLEARED')
        phase, focus, destination, holding = 'idle', None, None, False
        completed, counts, wait = Counter(), Counter(), 0.0
        for i, s in enumerate(steps):
            path, action = f'$.steps[{i}]', s['action']
            spec = self.contract.actions[action]
            obj, dest = s['object'], s['destination']
            counts[action] += 1
            cap = self.contract.limits['max_action_occurrences'].get(action)
            if cap is not None and counts[action] > cap:
                self.add(issues, 'ACTION_REPEAT_BUDGET_EXCEEDED', path, 'BLOCK')
            if obj != 'none' and obj not in self.contract.objects:
                self.add(issues, 'OBJECT_NOT_AUTHORIZED', path, 'BLOCK')
            if dest != 'none' and dest not in self.contract.destinations:
                self.add(issues, 'DESTINATION_NOT_AUTHORIZED', path, 'BLOCK')
            if spec.object_mode == 'new':
                if obj not in self.contract.goals:
                    self.add(issues, 'TASK_OBJECT_MISMATCH', path, 'BLOCK')
            elif spec.object_mode == 'focus' and (focus is None or obj != focus):
                self.add(issues, 'OBJECT_FOCUS_MISMATCH', path, 'BLOCK')
            elif spec.object_mode == 'none' and obj != 'none':
                self.add(issues, 'UNEXPECTED_OBJECT', path, 'BLOCK')
            goal = self.contract.goals.get(obj)
            if dest != 'none' and goal and dest != goal[0]:
                self.add(issues, 'GOAL_MISMATCH', path, 'BLOCK')
            if spec.destination_mode == 'goal' and (goal is None or dest != goal[0]):
                self.add(issues, 'GOAL_MISMATCH', path, 'BLOCK')
            if spec.destination_mode == 'focus' and (destination is None or dest not in (destination, 'none')):
                self.add(issues, 'DESTINATION_FOCUS_MISMATCH', path, 'BLOCK')
            if spec.destination_mode == 'none' and dest != 'none':
                self.add(issues, 'UNEXPECTED_DESTINATION', path, 'BLOCK')
            if phase == 'stopped' or '*' not in spec.before and phase not in spec.before:
                self.add(issues, 'INVALID_TRANSITION', path, 'BLOCK')
            if spec.role == 'stop':
                phase = 'stopped'
                if i != len(steps) - 1:
                    self.add(issues, 'ACTION_AFTER_STOP', path, 'BLOCK')
            elif spec.role == 'wait':
                seconds = s.get('seconds')
                pause = s.get('pause_seconds')
                if seconds is not None and pause is not None and seconds != pause:
                    self.add(issues, 'AMBIGUOUS_WAIT_PARAMETERS', path)
                duration = seconds if seconds is not None else pause if pause is not None else 1.0
                wait += duration
                if duration > self.contract.limits['max_step_wait_s'] or wait > self.contract.limits['max_total_wait_s']:
                    self.add(issues, 'WAIT_BUDGET_EXCEEDED', path, 'BLOCK')
            if valid_state:
                self._hazard(s, state, issues)
            if spec.effect == 'focus':
                focus, destination = obj, None
            elif spec.effect == 'hold':
                holding = True
            elif spec.effect == 'destination':
                if not holding:
                    self.add(issues, 'CARRY_WITHOUT_LOGICAL_GRASP', path, 'BLOCK')
                destination = dest
            elif spec.effect == 'complete':
                actual_destination = dest if spec.role in {'publish', 'configure'} else destination
                if goal is None or actual_destination != goal[0] or not holding and spec.role in {'release', 'controlled_release'}:
                    self.add(issues, 'OUTCOME_NOT_REACHED', path, 'BLOCK')
                completed[obj] += 1
                holding = False
            elif spec.effect == 'clear':
                if holding:
                    self.add(issues, 'CLEAR_WHILE_HOLDING', path, 'BLOCK')
                focus, destination = None, None
            if spec.after != 'same':
                phase = spec.after
        for obj, (_, _, count) in self.contract.goals.items():
            if completed[obj] < 1 or count is not None and completed[obj] != count:
                self.add(issues, 'INCOMPLETE_OR_REPEATED_GOAL', '$.steps', 'BLOCK')
            elif completed[obj] > 1 and count is None:
                self.add(issues, 'REPETITION_AUTHORITY_UNSPECIFIED', '$.steps')
        if holding or phase not in {'idle', 'stopped'}:
            self.add(issues, 'INCOMPLETE_TERMINAL_STATE', '$.steps', 'BLOCK')
        return self.result(issues, plan)

    def inspect_step(self, step, observation, index, *, binding, destination=None):
        issues = []
        try:
            step, state = snapshot(step), snapshot(observation)
            valid = self._state(state, issues, binding)
            spec = self.contract.actions[step['action']]
            if valid:
                dest = step['destination'] if step['destination'] != 'none' else destination
                for template in GUARDS[spec.role]:
                    if template not in state['capabilities']:
                        self.add(issues, 'OBSERVATION_PROVIDER_UNAVAILABLE', '$.capabilities', evidence=template)
                        continue
                    key = template.format(object=step['object'], destination=dest)
                    # Object observations must follow the immediately previous
                    # completed stage, not an earlier initial/static assertion.
                    object_guard = template.startswith(('holding_', 'support_', 'containment_'))
                    self._fact(state, key, issues, index=index if object_guard else None)
                self._hazard(step, state, issues)
        except (ValueError, TypeError, KeyError, OverflowError, RecursionError):
            self.add(issues, 'STEP_STATE_INVALID')
        return self.result(issues)

    def inspect_outcomes(self, observation, index, *, binding):
        issues = []
        try:
            state = snapshot(observation)
            if self._state(state, issues, binding):
                if 'outcome_confirmed:{object}:{destination}' not in state['capabilities']:
                    self.add(issues, 'OBSERVATION_PROVIDER_UNAVAILABLE')
                else:
                    for obj, (dest, outcome, count) in self.contract.goals.items():
                        self._fact(state, f'outcome_confirmed:{obj}:{dest}', issues, index=index, code='FINAL_OUTCOME_NOT_CONFIRMED')
        except (ValueError, TypeError, KeyError, OverflowError, RecursionError):
            self.add(issues, 'OUTCOME_STATE_INVALID')
        return self.result(issues)


class GatedExecutor:
    """Single owner in-process gate; each callback must wait for execution result.

    A stop callback must cancel active goals AND confirm a stopped controller.
    This class does not infer that a ROS2 stop marker physically stopped a robot.
    Use a durable host idempotency store for restarts/multiple worker processes.
    """
    def __init__(self, runtime, max_requests=4096):
        self.runtime = runtime
        self._lock = threading.Lock()
        self._used = set()
        self._episodes = set()
        self.max_requests = max_requests
        self._latched = False

    def run(self, candidate, observe, execute_step, cancel_and_confirm_stop):
        if not self._lock.acquire(blocking=False):
            return self.runtime.result([Issue('EXECUTOR_BUSY', 'DEFER', 'another plan is executing', '$')])
        completed, plan, stop_confirmed = 0, None, None
        try:
            if self._latched:
                return self.runtime.result([Issue('RECOVERY_REQUIRED', 'DEFER', 'trusted recovery and a new gate are required', '$')])
            plan = snapshot(candidate)
            initial = snapshot(observe(0))
            result = self.runtime.inspect(plan, initial, step_gated=True)
            if result.decision != 'PASS':
                return result
            protective_stop = result.plan_kind == 'protective_stop'
            binding = (initial['episode_id'], initial['request_id'])
            if binding in self._used or len(self._used) >= self.max_requests:
                return self.runtime.result([Issue('REPLAY_OR_REQUEST_CAPACITY', 'DEFER', 'duplicate request or capacity reached', '$')], plan)
            if binding[0] in self._episodes:
                return self.runtime.result([Issue('EPISODE_ALREADY_CONSUMED', 'DEFER', 'new request IDs do not authorize repeating an episode', '$')], plan)
            self._used.add(binding)  # consume before any side effect, including failures
            self._episodes.add(binding[0])
            if fingerprint(plan) != result.candidate_sha256:
                raise ContractError('CANDIDATE_CHANGED')
            destination = None
            for index, step in enumerate(plan['steps']):
                state = snapshot(observe(index))
                result = self.runtime.inspect_step(step, state, index, binding=binding, destination=destination)
                if result.decision != 'PASS':
                    self._latched = True
                    stopped = cancel_and_confirm_stop() is True
                    issues = list(result.issues)
                    if not stopped:
                        issues.append(Issue('STOP_NOT_CONFIRMED', 'DEFER', 'physical stop remains unknown', '$'))
                    return self.runtime.result(issues, plan, completed, stopped)
                spec = self.runtime.contract.actions[step['action']]
                if spec.role == 'stop':
                    if cancel_and_confirm_stop() is not True:
                        raise ContractError('STOP_NOT_CONFIRMED')
                    stop_confirmed = True
                elif execute_step(copy.deepcopy(step)) is not True:
                    raise ContractError('EXECUTION_RESULT_NOT_SUCCESSFUL')
                completed += 1
                if spec.effect == 'destination':
                    destination = step['destination']
                elif spec.effect == 'clear':
                    destination = None
            if not protective_stop:
                final = self.runtime.inspect_outcomes(observe(len(plan['steps'])), len(plan['steps']), binding=binding)
                if final.decision != 'PASS':
                    self._latched = True
                    stopped = cancel_and_confirm_stop() is True
                    issues = list(final.issues)
                    if not stopped:
                        issues.append(Issue('STOP_NOT_CONFIRMED', 'DEFER', 'physical stop remains unknown', '$'))
                    return self.runtime.result(issues, plan, completed, stopped)
            return self.runtime.result([], plan, completed, stop_confirmed)
        except Exception as exc:
            # Boundary errors must not cause subsequent commands or a fallback.
            self._latched = True
            try:
                stopped = cancel_and_confirm_stop() is True
            except Exception:
                stopped = False
            issues = [Issue('EXECUTION_BOUNDARY_FAILURE', 'DEFER', type(exc).__name__ + ': ' + str(exc), '$')]
            if not stopped:
                issues.append(Issue('STOP_NOT_CONFIRMED', 'DEFER', 'physical stop remains unknown', '$'))
            return self.runtime.result(issues, plan, completed, stopped)
        finally:
            self._lock.release()
