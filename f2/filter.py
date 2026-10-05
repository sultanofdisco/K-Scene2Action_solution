from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Issue:
    code: str
    disposition: str
    message: str
    path: str
    category: str | None = None
    evidence: str | None = None


@dataclass(frozen=True)
class Result:
    decision: str
    issues: tuple[Issue, ...]
    approved_plan: dict | None = None
    plan_kind: str = 'task'

    @property
    def reason_codes(self) -> list[str]:
        return sorted({issue.code for issue in self.issues})

    @property
    def harmful_categories(self) -> list[str]:
        return sorted({issue.category for issue in self.issues if issue.category})

    @property
    def analysis_complete(self) -> bool:
        return not any(issue.disposition == 'DEFER' for issue in self.issues)

    def to_dict(self) -> dict:
        return {**asdict(self), 'reason_codes': self.reason_codes,
                'harmful_categories': self.harmful_categories,
                'analysis_complete': self.analysis_complete}


def conclude(issues: list[Issue], plan: dict | None = None, kind='task') -> Result:
    decision = ('BLOCK' if any(i.disposition == 'BLOCK' for i in issues)
                else 'DEFER' if issues else 'PASS')
    return Result(decision, tuple(issues), copy.deepcopy(plan) if decision == 'PASS' else None, kind)


def number(value: Any, positive=True) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value) and (value > 0 if positive else value >= 0)
    except OverflowError:
        return False


def identifier(value: Any) -> bool:
    return isinstance(value, str) and 0 < len(value) <= 128 and value.strip() == value


FIELDS = {
    'pick': {'type', 'target_id', 'grip_force_n', 'duration_s'},
    'move': {'type', 'destination_id', 'route_id', 'speed_m_s', 'duration_s'},
    'place': {'type', 'target_id', 'destination_id', 'duration_s'},
    'wait': {'type', 'duration_s'}, 'finish': {'type'}, 'stop': {'type'},
}
PLAN_FIELDS = {'schema_version', 'request_id', 'context_id', 'environment_version', 'actions'}
LIMITS = {'max_actions', 'max_duration_s', 'max_speed_m_s',
          'max_grip_force_n', 'max_hold_time_s'}
BINDINGS = ('request_id', 'context_id', 'environment_version')


def inspect_plan(plan: Any, context: Any) -> Result:
    """Check a snapshot of a normalized plan against independently trusted context.

    PASS approves this symbolic plan only; it is not proof of trajectory safety.
    BLOCK outranks DEFER, but uncertainty remains visible in analysis_complete.
    """
    issues: list[Issue] = []

    def add(code, message, path, disposition='BLOCK', category=None, evidence=None):
        issues.append(Issue(code, disposition, message, path, category, evidence))

    if not isinstance(plan, dict) or set(plan) != PLAN_FIELDS:
        add('INVALID_PLAN', 'Expected the exact normalized plan fields.', '$', 'DEFER')
        return conclude(issues)
    plan = copy.deepcopy(plan)
    actions = plan.get('actions')
    if plan['schema_version'] != '1.0' or not isinstance(actions, list) or not actions:
        add('INVALID_PLAN', 'Unsupported version or empty/non-list actions.', '$', 'DEFER')
        return conclude(issues)
    # Fixed parser budget is independent from operator-supplied limits.
    if len(actions) > 256:
        add('PLAN_TOO_LARGE', 'At most 256 normalized actions can be inspected.', '$.actions')
        return conclude(issues)
    for i, action in enumerate(actions):
        path = f'$.actions[{i}]'
        if not isinstance(action, dict) or not isinstance(action.get('type'), str):
            add('INVALID_PLAN', 'Action must be an object with a string type.', path, 'DEFER')
            continue
        name = action['type']
        if name not in FIELDS:
            add('UNSUPPORTED_ACTION', 'Action has no defined execution semantics.', path, 'DEFER')
            continue
        if set(action) != FIELDS[name]:
            add('INVALID_PLAN', 'Missing or unexpected action fields.', path, 'DEFER')
        for key, value in action.items():
            if key.endswith(('_s', '_n')) and not number(value):
                add('INVALID_PLAN', 'Expected a finite positive numeric value.', path + '.' + key, 'DEFER')
            if key.endswith('_id') and not identifier(value):
                add('INVALID_PLAN', 'Expected an explicit registered identifier.', path + '.' + key, 'DEFER')
    if issues:
        return conclude(issues)
    if not isinstance(context, dict):
        add('INVALID_CONTEXT', 'Trusted context is required.', '$.context', 'DEFER')
        return conclude(issues)
    context = copy.deepcopy(context)
    for key in BINDINGS:
        if not identifier(plan.get(key)) or not identifier(context.get(key)) or plan[key] != context[key]:
            add('CONTEXT_MISMATCH', 'Plan belongs to another or missing context.', '$.' + key, 'DEFER')
    if context.get('schema_version') != '1.0' or context.get('context_status') != 'READY':
        add('CONTEXT_NOT_READY', 'Context is not a validated current snapshot.', '$.context', 'DEFER')
    limits = context.get('limits')
    if (not isinstance(limits, dict) or not LIMITS <= set(limits)
            or any(not number(limits[k]) for k in LIMITS)
            or type(limits['max_actions']) is not int):
        add('INVALID_LIMITS', 'All required positive limits must be configured.', '$.context.limits', 'DEFER')
        return conclude(issues)
    task = context.get('task')
    goals = task.get('placements') if isinstance(task, dict) and set(task) == {'placements'} else [task]
    task_fields = {'target_id', 'goal_region_id', 'placement_count'}
    if (not isinstance(goals, list) or not 1 <= len(goals) <= 256
            or any(not isinstance(goal, dict) or set(goal) != task_fields
                   or not identifier(goal.get('target_id')) or not identifier(goal.get('goal_region_id'))
                   or type(goal.get('placement_count')) is not int or not 1 <= goal['placement_count'] <= 256
                   for goal in goals)):
        add('INVALID_TASK', 'Register explicit targets, destinations and positive placement counts.', '$.context.task', 'DEFER')
        return conclude(issues)
    targets = {goal['target_id']: goal for goal in goals}
    if len(targets) != len(goals):
        add('INVALID_TASK', 'Duplicate target goals are ambiguous.', '$.context.task', 'DEFER')
        return conclude(issues)
    for key in ('objects', 'regions', 'routes', 'execution'):
        if not isinstance(context.get(key), dict):
            add('INVALID_CONTEXT', 'Missing registered state dictionary.', '$.context.' + key, 'DEFER')
    if any(i.code == 'INVALID_CONTEXT' for i in issues):
        return conclude(issues)
    execution = context['execution']
    protective_stop = actions == [{'type': 'stop'}]
    if protective_stop:
        # No completion claim; actual emergency stop must bypass this software.
        return conclude(issues, plan, 'protective_stop')
    for flag, expected in (('robot_error', False), ('safety_enabled', True)):
        if type(execution.get(flag)) is not bool:
            add('EXECUTION_STATE_UNKNOWN', 'Controller state must be confirmed.', '$.context.execution.' + flag, 'DEFER')
        elif execution[flag] != expected:
            add('EXECUTION_NOT_READY', 'Normal motion is not allowed in this controller state.', '$.context.execution.' + flag)
    if execution.get('state') != 'ready' or 'holding' not in execution or execution['holding'] is not None:
        add('INITIAL_STATE_UNKNOWN', 'Task must start ready with an empty gripper.', '$.context.execution', 'DEFER')
    if not identifier(execution.get('location_id')):
        add('INITIAL_STATE_UNKNOWN', 'Registered initial pose is required.', '$.context.execution.location_id', 'DEFER')
    if len(actions) > limits['max_actions']:
        add('ACTION_COUNT_EXCEEDED', 'Plan exceeds configured action count.', '$.actions')

    def require_bool(record, key, path, false_code, category=None):
        if type(record.get(key)) is not bool:
            add('EVIDENCE_MISSING', 'Required state is unconfirmed.', path + '.' + key, 'DEFER')
        elif record[key] is False:
            add(false_code, 'Trusted state rejects this operation.', path + '.' + key,
                category=category, evidence=f'{path}.{key}=false')

    def check_route(route_id, source, destination, path):
        route = context['routes'].get(route_id) if identifier(route_id) else None
        if not isinstance(route, dict):
            add('ROUTE_UNKNOWN', 'No registered route evidence.', path, 'DEFER')
            return
        if route.get('source_id') != source or route.get('destination_id') != destination:
            add('ROUTE_BINDING_MISMATCH', 'Route does not describe this source and destination.', path, 'DEFER')
            return
        assessment, hazards = route.get('assessment'), route.get('hazards')
        if assessment == 'CLEAR' and hazards == []:
            return
        if assessment == 'HAZARD' and isinstance(hazards, list) and hazards:
            check_hazards(hazards, path, 'ROUTE_HAZARD')
        else:
            add('ROUTE_UNASSESSED', 'Route has unknown or inconsistent evidence.', path, 'DEFER')

    def check_hazards(hazards, path, code):
        if not isinstance(hazards, list) or len(hazards) > 256:
            add('EVIDENCE_MISSING', 'Hazards must be a bounded list.', path, 'DEFER')
            return
        for hazard in hazards:
            if (not isinstance(hazard, dict) or not isinstance(hazard.get('category'), str)
                    or hazard['category'] not in {f'HA{i}' for i in range(1, 9)}
                    or not isinstance(hazard.get('evidence'), str) or not hazard['evidence'].strip()):
                add('EVIDENCE_MISSING', 'Hazard requires trusted category and causal evidence.', path, 'DEFER')
            else:
                add(code, 'Trusted assessment identifies a harmful-action candidate.', path,
                    category=hazard['category'], evidence=hazard['evidence'])

    held, location, elapsed, hold_time, finished = None, execution.get('location_id'), 0.0, 0.0, False
    placements = {}
    for i, action in enumerate(actions):
        path, name = f'$.actions[{i}]', action['type']
        if finished:
            add('ACTION_AFTER_TERMINATION', 'No action may follow termination.', path)
        duration = action.get('duration_s', 0)
        elapsed += duration
        if held is not None or name == 'pick':
            hold_time += duration
        if hold_time > limits['max_hold_time_s']:
            add('HOLD_TIME_LIMIT_EXCEEDED', 'Planned holding exceeds configured bound.', path)
        if name == 'pick':
            if held is not None:
                add('INVALID_SEQUENCE', 'Cannot pick while already holding.', path)
            target = action['target_id']
            if target not in targets:
                add('TASK_TARGET_MISMATCH', 'Target differs from the approved task.', path)
            obj = context['objects'].get(target)
            if not isinstance(obj, dict):
                add('OBJECT_UNKNOWN', 'Target has no trusted observation.', path, 'DEFER')
            else:
                for key in ('authorized', 'observed', 'graspable'):
                    require_bool(obj, key, '$.context.objects.' + target, 'OBJECT_NOT_ALLOWED')
                check_hazards(obj.get('hazards', []), path, 'OBJECT_HAZARD')
                cap = obj.get('max_grip_force_n')
                if not number(cap):
                    add('EVIDENCE_MISSING', 'Object-specific grip limit is required.', path, 'DEFER')
                elif action['grip_force_n'] > min(cap, limits['max_grip_force_n']):
                    add('GRIP_LIMIT_EXCEEDED', 'Grip force exceeds the configured object/robot limit.', path)
                check_route(obj.get('approach_route_id'), location, target, path)
            held, location = target, target
        elif name == 'move':
            if held is None:
                add('INVALID_SEQUENCE', 'Carry requires a held target.', path)
            destination = action['destination_id']
            if held not in targets or destination != targets[held]['goal_region_id']:
                add('TASK_GOAL_MISMATCH', 'Destination differs from the approved task.', path)
            check_route(action['route_id'], location, destination, path)
            if action['speed_m_s'] > limits['max_speed_m_s']:
                add('SPEED_LIMIT_EXCEEDED', 'Speed exceeds configured limit.', path)
            location = destination
        elif name == 'place':
            destination, target = action['destination_id'], action['target_id']
            if held != target or location != destination:
                add('INVALID_SEQUENCE', 'Place requires the same held object at its destination.', path)
            if target not in targets:
                add('TASK_TARGET_MISMATCH', 'Placed object differs from the task.', path)
            if target not in targets or destination != targets[target]['goal_region_id']:
                add('TASK_GOAL_MISMATCH', 'Placement differs from the task.', path)
            region = context['regions'].get(destination)
            if not isinstance(region, dict):
                add('REGION_UNKNOWN', 'Placement region is unregistered.', path, 'DEFER')
            else:
                require_bool(region, 'placement_allowed', '$.context.regions.' + destination, 'PLACEMENT_NOT_ALLOWED')
                for key in ('support_verified', 'stable_verified'):
                    require_bool(region, key, '$.context.regions.' + destination, 'UNSUPPORTED_PLACEMENT')
                check_hazards(region.get('hazards', []), path, 'PLACEMENT_HAZARD')
            held, hold_time = None, 0.0
            placements[target] = placements.get(target, 0) + 1
            if target in targets and placements[target] > targets[target]['placement_count']:
                add('PLACEMENT_COUNT_EXCEEDED', 'Repeated placement is outside the approved task.', path)
        elif name == 'finish':
            expected_counts = {key: value['placement_count'] for key, value in targets.items()}
            if held is not None or placements != expected_counts:
                add('INCOMPLETE_TASK', 'Finish requires completed placement and empty gripper.', path)
            finished = True
        elif name == 'stop':
            add('STOP_REQUIRES_SEPARATE_PLAN', 'Use a standalone protective stop, not partial task completion.', path)
            finished = True
    if not finished:
        add('MISSING_TERMINATION', 'Task plan requires explicit finish.', '$.actions')
    if elapsed > limits['max_duration_s']:
        add('DURATION_LIMIT_EXCEEDED', 'Plan exceeds the configured duration bound.', '$.actions')
    return conclude(issues, plan)


def run_if_approved(plan: Any, context: Any, adapter: Callable[[dict], Any]) -> Result:
    """The callback receives only the exact validated snapshot; never raw input."""
    result = inspect_plan(plan, context)
    if result.decision == 'PASS':
        adapter(copy.deepcopy(result.approved_plan))
    return result
