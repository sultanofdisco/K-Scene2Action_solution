"""Strict adapter for the recorded VLM steps/action/object/destination contract.

Explanations never supply authority, coordinates, observations, or limits.
Only normalized actions are forwarded; legacy raw-step executors must not be
called in parallel with this gate.
"""
from __future__ import annotations

import copy
from typing import Any, Callable

from .filter import BINDINGS, Issue, Result, conclude, inspect_plan, number

FORBIDDEN = {'drop', 'throw', 'strike', 'slam', 'break_joint'}
SUPPORTED = {'verify', 'pick', 'move', 'place', 'wait', 'stop',
             'approach', 'descend', 'grasp', 'lift', 'transfer', 'release', 'retreat'}
PROFILE_NUMBERS = {'grip_force_n', 'speed_m_s', 'pick_duration_s', 'move_duration_s',
                   'place_duration_s', 'verify_duration_s'}


def inspect_steps(plan: Any, context: Any) -> Result:
    """Normalize supported source steps and inspect against trusted host context.

    The host associates this response with the live request. It must not take
    the context from model output or from an attacker-controlled attachment.
    """
    issues: list[Issue] = []

    def add(code, message, path='$', disposition='DEFER'):
        issues.append(Issue(code, disposition, message, path))

    root_fields = {'steps', 'decision', 'reason', 'notes', 'scene_text'}
    if not isinstance(plan, dict) or not set(plan) <= root_fields or 'steps' not in plan:
        add('INVALID_SOURCE_PLAN', 'Provide the model plan itself, not a report wrapper or context.')
        return conclude(issues)
    if 'decision' in plan and plan['decision'] != 'EXECUTE':
        add('MODEL_DID_NOT_PROPOSE_EXECUTION', 'This response is not an executable proposal.')
    for key in ('reason', 'notes'):
        if key in plan and (not isinstance(plan[key], str) or len(plan[key]) > 16384):
            add('INVALID_SOURCE_PLAN', 'Invalid explanatory metadata.', '$.' + key)
    if 'scene_text' in plan and (not isinstance(plan['scene_text'], list)
                               or len(plan['scene_text']) > 256
                               or any(not isinstance(s, str) or len(s) > 16384 for s in plan['scene_text'])):
        add('INVALID_SOURCE_PLAN', 'Invalid scene text metadata.', '$.scene_text')
    steps = plan.get('steps')
    if not isinstance(steps, list) or not steps or len(steps) > 256:
        add('INVALID_SOURCE_PLAN', 'Expected 1 to 256 steps.')
        return conclude(issues)
    for i, step in enumerate(steps):
        path = f'$.steps[{i}]'
        if not isinstance(step, dict) or not isinstance(step.get('action'), str):
            add('INVALID_SOURCE_PLAN', 'Invalid step object.', path)
            continue
        action = step['action']
        if action in FORBIDDEN:
            add('FORBIDDEN_ACTION', 'Destructive/drop primitives are outside the approved placement contract.', path, 'BLOCK')
        elif action not in SUPPORTED:
            add('UNSUPPORTED_ACTION', 'No defined semantics for this action.', path)
        expected = {'action', 'object', 'destination'} | ({'seconds'} if action == 'wait' else set())
        if set(step) != expected or any(not isinstance(step.get(k), str) for k in ('object', 'destination')):
            add('INVALID_SOURCE_PLAN', 'Missing or unknown step fields.', path)
        if action == 'wait' and not number(step.get('seconds')):
            add('INVALID_SOURCE_PLAN', 'Wait seconds must be finite and positive.', path)
    if issues:
        return conclude(issues)
    if not isinstance(context, dict):
        add('INVALID_CONTEXT', 'Trusted host context is required.')
        return conclude(issues)
    context, steps = copy.deepcopy(context), copy.deepcopy(steps)
    for key in ('objects', 'regions', 'routes', 'execution'):
        if not isinstance(context.get(key), dict):
            add('INVALID_CONTEXT', 'Missing trusted state dictionary.', '$.context.' + key)
    if issues:
        return conclude(issues)
    normalized = {'schema_version': '1.0', **{key: context.get(key) for key in BINDINGS}, 'actions': []}
    if steps == [{'action': 'stop', 'object': 'none', 'destination': 'none'}]:
        normalized['actions'] = [{'type': 'stop'}]
        return inspect_plan(normalized, context)
    profile = context.get('native_profile')
    if (not isinstance(profile, dict) or any(not number(profile.get(k)) for k in PROFILE_NUMBERS)
            or not isinstance(profile.get('destination_aliases'), dict)
            or not isinstance(profile.get('routes'), dict)
            or type(profile.get('max_steps')) is not int or profile['max_steps'] <= 0):
        add('INVALID_PROFILE', 'Register the execution profile, aliases, route bindings and step budget.')
        return conclude(issues)
    if len(steps) > profile['max_steps']:
        add('SOURCE_STEP_LIMIT_EXCEEDED', 'Source steps exceed the configured budget.', disposition='BLOCK')
    primitive = any(s['action'] in {'approach', 'descend', 'grasp', 'lift', 'release', 'retreat'} for s in steps)
    if primitive and any(s['action'] in ('pick', 'move') for s in steps):
        add('MIXED_DIALECT', 'Do not mix high-level and primitive manipulation dialects.')
        return conclude(issues)
    actions = normalized['actions']
    active, phase, destination = None, 'idle', None

    def target_matches(obj, path):
        if obj != active:
            add('SOURCE_OBJECT_MISMATCH', 'Primitive object differs from the active object.', path, 'BLOCK')

    def resolve_destination(value, path):
        dest = profile['destination_aliases'].get(value)
        if not isinstance(dest, str) or dest not in context.get('regions', {}):
            add('DESTINATION_UNKNOWN', 'Destination is not explicitly registered; no alias guessing.', path)
            return None
        return dest

    def carry(dest, path):
        route = profile['routes'].get(active, {})
        route_id = route.get(dest) if isinstance(route, dict) else None
        if not isinstance(route_id, str):
            add('ROUTE_UNKNOWN', 'No registered object-to-destination route.', path)
            return
        actions.append({'type': 'move', 'destination_id': dest, 'route_id': route_id,
                        'speed_m_s': profile['speed_m_s'], 'duration_s': profile['move_duration_s']})

    def pick(obj):
        actions.append({'type': 'pick', 'target_id': obj, 'grip_force_n': profile['grip_force_n'],
                        'duration_s': profile['pick_duration_s']})

    def place(obj, dest):
        actions.append({'type': 'place', 'target_id': obj, 'destination_id': dest,
                        'duration_s': profile['place_duration_s']})

    for i, step in enumerate(steps):
        name, obj, dest = step['action'], step['object'], step['destination']
        path = f'$.steps[{i}]'
        if name == 'verify':
            # Counts as observation time only; never turns a missing fact true.
            if phase not in ('idle', 'carrying'):
                add('UNSUPPORTED_OBSERVATION_PHASE', 'Verification timing inside a primitive manipulation is undefined.', path)
            if (obj != 'none' and obj not in context.get('objects', {})) or (
                    dest != 'none' and resolve_destination(dest, path) is None):
                add('VERIFY_TARGET_UNKNOWN', 'Observation target is unregistered.', path)
            actions.append({'type': 'wait', 'duration_s': profile['verify_duration_s']})
            continue
        if name == 'wait':
            if obj != 'none' or dest != 'none' or phase not in ('idle', 'carrying'):
                add('INVALID_SOURCE_SEQUENCE', 'Wait semantics are undefined in this phase.', path, 'BLOCK')
            actions.append({'type': 'wait', 'duration_s': step['seconds']})
            continue
        if name in ('pick', 'approach'):
            if phase != 'idle' or obj == 'none' or dest != 'none':
                add('INVALID_SOURCE_SEQUENCE', 'Pick/approach requires idle, a target and no destination.', path, 'BLOCK')
            active, destination = obj, None
            phase = 'approached' if name == 'approach' else 'carrying'
            if name == 'pick':
                pick(obj)
        elif name in ('descend', 'grasp', 'lift'):
            target_matches(obj, path)
            expected_phase = {'descend': 'approached', 'grasp': 'descended', 'lift': 'grasped'}[name]
            if phase != expected_phase or dest != 'none':
                add('INVALID_SOURCE_SEQUENCE', 'Primitive order or destination is invalid.', path, 'BLOCK')
            phase = {'descend': 'descended', 'grasp': 'grasped', 'lift': 'carrying'}[name]
            if name == 'lift':
                pick(obj)
        elif name in ('move', 'transfer'):
            target_matches(obj, path)
            if phase != 'carrying' or destination is not None:
                add('INVALID_SOURCE_SEQUENCE', 'Only one carry route is supported per placement.', path, 'BLOCK')
            destination = resolve_destination(dest, path)
            if destination:
                carry(destination, path)
        elif name == 'place':
            target_matches(obj, path)
            if phase != 'carrying':
                add('INVALID_SOURCE_SEQUENCE', 'Place requires a carrying phase.', path, 'BLOCK')
            resolved = resolve_destination(dest, path)
            if destination is not None and resolved != destination:
                add('DESTINATION_CONFLICT', 'Transfer and placement destinations differ.', path, 'BLOCK')
            if destination is None and resolved:
                carry(resolved, path)
            destination = resolved
            if resolved:
                place(obj, resolved)
            # Primitive form must explicitly include release and retreat.
            phase = 'placed' if primitive else 'idle'
            if not primitive:
                active, destination = None, None
        elif name == 'release':
            target_matches(obj, path)
            if phase != 'placed' or resolve_destination(dest, path) != destination:
                add('UNSUPPORTED_RELEASE', 'Release requires prior supported placement at the same destination.', path, 'BLOCK')
            phase = 'released'
        elif name == 'retreat':
            target_matches(obj, path)
            if phase != 'released' or dest != 'none':
                add('INVALID_SOURCE_SEQUENCE', 'Retreat requires released phase and no new destination.', path, 'BLOCK')
            phase, active, destination = 'idle', None, None
        elif name == 'stop':
            if i != len(steps) - 1 or phase != 'idle' or obj != 'none' or dest != 'none':
                add('INVALID_SOURCE_SEQUENCE', 'Task stop must follow completed placement.', path, 'BLOCK')
    if phase != 'idle':
        add('INCOMPLETE_SOURCE_SEQUENCE', 'Primitive sequence leaves an unfinished manipulation.', disposition='BLOCK')
    actions.append({'type': 'finish'})
    if issues:
        return conclude(issues)
    return inspect_plan(normalized, context)


def run_steps_if_approved(plan: Any, context: Any, adapter: Callable[[dict], Any]) -> Result:
    result = inspect_steps(plan, context)
    if result.decision == 'PASS':
        adapter(copy.deepcopy(result.approved_plan))
    return result
