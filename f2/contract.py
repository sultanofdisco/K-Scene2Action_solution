"""Trusted initialization only. LLM output selects data, never executable code."""
from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from .json_input import parse_json

POLICY_VERSION = 'ha1-ha8-v1.0-stage-gates-v1'
REGISTRY_VERSION = 'f2-declarative-actions-v1'
MAX_STEPS = 256
MAX_PREDICATE_CHECKS = 4096
MAX_NODES = 20000
ROLES = frozenset({'motion', 'carry', 'release', 'controlled_release', 'manipulate',
                   'publish', 'configure', 'wait', 'stop', 'verify', 'resume'})
EFFECTS = frozenset({'none', 'focus', 'hold', 'destination', 'complete', 'clear'})
OUTCOMES = frozenset({'supported_placement', 'contained_disposal', 'authorized_delivery', 'authorized_update'})
GUARDS = MappingProxyType({
    'motion': ('controller_ready', 'motion_safety_clear'),
    'carry': ('controller_ready', 'motion_safety_clear', 'holding_confirmed:{object}'),
    'manipulate': ('controller_ready', 'motion_safety_clear'),
    'release': ('controller_ready', 'support_confirmed:{object}:{destination}'),
    'controlled_release': ('controller_ready', 'containment_confirmed:{object}:{destination}'),
    'publish': ('disclosure_authorized:{object}:{destination}',),
    'configure': ('change_authorized:{object}:{destination}',),
    'wait': ('controller_ready',), 'resume': ('resume_authorized', 'controller_ready'),
    'verify': (), 'stop': (),
    'completed_outcome': ('outcome_confirmed:{object}:{destination}',),
})


class ContractError(ValueError):
    pass


def finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def snapshot(value):
    """Bound nesting/bytes/node count before copying untrusted Python or JSON input."""
    pending, nodes = [(value, 0)], 0
    while pending:
        node, depth = pending.pop()
        nodes += 1
        if depth > 16 or nodes > MAX_NODES:
            raise ContractError('INPUT_BUDGET_EXCEEDED')
        if isinstance(node, dict):
            if len(node) > MAX_NODES:
                raise ContractError('INPUT_BUDGET_EXCEEDED')
            if any(type(k) is not str for k in node):
                raise ContractError('NON_STRING_KEY')
            pending.extend((v, depth + 1) for v in node.values())
        elif isinstance(node, list):
            if len(node) > MAX_NODES:
                raise ContractError('INPUT_BUDGET_EXCEEDED')
            pending.extend((v, depth + 1) for v in node)
        elif type(node) not in (str, int, float, bool, type(None)):
            raise ContractError('NON_JSON_VALUE')
        if isinstance(node, str) and len(node) > 16384:
            raise ContractError('STRING_BUDGET_EXCEEDED')
    try:
        return parse_json(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')))
    except (ValueError, OverflowError, RecursionError) as exc:
        raise ContractError('INVALID_JSON: ' + str(exc)) from exc


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def policy_fingerprint():
    return fingerprint({'version': POLICY_VERSION, 'guards': dict(GUARDS),
                        'ha_categories': [f'HA{i}' for i in range(1, 9)],
                        'ha5_zone_entry_excluded': True})


def exact(record, fields, label):
    if not isinstance(record, dict) or set(record) != set(fields):
        raise ContractError('INVALID_' + label)


def names(values, label):
    if (not isinstance(values, list) or len(values) > 256
            or any(type(v) is not str or not v or len(v) > 128 or v.strip() != v for v in values)
            or len(set(values)) != len(values)):
        raise ContractError('INVALID_' + label)
    return frozenset(values)


@dataclass(frozen=True)
class ActionSpec:
    before: frozenset
    after: str
    effect: str
    role: str
    object_mode: str
    destination_mode: str
    parameters_json: str


@dataclass(frozen=True)
class CompiledContract:
    contract_id: str
    bindings: tuple
    actions: object
    allowed_actions: frozenset
    objects: frozenset
    destinations: frozenset
    goals: object
    limits: object
    fact_sources: frozenset
    hazard_rules: tuple
    artifact_json: str

    def bindings_dict(self):
        return dict(self.bindings)

    def save(self, path):
        """Atomic replacement in a trusted local directory; not an authenticated store."""
        path = Path(path)
        import os
        import tempfile
        handle, temporary = tempfile.mkstemp(prefix='f2-', suffix='.tmp', dir=path.parent)
        try:
            with os.fdopen(handle, 'w', encoding='utf-8') as stream:
                stream.write(self.artifact_json)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def mapping_schema(task, environment):
    def enum(values):
        return {'type': 'string', 'enum': values}
    return {'type': 'object', 'additionalProperties': False,
            'required': ['schema_version', 'goals', 'allowed_actions', 'unresolved'],
            'properties': {
                'schema_version': {'const': 'f2-mapping-v1'},
                'goals': {'type': 'array', 'minItems': 1, 'maxItems': 256, 'items': {
                    'type': 'object', 'additionalProperties': False,
                    'required': ['object', 'destination', 'outcome', 'count'],
                    'properties': {'object': enum(task['authorized_objects']),
                                   'destination': enum(task['authorized_destinations']),
                                   'outcome': enum(sorted(OUTCOMES)),
                                   'count': {'anyOf': [{'type': 'null'}, {'type': 'integer', 'minimum': 1, 'maximum': 256}]}}}},
                'allowed_actions': {'type': 'array', 'minItems': 1, 'maxItems': 256, 'uniqueItems': True,
                                    'items': enum(task['allowed_actions'])},
                'unresolved': {'type': 'array', 'maxItems': 256, 'items': {'type': 'string', 'maxLength': 1024}}}}


def compile_contract(mapping, task, environment, *, contrasts=(), provenance=None):
    try:
        contract = _compile_contract(mapping, task, environment, provenance)
        validate_semantics(contract, task, contrasts)
        return contract
    except ContractError:
        raise
    except (TypeError, KeyError, ValueError, OverflowError, RecursionError) as exc:
        raise ContractError('INVALID_CONTRACT_STRUCTURE') from exc


def _compile_contract(mapping, task, environment, provenance=None):
    mapping, task, environment = snapshot(mapping), snapshot(task), snapshot(environment)
    exact(task, ['task_id', 'instruction', 'authorized_objects', 'authorized_destinations', 'allowed_actions', 'expected_goals'], 'TASK')
    if not all(type(task[k]) is str and task[k].strip() for k in ('task_id', 'instruction')):
        raise ContractError('INVALID_TASK_ID_OR_INSTRUCTION')
    exact(environment, ['schema_version', 'environment_id', 'adapter_ref', 'objects', 'destinations',
                        'actions', 'limits', 'fact_sources', 'hazard_rules'], 'ENVIRONMENT')
    if environment['schema_version'] != 'f2-environment-v1':
        raise ContractError('UNSUPPORTED_ENVIRONMENT_VERSION')
    for k in ('environment_id', 'adapter_ref'):
        if type(environment[k]) is not str or not environment[k].strip():
            raise ContractError('INVALID_ENVIRONMENT_BINDING')
    objects, destinations = names(environment['objects'], 'OBJECTS'), names(environment['destinations'], 'DESTINATIONS')
    authorized = names(task['authorized_objects'], 'AUTHORIZED_OBJECTS')
    authorized_dest = names(task['authorized_destinations'], 'AUTHORIZED_DESTINATIONS')
    sources = names(environment['fact_sources'], 'SOURCES')
    if not sources or sources & {'vlm', 'llm', 'model', 'reason', 'plan'}:
        raise ContractError('UNTRUSTED_FACT_SOURCE')
    if not authorized <= objects or not authorized_dest <= destinations:
        raise ContractError('UNKNOWN_AUTHORIZED_REFERENCE')
    limits = environment['limits']
    required_limits = {'max_observation_age_s', 'max_rgbd_skew_s'}
    allowed_limits = required_limits | {'max_step_wait_s', 'max_total_wait_s', 'max_action_occurrences'}
    if (not isinstance(limits, dict) or not required_limits <= set(limits) or not set(limits) <= allowed_limits
            or any(not finite(v) or v <= 0 for k, v in limits.items() if k != 'max_action_occurrences')):
        raise ContractError('INVALID_TRUSTED_LIMITS')
    occurrences = limits.get('max_action_occurrences', {})
    if not isinstance(occurrences, dict) or any(type(v) is not int or v < 1 for v in occurrences.values()):
        raise ContractError('INVALID_OCCURRENCE_LIMITS')
    catalog = environment['actions']
    if not isinstance(catalog, dict) or not 1 <= len(catalog) <= 256:
        raise ContractError('INVALID_ACTION_CATALOG')
    specs = {}
    for action, value in catalog.items():
        names([action], 'ACTION_NAME')
        exact(value, ['before', 'after', 'effect', 'role', 'object_mode', 'destination_mode', 'parameters'], 'ACTION_SPEC')
        before = names(value['before'], 'ACTION_PHASES')
        if (not before or type(value['after']) is not str or not value['after']
                or value['effect'] not in EFFECTS or value['role'] not in ROLES
                or value['object_mode'] not in {'new', 'focus', 'none', 'any'}
                or value['destination_mode'] not in {'goal', 'focus', 'none', 'annotation'}):
            raise ContractError('INVALID_ACTION_SEMANTICS')
        params = value['parameters']
        if not isinstance(params, dict) or len(params) > 16:
            raise ContractError('INVALID_PARAMETER_SCHEMA')
        for key, rule in params.items():
            if key not in {'seconds', 'turns', 'direction', 'motion_phase', 'at_fraction', 'pause_seconds', 'resume_without_reset', 'force_mode', 'limit_relation'}:
                raise ContractError('UNKNOWN_PARAMETER')
            if not isinstance(rule, dict):
                raise ContractError('INVALID_PARAMETER_SCHEMA')
            if 'required' in rule and type(rule['required']) is not bool:
                raise ContractError('INVALID_REQUIRED_PARAMETER')
            fields = set(rule) - {'required'}
            numeric = key in {'seconds', 'pause_seconds', 'turns', 'at_fraction'}
            if fields == {'min', 'max'}:
                if not numeric:
                    raise ContractError('NUMERIC_PARAMETER_EXPECTED')
                if not finite(rule['min']) or not finite(rule['max']) or rule['min'] > rule['max']:
                    raise ContractError('INVALID_PARAMETER_RANGE')
            elif fields == {'enum'}:
                if numeric:
                    raise ContractError('NUMERIC_PARAMETER_RANGE_REQUIRED')
                if not isinstance(rule['enum'], list) or not rule['enum']:
                    raise ContractError('INVALID_PARAMETER_ENUM')
                expected_type = bool if key == 'resume_without_reset' else str
                if any(type(value) is not expected_type for value in rule['enum']):
                    raise ContractError('PARAMETER_ENUM_TYPE_MISMATCH')
            else:
                raise ContractError('INVALID_PARAMETER_SCHEMA')
        specs[action] = ActionSpec(before, value['after'], value['effect'], value['role'], value['object_mode'],
                                   value['destination_mode'], json.dumps(params, separators=(',', ':')))
    if not set(occurrences) <= set(specs):
        raise ContractError('UNKNOWN_OCCURRENCE_ACTION')
    exact(mapping, ['schema_version', 'goals', 'allowed_actions', 'unresolved'], 'MAPPING')
    if mapping['schema_version'] != 'f2-mapping-v1' or mapping['unresolved'] != []:
        raise ContractError('MAPPING_UNRESOLVED_OR_UNSUPPORTED')
    allowed = names(mapping['allowed_actions'], 'MAPPING_ACTIONS')
    expected_allowed = names(task['allowed_actions'], 'TASK_ACTIONS')
    if not allowed or allowed != expected_allowed or not allowed <= set(specs):
        raise ContractError('ACTION_AUTHORITY_MISMATCH')
    if any(specs[a].role == 'wait' for a in allowed) and not {'max_step_wait_s', 'max_total_wait_s'} <= set(limits):
        raise ContractError('WAIT_LIMITS_REQUIRED')
    goals = mapping['goals']
    if not isinstance(goals, list) or not 1 <= len(goals) <= 256:
        raise ContractError('INVALID_GOALS')
    compiled_goals = {}
    for goal in goals:
        exact(goal, ['object', 'destination', 'outcome', 'count'], 'GOAL')
        o, d, outcome, count = (goal[k] for k in ('object', 'destination', 'outcome', 'count'))
        if (o not in authorized or d not in authorized_dest or outcome not in OUTCOMES
                or o in compiled_goals or count is not None and (type(count) is not int or not 1 <= count <= 256)):
            raise ContractError('INVALID_GOAL_REFERENCE')
        compiled_goals[o] = (d, outcome, count)
    # A trusted expected mapping is a semantic oracle when supplied. Otherwise
    # activation requires host-supplied contrast cases (ContractManager).
    if task['expected_goals'] is not None and sorted(goals, key=lambda g: g['object']) != sorted(task['expected_goals'], key=lambda g: g['object']):
        raise ContractError('TASK_SEMANTIC_MISMATCH')
    for a in allowed:
        spec = specs[a]
        if spec.effect == 'complete':
            expected = {'release': 'supported_placement', 'controlled_release': 'contained_disposal',
                        'publish': 'authorized_delivery', 'configure': 'authorized_update'}.get(spec.role)
            if expected is None or any(g[1] != expected for g in compiled_goals.values()):
                raise ContractError('COMPLETION_SEMANTIC_MISMATCH')
    rules = environment['hazard_rules']
    if not isinstance(rules, list) or len(rules) > 256:
        raise ContractError('INVALID_HAZARD_RULES')
    for r in rules:
        exact(r, ['id', 'category', 'actions', 'object', 'destination', 'fact', 'protected_target',
                  'protected_condition', 'harm_path', 'exposure_condition'], 'HAZARD_RULE')
        if (r['category'] not in {f'HA{i}' for i in range(1, 9)} or not names(r['actions'], 'HAZARD_ACTIONS') <= set(specs)
                or any(type(r[k]) is not str or not r[k].strip() for k in r if k != 'actions')
                or r['object'] != '*' and r['object'] not in objects
                or r['destination'] != '*' and r['destination'] not in destinations):
            raise ContractError('INVALID_HAZARD_CAUSAL_REFERENCE')
        if r['category'] == 'HA5' and any(specs[a].role not in {'manipulate', 'carry', 'release', 'controlled_release'} for a in r['actions']):
            raise ContractError('HA5_REQUIRES_OBJECT_MANIPULATION_NOT_ZONE_ENTRY')
    bindings = dict(task_id=task['task_id'], task_sha256=fingerprint(task), environment_sha256=fingerprint(environment),
                    policy_version=POLICY_VERSION, policy_sha256=policy_fingerprint(),
                    registry_version=REGISTRY_VERSION, adapter_ref=environment['adapter_ref'])
    generation = provenance or dict(mapper_ref='host-validated-mapping-v1', prompt_version='f2-setup-v1')
    generation = snapshot(generation)
    exact(generation, ['mapper_ref', 'prompt_version'], 'GENERATION_PROVENANCE')
    if any(type(v) is not str or not v for v in generation.values()):
        raise ContractError('INVALID_GENERATION_PROVENANCE')
    artifact = dict(schema_version='f2-contract-v1', mapping=mapping, task=task, environment=environment,
                    bindings=bindings, generation=generation)
    contract_id = fingerprint(artifact)
    artifact['contract_id'] = contract_id
    # Immutable runtime lookup tables. Dynamic evidence is never stored here.
    frozen_limits = dict(limits)
    frozen_limits['max_action_occurrences'] = MappingProxyType(dict(occurrences))
    return CompiledContract(contract_id, tuple(sorted(bindings.items())), MappingProxyType(specs), allowed, authorized,
                            authorized_dest, MappingProxyType(compiled_goals), MappingProxyType(frozen_limits), sources,
                            tuple(json.dumps(r, separators=(',', ':')) for r in rules), json.dumps(artifact, ensure_ascii=False, indent=2))


def validate_semantics(contract, task, contrasts):
    if task['expected_goals'] is None and not contrasts:
        raise ContractError('SEMANTIC_VALIDATION_REQUIRED')
    if contrasts:
        from .runtime import RuntimeFilter
        seen = set()
        for case in contrasts:
            s = snapshot(case['state'])
            s['bindings'] = contract.bindings_dict()
            result = RuntimeFilter(contract, clock=lambda s=s: s['observed_at']).inspect(case['plan'], s, step_gated=True)
            if result.decision != case['expected']:
                raise ContractError('SEMANTIC_CONTRAST_FAILED')
            seen.add(case['expected'])
        if task['expected_goals'] is None and not {'PASS', 'BLOCK'} <= seen:
            raise ContractError('NORMAL_AND_UNSAFE_CONTRASTS_REQUIRED')


def load_contract(path, task, environment, *, contrasts=()):
    from .json_input import load_json
    artifact = load_json(path)
    exact(artifact, ['schema_version', 'mapping', 'task', 'environment', 'bindings', 'contract_id', 'generation'], 'CONTRACT_ARTIFACT')
    if artifact['schema_version'] != 'f2-contract-v1' or artifact['task'] != task or artifact['environment'] != environment:
        raise ContractError('STORED_CONTRACT_CONTEXT_MISMATCH')
    result = compile_contract(artifact['mapping'], task, environment, contrasts=contrasts, provenance=artifact['generation'])
    if result.contract_id != artifact['contract_id'] or result.bindings_dict() != artifact['bindings']:
        raise ContractError('STORED_CONTRACT_INTEGRITY_MISMATCH')
    return result


class ContractManager:
    """Serial trusted setup API; never expose to VLM/attacker-controlled tools."""
    def __init__(self, mapper, mapper_ref='qwen3-4b-json-v1'):
        self.mapper, self.mapper_ref = mapper, mapper_ref
        self._cache = OrderedDict()
        self.active = None
        self.llm_calls = 0

    def prepare(self, task, environment, contrasts=()):
        self.active = None  # failed new setup must not silently run the previous task
        task, environment = snapshot(task), snapshot(environment)
        key = fingerprint([task, environment, POLICY_VERSION, policy_fingerprint(), REGISTRY_VERSION, self.mapper_ref, contrasts])
        if key in self._cache:
            self.active = self._cache[key]
            self._cache.move_to_end(key)
            return self.active
        if task.get('expected_goals') is None and not contrasts:
            raise ContractError('SEMANTIC_VALIDATION_REQUIRED')
        schema = mapping_schema(task, environment)
        payload = dict(approved_task=task, static_environment=environment, policy_version=POLICY_VERSION,
                       supported_outcomes=sorted(OUTCOMES), mandatory_guards={k: list(v) for k, v in GUARDS.items()})
        self.llm_calls += 1
        mapping = self.mapper(payload, schema)
        if isinstance(mapping, (str, bytes)):
            mapping = parse_json(mapping)
        contract = compile_contract(mapping, task, environment, contrasts=contrasts,
                                    provenance=dict(mapper_ref=self.mapper_ref, prompt_version='f2-setup-v1'))
        self._cache[key] = contract
        if len(self._cache) > 16:
            self._cache.popitem(last=False)
        self.active = contract
        return contract
