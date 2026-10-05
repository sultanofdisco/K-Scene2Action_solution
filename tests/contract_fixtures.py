import copy
from f2.contract import GUARDS


def environment(objects=('red_ball', 'blue_ball'), destinations=('zone1', 'zone2')):
    actions = {}
    chain = [
        ('approach', ['idle'], 'approached', 'focus', 'motion', 'new', 'annotation'),
        ('descend', ['approached'], 'descended', 'none', 'motion', 'focus', 'annotation'),
        ('grasp', ['descended'], 'grasped', 'hold', 'manipulate', 'focus', 'annotation'),
        ('lift', ['grasped'], 'lifted', 'none', 'motion', 'focus', 'annotation'),
        ('transfer', ['lifted', 'transferred'], 'transferred', 'destination', 'carry', 'focus', 'goal'),
        ('place', ['transferred'], 'placed', 'none', 'carry', 'focus', 'focus'),
        ('release', ['placed'], 'released', 'complete', 'release', 'focus', 'focus'),
        ('retreat', ['released'], 'idle', 'clear', 'motion', 'focus', 'annotation'),
    ]
    for name, before, after, effect, role, obj, dest in chain:
        actions[name] = dict(before=before, after=after, effect=effect, role=role,
                             object_mode=obj, destination_mode=dest, parameters={})
    for name, role in [('verify', 'verify'), ('stop', 'stop'), ('hold', 'wait'), ('pause', 'wait'), ('resume', 'resume')]:
        actions[name] = dict(before=['*'], after='same', effect='none', role=role,
                             object_mode='any', destination_mode='annotation', parameters={})
    actions['release']['parameters'] = {'motion_phase': {'enum': ['none', 'stationary']}}
    actions['drop'] = dict(before=['transferred'], after='released', effect='complete',
                           role='controlled_release', object_mode='focus', destination_mode='focus', parameters={})
    for name in ('throw', 'twist', 'swing', 'push', 'strike', 'crush'):
        actions[name] = dict(before=['*'], after='same', effect='none', role='motion',
                            object_mode='any', destination_mode='annotation', parameters={})
    return dict(schema_version='f2-environment-v1', environment_id='table-v9',
                adapter_ref='direct-ik-test-v1', objects=list(objects) + ['knife', 'robot_joint'],
                destinations=list(destinations) + ['edge', 'off_table', 'human', 'front_strip'],
                actions=actions, limits={'max_observation_age_s': 2.0, 'max_rgbd_skew_s': 0.05,
                                         'max_step_wait_s': 1.0, 'max_total_wait_s': 2.0},
                fact_sources=['sensor', 'controller', 'safety_monitor'], hazard_rules=[])


def task(objects=('red_ball', 'blue_ball'), destinations=('zone1', 'zone2')):
    goals = [dict(object=o, destination=d, outcome='supported_placement', count=None)
             for o, d in zip(objects, destinations)]
    return dict(task_id='task-1', instruction='빨간 공은 1구역, 파란 공은 2구역에 놓는다.',
                authorized_objects=list(objects), authorized_destinations=list(destinations),
                allowed_actions=['approach', 'descend', 'grasp', 'lift', 'transfer', 'place', 'release', 'retreat', 'verify', 'stop'],
                expected_goals=goals)


def draft(t=None):
    t = t or task()
    return dict(schema_version='f2-mapping-v1', goals=copy.deepcopy(t['expected_goals']),
                allowed_actions=t['allowed_actions'][:], unresolved=[])


def plan(objects=('red_ball', 'blue_ball'), destinations=('zone1', 'zone2')):
    steps = []
    for obj, dest in zip(objects, destinations):
        for action in ('approach', 'descend', 'grasp', 'lift', 'transfer', 'place', 'release', 'retreat'):
            steps.append(dict(action=action, object=obj, destination=dest))
    return dict(decision='EXECUTE', steps=steps, reason='normal task')


def state(contract, now=100.0, step_index=-1, obj='red_ball', dest='zone1'):
    facts = {}
    for key in ('controller_ready', 'initial_gripper_empty', 'motion_safety_clear', 'holding_confirmed:' + obj,
                'support_confirmed:' + obj + ':' + dest, 'containment_confirmed:' + obj + ':' + dest):
        facts[key] = dict(value=True, source='sensor', observed_at=now, step_index=step_index, evidence='test observation')
    for target, (goal, _, _) in contract.goals.items():
        facts[f'outcome_confirmed:{target}:{goal}'] = dict(value=True, source='sensor', observed_at=now, step_index=step_index, evidence='test completed outcome')
    return dict(bindings=contract.bindings_dict(), episode_id='episode-1', request_id='request-1',
                observed_at=now, rgb_stamp=10.0, depth_stamp=10.0, facts=facts,
                capabilities=sorted({g for guards in GUARDS.values() for g in guards}))
