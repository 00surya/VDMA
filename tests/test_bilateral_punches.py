"""Slow punch evidence uses actual wrist trajectories, not relaxed fast thresholds."""
import math

import pytest

from vmd.heuristics import FightHeuristic, Person, Rules
from vmd.stabilization import PoseStabilizer


def slow_actor(track, x):
    points = [(x, 80, .99)] * 17
    points[5:7] = [(x-25, 110, .99), (x+25, 110, .99)]
    points[7:9] = [(x-30, 140, .99), (x+30, 140, .99)]
    points[9:11] = [(x+25, 155, .99), (x-25, 155, .99)]
    points[11:13] = [(x-20, 200, .99), (x+20, 200, .99)]
    points[13:15] = [(x-20, 245, .99), (x+20, 245, .99)]
    points[15:17] = [(x-25, 290, .99), (x+25, 290, .99)]
    return Person(track, (x-80, 50, x+80, 300), points, body_scale=250, local_flow=.002)


def slow_people(timestamp, period=3, scene='punch'):
    a, b = slow_actor(1, 190), slow_actor(2, 300)
    fraction = lambda offset=0: (1-math.cos((timestamp+offset)*2*math.pi/period))/2
    a.keypoints[9] = (215+70*fraction(), 155, .99)
    b.keypoints[9] = (275-70*fraction(period/2), 155, .99)
    if scene == 'one_sided':
        b.keypoints[9] = (275, 155, .99)
    elif scene == 'wave':
        a.keypoints[9] = (215, 155+35*math.sin(timestamp*math.pi), .99)
        b.keypoints[9] = (275, 155+35*math.sin(timestamp*math.pi), .99)
    elif scene == 'handshake':
        a, b = slow_actor(1, 210), slow_actor(2, 270)
        for actor in (a, b):
            actor.keypoints[9] = (240, 175+20*math.sin(timestamp*math.pi), .99)
    elif scene in {'hug', 'shoulder_pat'}:
        amplitude = 4 if scene == 'hug' else 20
        a.keypoints[9] = (285, 120+amplitude*math.sin(timestamp*math.pi), .99)
        b.keypoints[9] = (205, 120+amplitude*math.sin(timestamp*math.pi), .99)
    elif scene in {'translation', 'static'}:
        a.keypoints[9], b.keypoints[9] = (285, 155, .99), (205, 155, .99)
        if scene == 'translation':
            for actor in (a, b):
                actor.keypoints = [(x+timestamp*10, y, c) for x, y, c in actor.keypoints]
                actor.box = tuple(value+timestamp*10 if index % 2 == 0 else value
                                  for index, value in enumerate(actor.box))
    return [a, b]


def run_slow(scene='punch', period=3, image_support=True, duration=18, hidden_hip=None):
    detector, stabilizer = FightHeuristic(), PoseStabilizer()
    results, speeds = [], []
    for n in range(int(duration*8)):
        timestamp = n/8
        raw = slow_people(timestamp, period, scene)
        if hidden_hip is not None:
            for actor in raw:
                actor.keypoints[12] = (hidden_hip, hidden_hip, .1)
        people = stabilizer.update(raw, timestamp,
            lambda actor, index: .25 if image_support and index == 9 else 0)
        if scene == 'static':
            # Falsely populated motion metadata cannot replace an observed path.
            for actor in people:
                actor.limb_speeds, actor.limb_motion = {9: .25}, {9: .25}
        speeds.extend(actor.limb_speeds.get(9, 0) for actor in people)
        results.append(detector.update(people, timestamp, .002))
    return detector, results, speeds


@pytest.mark.parametrize('period', [2.5, 3, 4, 5])
def test_slow_bilateral_contact_withdrawals_survive_actual_stabilizer(period):
    _, results, speeds = run_slow(period=period)
    assert max(speeds) < Rules.wrist_speed*.7  # No accidental fast-rule positive.
    assert not any(result.signals.get('candidate') for result in results)
    bilateral = [result for result in results if result.signals.get('bilateral_punching')]
    assert bilateral
    records = bilateral[0].signals['bilateral_punch_cycles']
    assert len(records) >= 3
    for actor in (1, 2):
        assert sum(record['actor'] == actor for record in records) >= 1
    assert all(record['start'] < record['contact'] < record['time'] for record in records)
    assert any(result.signals.get('slow_punch_activity') and result.signals['slow_punch_score'] >= .6
               for result in results)


@pytest.mark.parametrize('scene', ['one_sided', 'wave', 'handshake', 'hug', 'shoulder_pat', 'translation', 'static'])
def test_ordinary_actions_and_one_sided_slow_motion_do_not_qualify_bilaterally(scene):
    _, results, _ = run_slow(scene=scene)
    assert not any(result.signals.get('bilateral_punching') for result in results)
    if scene != 'one_sided':
        assert not any(result.signals.get('bilateral_punch_cycles') for result in results)


def test_unsupported_pose_movement_cannot_make_slow_punch_cycles():
    _, results, _ = run_slow(image_support=False)
    assert not any(result.signals.get('slow_punch_activity') for result in results)
    assert not any(result.signals.get('bilateral_punch_cycles') for result in results)


@pytest.mark.parametrize('break_kind', ['gap', 'camera', 'lost', 'new_id', 'separated', 'bad_pose'])
def test_slow_cycles_reset_with_invalid_pair_or_motion_history(break_kind):
    detector, _, _ = run_slow()
    assert detector.slow_punch_cycles
    people = slow_people(18)
    for actor in people:
        actor.limb_speeds, actor.limb_motion = {9: .25}, {9: .25}
    timestamp, camera = 18, 0
    if break_kind == 'gap':
        timestamp += 2
    elif break_kind == 'camera':
        camera = .2
    elif break_kind == 'lost':
        people = people[:1]
    elif break_kind == 'new_id':
        people[1].track_id = 3
    elif break_kind == 'separated':
        detector.reset_slow_punch_pair((1, 2))
    elif break_kind == 'bad_pose':
        people[1].pose_reliable = False
    result = detector.update(people, timestamp, .002, camera)
    assert not result.signals.get('bilateral_punching')
    assert (1, 2) not in detector.slow_punch_cycles


def test_completed_history_without_current_wrist_motion_is_not_activity():
    detector, _, _ = run_slow()
    people = slow_people(17.875)
    for actor in people:
        actor.limb_speeds, actor.limb_motion = {}, {}
    result = detector.update(people, 18, .002)
    assert result.signals['bilateral_punching']
    assert not result.signals['slow_punch_activity']
    assert result.signals['slow_punch_score'] == 0


@pytest.mark.parametrize('hidden_hip', [0, 1000])
def test_one_visible_hip_is_a_valid_anchor_without_using_hidden_hip_coordinates(hidden_hip):
    _, results, _ = run_slow(hidden_hip=hidden_hip)
    assert any(result.signals.get('bilateral_punching') for result in results)


def test_changed_torso_anchor_does_not_reuse_old_cycles():
    detector, _, _ = run_slow()
    people = slow_people(18)
    people[1].keypoints[12] = (0, 0, .1)
    for actor in people:
        actor.limb_speeds, actor.limb_motion = {9: .25}, {9: .25}
    result = detector.update(people, 18, .002)
    assert not result.signals['bilateral_punch_cycles']


def test_moving_wrist_cannot_borrow_image_support_from_other_wrist():
    detector, stabilizer = FightHeuristic(), PoseStabilizer()
    for n in range(100):
        people = stabilizer.update(slow_people(n/8), n/8,
            lambda actor, index: .25 if index == 10 else 0)
        result = detector.update(people, n/8, .002)
        assert not result.signals.get('slow_punch_activity')
        assert not result.signals.get('bilateral_punch_cycles')


def test_one_simultaneous_two_handed_contact_release_is_not_two_punches_per_actor():
    detector, stabilizer = FightHeuristic(), PoseStabilizer()
    bilateral = False
    max_per_actor = 0
    for n in range(45):
        timestamp = n/8
        people = slow_people(min(timestamp, 3), period=3)
        # Both arms of each person move together through just one contact-release.
        # Start both actors at the same phase rather than alternate their turns.
        fraction = (1-math.cos(min(timestamp, 3)*2*math.pi/3))/2
        people[1].keypoints[9] = (275-70*fraction, 155, .99)
        for actor in people:
            actor.keypoints[10] = actor.keypoints[9]
        filtered = stabilizer.update(people, timestamp,
            lambda actor, index: .25 if index in (9, 10) else 0)
        result = detector.update(filtered, timestamp, .002)
        records = result.signals.get('bilateral_punch_cycles', [])
        max_per_actor = max(max_per_actor, *(sum(record['actor'] == actor for record in records) for actor in (1, 2)))
        bilateral |= bool(result.signals.get('bilateral_punching'))
    assert max_per_actor == 1
    assert not bilateral
