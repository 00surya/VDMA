"""Short visible grabs need articulated arms and observed same-actor escape."""
from dataclasses import replace

import pytest

from vmd.heuristics import Rules
from vmd.snatching import SnatchingHeuristic
from test_snatching import person, scene


def triggers(detector, people, timestamp):
    return [event for event in detector.update(people, timestamp) if event.trigger]


@pytest.mark.parametrize('missing_joint', [11, 12])
def test_one_occluded_hip_uses_visible_anchors_and_preserves_original_depth(missing_joint):
    detector, events = SnatchingHeuristic(Rules()), []
    for n in range(24):
        timestamp = n/8
        people = scene(timestamp)
        for p in people:
            # A missing landmark must not drag the torso/departure estimate.
            p.keypoints[missing_joint] = (4000, -2000, .05)
        if n == 3:
            detector.observe_depth(.125, [{'tracks': [1, 2], 'status': 'compatible'}], timestamp)
        events.extend(triggers(detector, people, timestamp))
    assert [event.event_type for event in events] == ['possible_snatching', 'snatching_detected']
    assert events[-1].signals['depth_source_time'] == .125
    assert events[0].signals['episode_id'] == events[-1].signals['episode_id']
    assert events[0].signals['pull_body_heights'] >= .08


@pytest.mark.parametrize('depth', [None, 'compatible', 'separated'])
@pytest.mark.parametrize('cropped', [False, True])
def test_one_frame_contact_at_two_fps_retains_multiple_observed_escape_samples(depth, cropped):
    detector, events = SnatchingHeuristic(Rules()), []
    for n in range(10):
        timestamp = n*.5
        travel = max(0, timestamp-1)*300
        wrist = (430, 102, .99) if n == 0 else (525, 102, .99) if n == 1 else (475-travel, 102, .99)
        actor, target = person(1, 400-travel, wrist, moving=timestamp > 1), person(2, 540)
        if cropped:
            for joint in (15, 16):
                actor.keypoints[joint] = (*actor.keypoints[joint][:2], .05)
                actor.limb_speeds.pop(joint, None)
                actor.limb_motion.pop(joint, None)
        if n == 2 and depth:
            detector.observe_depth(.5, [{'tracks': [1, 2], 'status': depth}], timestamp)
        events.extend((timestamp, event) for event in triggers(detector, [actor, target], timestamp))
    if depth == 'separated':
        assert events == []
    elif depth is None:
        assert [event.event_type for _, event in events] == ['possible_snatching']
    else:
        assert [event.event_type for _, event in events] == ['possible_snatching', 'snatching_detected']
        confirmed_time, confirmed = events[-1]
        assert confirmed_time >= (3 if cropped else 2.5)
        assert confirmed.signals['depth_source_time'] == .5
        assert confirmed.signals['escape_support'] == ('visible_torso' if cropped else 'legs')


@pytest.mark.parametrize('moving_person', ['target', 'actor'])
def test_body_translation_with_frozen_actor_arm_cannot_be_a_reach(moving_person):
    detector, events = SnatchingHeuristic(Rules()), []
    for n, displacement in enumerate((0, 95, 45, 0)):
        if moving_person == 'target':
            people = [person(1, 360, (430, 102, .99)), person(2, 540-displacement)]
        else:
            people = [person(1, 360+displacement, (430+displacement, 102, .99)), person(2, 540)]
        # Even over-reported limb speed/flow cannot replace articulated motion.
        events.extend(triggers(detector, people, n/8))
    assert events == []
    assert detector.pending == {}


def test_the_other_wrist_can_start_and_complete_the_same_sequence():
    detector, events = SnatchingHeuristic(Rules()), []
    for n in range(24):
        timestamp = n/8
        people = scene(timestamp)
        actor = people[0]
        actor.keypoints[10] = actor.keypoints[9]
        actor.keypoints[9] = (actor.box[0]+20, 165, .99)
        actor.limb_speeds[10] = actor.limb_speeds.pop(9)
        actor.limb_motion[10] = actor.limb_motion.pop(9)
        if n == 3:
            detector.observe_depth(.125, [{'tracks': [2, 1], 'status': 'compatible'}], timestamp)
        events.extend(triggers(detector, people, timestamp))
    assert [event.event_type for event in events] == ['possible_snatching', 'snatching_detected']
    assert events[-1].signals['actor_track'] == 1
    assert events[-1].signals['depth_source_time'] == .125


def test_body_translation_after_a_true_reach_cannot_supply_the_sharp_pull():
    detector, events = SnatchingHeuristic(Rules()), []
    for n in range(24):
        timestamp = n/8
        if n < 2:
            people = scene(timestamp)
        else:
            x = 350-max(0, timestamp-.5)*300
            # The actor moves away with the arm held in the contact posture.
            people = [person(1, x, (x+125, 102, .99), moving=True), person(2, 540)]
        if n == 3:
            detector.observe_depth(.125, [{'tracks': [1, 2], 'status': 'compatible'}], timestamp)
        events.extend(triggers(detector, people, timestamp))
    assert events == []


def test_target_running_faster_cannot_turn_an_actor_following_them_into_escape():
    detector, events = SnatchingHeuristic(Rules()), []
    for n in range(30):
        timestamp = n/8
        if timestamp <= .5:
            people = scene(timestamp)
        else:
            elapsed = timestamp-.5
            people = [person(1, 400+elapsed*300, (475+elapsed*300, 102, .99), moving=True),
                      person(2, 540+elapsed*600, moving=True)]
        if n == 3:
            detector.observe_depth(.125, [{'tracks': [1, 2], 'status': 'compatible'}], timestamp)
        events.extend(triggers(detector, people, timestamp))
    assert [event.event_type for event in events] == ['possible_snatching']
    assert not events[0].signals['escape_detected']


def translate(person, dx=0, dy=0):
    return replace(person, box=(person.box[0]+dx, person.box[1]+dy,
                                person.box[2]+dx, person.box[3]+dy),
                   keypoints=[(x+dx, y+dy, confidence) for x, y, confidence in person.keypoints])


def sideways_scene(timestamp, direction=1, action='escape'):
    people = [translate(p, dy=350) for p in scene(min(timestamp, .5))]
    if timestamp > .5:
        actor = people[0]
        elapsed = timestamp-.5
        displacement = elapsed*300
        if action == 'zigzag':
            # Supported fast motion repeatedly returns toward the victim. It is
            # not a continuing outward escape even though individual steps run.
            displacement = 30 if round(elapsed*8) % 2 else 0
        actor = translate(actor, dy=direction*displacement)
        actor.limb_speeds = {15: 1.2}
        actor.limb_motion = {15: .4}
        people[0] = actor
    return people


@pytest.mark.parametrize('direction', [-1, 1])
@pytest.mark.parametrize('depth', ['compatible', None, 'separated'])
def test_sideways_escape_confirms_before_short_visible_actor_leaves_only_with_grab_depth(direction, depth):
    detector, events = SnatchingHeuristic(Rules()), []
    for n in range(24):
        timestamp = n/8
        people = sideways_scene(timestamp, direction)
        if timestamp > 1.25:
            people = people[1:]
        if n == 3 and depth:
            detector.observe_depth(.125, [{'tracks': [1, 2], 'status': depth}], timestamp)
        events.extend((timestamp, event) for event in triggers(detector, people, timestamp))
    expected = (['possible_snatching', 'snatching_detected'] if depth == 'compatible'
                else ['possible_snatching'])
    assert [event.event_type for _, event in events] == expected
    if depth == 'compatible':
        assert events[-1][0] == 1.25
        assert events[-1][1].signals['depth_source_time'] == .125
        assert events[0][1].signals['episode_id'] == events[-1][1].signals['episode_id']


@pytest.mark.parametrize('direction', [-1, 1])
def test_supported_sideways_zigzag_cannot_accumulate_signed_escape_distance(direction):
    detector, events = SnatchingHeuristic(Rules()), []
    for n in range(32):
        timestamp = n/8
        if n == 3:
            detector.observe_depth(.125, [{'tracks': [1, 2], 'status': 'compatible'}], timestamp)
        events.extend(triggers(detector, sideways_scene(timestamp, direction, 'zigzag'), timestamp))
    assert [event.event_type for event in events] == ['possible_snatching']


def test_outward_steps_do_not_accumulate_when_actor_overall_follows_a_faster_target():
    detector, events = SnatchingHeuristic(Rules()), []
    for n in range(30):
        timestamp = n/8
        people = scene(min(timestamp, .5))
        if timestamp > .5:
            step = n-4
            displacement = step*40+(80 if step % 2 else 0)
            people = [person(1, 400+displacement, (475+displacement, 102, .99), moving=True),
                      person(2, 540+step*200, moving=True)]
        if n == 3:
            detector.observe_depth(.125, [{'tracks': [1, 2], 'status': 'compatible'}], timestamp)
        # Separation increases each frame. The actor occasionally steps away
        # but travels overall toward the faster target; ignoring inward steps
        # would incorrectly credit enough outward distance for escape.
        events.extend(triggers(detector, people, timestamp))
    assert [event.event_type for event in events] == ['possible_snatching']


@pytest.mark.parametrize('direction', [-1, 1])
def test_sideways_escape_survives_real_pose_filtering_before_actor_leaves(direction):
    import math
    from vmd.stabilization import PoseStabilizer

    detector, stabilizer, events = SnatchingHeuristic(Rules()), PoseStabilizer(), []
    for n in range(32):
        timestamp = n/8
        relative = max(0, timestamp-.75)
        people = sideways_scene(relative, direction)
        if relative > .5:
            for joint, sign in ((15, 1), (16, -1)):
                x, y, confidence = people[0].keypoints[joint]
                people[0].keypoints[joint] = (x+sign*35*math.sin(timestamp*9), y, confidence)
        if relative > 1.375:
            people = people[1:]
        filtered = stabilizer.update(people, timestamp, lambda *args: .8)
        if n == 9:
            detector.observe_depth(.875, [{'tracks': [1, 2], 'status': 'compatible'}], timestamp)
        events.extend((timestamp, event) for event in triggers(detector, filtered, timestamp))
    assert [event.event_type for _, event in events] == ['possible_snatching', 'snatching_detected']
    assert events[-1][0] == 2.125
    assert events[-1][1].signals['depth_source_time'] == .875


@pytest.mark.parametrize('discontinuity', ['position', 'scale', 'duplicate_id'])
def test_same_numeric_id_cannot_finish_a_grab_after_identity_discontinuity(discontinuity):
    detector, events = SnatchingHeuristic(Rules()), []
    for n in range(24):
        timestamp = n/8
        people = scene(timestamp)
        if n >= 4 and discontinuity == 'position':
            actor = people[0]
            people[0] = replace(actor, box=tuple(value-350 if axis % 2 == 0 else value
                                               for axis, value in enumerate(actor.box)),
                                keypoints=[(x-350, y, confidence) for x, y, confidence in actor.keypoints])
        if n >= 4 and discontinuity == 'scale':
            people[0] = replace(people[0], body_scale=750)
        if n == 4 and discontinuity == 'duplicate_id':
            people.append(person(1, 800))
        if n == 3:
            detector.observe_depth(.125, [{'tracks': [1, 2], 'status': 'compatible'}], timestamp)
        events.extend(triggers(detector, people, timestamp))
    assert [event.event_type for event in events] == ['possible_snatching']


@pytest.mark.parametrize('bad_landmarks', [(5,), (6,), (11, 12)])
def test_partial_visibility_still_requires_both_shoulders_and_a_hip(bad_landmarks):
    detector = SnatchingHeuristic(Rules())
    for n in range(24):
        people = scene(n/8)
        for p in people:
            for joint in bad_landmarks:
                p.keypoints[joint] = (*p.keypoints[joint][:2], .05)
        assert triggers(detector, people, n/8) == []
