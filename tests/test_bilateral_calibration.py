"""Calibration tolerance must preserve bilateral, directional motion evidence."""
import math

import pytest

from test_bilateral_punches import slow_actor, slow_people
from vmd.behavior import BehaviorHeuristic
from vmd.heuristics import Rules
from vmd.stabilization import PoseStabilizer


def calibration_people(timestamp, scene, amplitude=40):
    a, b = slow_actor(1, 190), slow_actor(2, 300)
    fraction = lambda stamp: (1-math.cos(stamp*2*math.pi/3))/2
    if scene == 'near_body':
        # Stops a little outside strict torso contact, within pose tolerance.
        a.keypoints[9] = (195+55*fraction(timestamp), 155, .99)
        b.keypoints[9] = (295-55*fraction(timestamp+1.5), 155, .99)
    elif scene == 'two_plus_one':
        a.keypoints[9] = (215+70*fraction(timestamp), 155, .99)
        b.keypoints[9] = (275-70*fraction(min(timestamp, 3)), 155, .99)
    elif scene == 'handshake':
        a, b = slow_actor(1, 210), slow_actor(2, 270)
        for actor in (a, b):
            actor.keypoints[9] = (240, 175+amplitude*math.sin(timestamp*math.pi/1.5), .99)
    elif scene == 'own_clothes':
        a, b = slow_people(timestamp)
        b.keypoints[9] = (300+20*math.sin(timestamp*math.pi), 160+20*math.cos(timestamp*math.pi), .99)
    elif scene == 'single_hug':
        movement = fraction(min(timestamp, 3))
        a.keypoints[9] = (215+70*movement, 155, .99)
        b.keypoints[9] = (275-70*movement, 155, .99)
        for actor in (a, b):
            actor.keypoints[10] = actor.keypoints[9]
    elif scene == 'moving_target':
        b = slow_actor(2, 300-35*fraction(timestamp))
        a.keypoints[9] = (250, 155, .99)
    elif scene == 'outside_tolerance':
        a.keypoints[9] = (180+57*fraction(timestamp), 155, .99)
        b.keypoints[9] = (310-57*fraction(timestamp+1.5), 155, .99)
    return [a, b]


def run_calibration(scene, *, amplitude=40, depth='compatible', duration=15):
    behavior = BehaviorHeuristic(Rules(threshold=.55, fight_confirmation_seconds=1), confirmation=True)
    stabilizer = PoseStabilizer()
    events, results = [], []
    last_depth = float('-inf')
    for n in range(int(duration*8)):
        timestamp = n/8
        people = stabilizer.update(calibration_people(timestamp, scene, amplitude), timestamp,
            lambda actor, index: .25 if index in (9, 10) else 0)
        sampled = timestamp-last_depth >= .5
        if sampled and depth != 'missing':
            behavior.observe_depth(timestamp, [{'tracks': [1, 2], 'status': depth}], timestamp)
            last_depth = timestamp
        result = behavior.update(people, timestamp, .002)
        priority = (any(state.since == timestamp for state in behavior.confirmation.pending.values())
                    or any(strike['contact'] == timestamp for strikes in behavior.fight.slow_punch_strikes.values()
                           for strike in strikes))
        if priority and not sampled and depth != 'missing':
            behavior.observe_depth(timestamp, [{'tracks': [1, 2], 'status': depth}], timestamp)
            last_depth = timestamp
        results.append(result)
        events.extend(result.events)
    return results, events


def test_calibrated_bilateral_strikes_confirm_without_waiting_for_three_completed_cycles():
    results, events = run_calibration('near_body')
    fights = [event for event in events if event.event_type == 'fight']
    assert len(fights) == 1
    assert not any(result.signals.get('candidate') for result in results)
    strikes = fights[0].signals['slow_punch_strikes']
    assert {strike['actor'] for strike in strikes} == {1, 2}
    assert max(strike['contact'] for strike in strikes)-min(strike['contact'] for strike in strikes) >= .5
    assert len(fights[0].signals['bilateral_punch_cycles']) < 3
    assert fights[0].signals['fight_timer_seconds'] >= 1
    assert fights[0].signals['depth_samples'] >= 2


def test_old_reciprocal_contact_cannot_bridge_a_long_pause_into_one_sided_motion():
    results, events = run_calibration('two_plus_one')
    # Both people act once together; after the last withdrawal at 2.125s,
    # only actor 1 resumes contact at 3.75s, beyond the 1.5-second grace.
    assert [event.event_type for event in events] == ['possible_fight']
    assert not any(result.state == 'fight_detected' for result in results)


@pytest.mark.parametrize('amplitude', [40, 60])
def test_same_depth_handshake_pumps_never_become_bilateral_punches(amplitude):
    results, events = run_calibration('handshake', amplitude=amplitude)
    assert not any(result.signals.get('bilateral_punching') for result in results)
    assert not [event for event in events if event.event_type in {'fight', 'possible_fight'}]


@pytest.mark.parametrize('scene', ['own_clothes', 'single_hug', 'moving_target', 'outside_tolerance'])
def test_tolerance_does_not_turn_unrelated_or_insufficient_actions_into_fights(scene):
    results, events = run_calibration(scene)
    assert not any(result.signals.get('bilateral_punching') for result in results)
    assert not [event for event in events if event.event_type == 'fight']
    if scene == 'own_clothes':
        # The first actor really punches toward the other torso. Immediate
        # review is intentional; the unrelated gesture cannot confirm it.
        assert [event.event_type for event in events] == ['possible_fight']
    else:
        assert not [event for event in events if event.event_type == 'possible_fight']
    if scene == 'single_hug':
        assert max(len(result.signals.get('bilateral_punch_cycles', [])) for result in results) <= 2


@pytest.mark.parametrize('depth', ['missing', 'separated', 'uncertain'])
def test_relaxed_contact_tolerance_still_cannot_confirm_without_matching_depth(depth):
    _, events = run_calibration('near_body', depth=depth)
    assert not any(event.event_type == 'fight' for event in events)
