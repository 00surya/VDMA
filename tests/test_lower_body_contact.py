"""A slow hand at hip level must not turn torso padding into a strike target."""
import pytest

from test_bilateral_punches import slow_people
from test_strike_depth import contact, paint_arm, scene
from vmd.heuristics import FightHeuristic, Rules, body_contact_regions
from vmd.spatial import fight_depth_evidence
from vmd.stabilization import PoseStabilizer


def observe_height(height, *, scale=1, hidden_hip=False):
    detector, stabilizer = FightHeuristic(Rules(threshold=.55)), PoseStabilizer()
    results = []
    for n in range(80):
        timestamp = n/8
        people = slow_people(timestamp)
        for actor in people:
            x, _, confidence = actor.keypoints[9]
            actor.keypoints[9] = (x, height, confidence)
            if hidden_hip:
                actor.keypoints[12] = (0, 1000, .1)
            actor.keypoints = [(x*scale, y*scale, confidence) for x, y, confidence in actor.keypoints]
            actor.box = tuple(value*scale for value in actor.box)
            actor.body_scale *= scale
        stable = stabilizer.update(people, timestamp,
            lambda actor, joint: .25 if joint == 9 else 0)
        results.append(detector.update(stable, timestamp, .002))
    return results


@pytest.mark.parametrize('height', [201, 212, 225])
@pytest.mark.parametrize('scale', [1, 2])
def test_supported_low_hand_approach_inside_padded_boxes_is_not_a_slow_strike(height, scale):
    results = observe_height(height, scale=scale)
    # The movement has a real measured approach and sufficient score. It is
    # rejected because it reaches below the target hips, not due to no motion.
    assert any(result.signals.get('slow_punch_activity')
               and result.signals['slow_punch_score'] >= .55 for result in results)
    assert not any(result.signals.get('candidate') for result in results)
    assert not any(result.signals.get('slow_punch_strikes') for result in results)
    assert not any(result.signals.get('bilateral_punch_cycles') for result in results)
    assert not any(result.events for result in results)


@pytest.mark.parametrize('height', [155, 190, 200])
def test_the_same_supported_approach_to_visible_torso_still_produces_strikes(height):
    results = observe_height(height)
    strikes = [strike for result in results for strike in result.signals.get('slow_punch_strikes', [])]
    assert {strike['actor'] for strike in strikes} == {1, 2}
    assert all(strike['contact_kind'] == 'body' for strike in strikes)
    assert any(result.signals.get('bilateral_punching') for result in results)
    assert not any(result.signals.get('candidate') for result in results)


def test_hidden_hip_coordinates_do_not_extend_the_lower_body_target():
    results = observe_height(212, hidden_hip=True)
    assert any(result.signals.get('slow_punch_activity') for result in results)
    assert not any(result.signals.get('slow_punch_strikes') for result in results)
    valid = observe_height(155, hidden_hip=True)
    assert any(result.signals.get('slow_punch_strikes') for result in valid)


@pytest.mark.parametrize(('height', 'eligible'), [(190, True), (200, True), (201, False), (212, False)])
def test_raw_depth_contact_geometry_uses_the_same_visible_hip_boundary(height, eligible):
    people, depth = scene()
    people[0].keypoints[9], people[0].keypoints[7] = (230, height, .99), (160, height, .99)
    paint_arm(depth, people[0], 9, .42)
    # All these points remain in the old padded target rectangle.
    assert any(left <= 230 <= right and top <= height <= bottom
               for left, top, right, bottom in body_contact_regions(people[1], Rules()))
    reading = fight_depth_evidence(depth, people)[0]
    evidence = contact(reading)
    assert evidence['status'] == 'compatible'
    assert evidence['geometry_match'] is eligible
    assert reading['fight_status'] == ('compatible' if eligible else 'uncertain')
