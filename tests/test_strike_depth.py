"""Original-frame contact depth retains each arm and target surface's identity."""
from dataclasses import replace

import numpy as np
import pytest

from vmd.heuristics import Person
from vmd.spatial import fight_depth_evidence, limb_depth


def actor(track, x):
    joints = [(x, 75, .99)] * 17
    joints[5:7] = [(x-15, 100, .99), (x+15, 100, .99)]
    joints[7:9] = [(x-20, 165, .99), (x+20, 165, .99)]
    joints[9:11] = [(x-20, 235, .99), (x+20, 235, .99)]
    joints[11:13] = [(x-15, 200, .99), (x+15, 200, .99)]
    return Person(track, (x-35, 45, x+35, 280), joints)


def paint_arm(depth, person, wrist, value):
    hand, elbow = (person.keypoints[index] for index in (wrist, wrist-2))
    for fraction in (0, .2, .35):
        x, y = [round(hand[axis]*(1-fraction)+elbow[axis]*fraction) for axis in (0, 1)]
        depth[y-3:y+4, x-3:x+4] = value


def scene(kind='body', torso_gap=.2, source_level=.42, target_level=.4):
    a, b = actor(1, 80), actor(2, 240)
    depth = np.full((320, 360), .05)
    depth[99:202, 64:97] = target_level+torso_gap
    depth[99:202, 224:257] = target_level
    if kind == 'guard':
        a.keypoints[9], a.keypoints[7] = (188, 95, .99), (145, 95, .99)
        b.keypoints[9], b.keypoints[7] = (190, 95, .99), (235, 135, .99)
        paint_arm(depth, b, 9, target_level)
    else:
        a.keypoints[9], a.keypoints[7] = (230, 130, .99), (160, 130, .99)
    paint_arm(depth, a, 9, source_level)
    return [a, b], depth


def contact(reading, actor_id=1, wrist=9, kind='body', target_joint=None):
    return next(item for item in reading['contacts'] if
                (item['actor'], item['wrist'], item['contact_kind'], item['target_joint'])
                == (actor_id, wrist, kind, target_joint))


def test_arm_can_reach_compatible_body_surface_without_equal_torso_depth():
    people, depth = scene()
    reading = fight_depth_evidence(depth, people)[0]
    assert reading['status'] == 'separated'  # Original snatching contract is unchanged.
    assert reading['relative_gap'] == pytest.approx(.2)
    evidence = contact(reading)
    assert evidence['target_track'] == 2
    assert evidence['geometry_match']
    assert evidence['status'] == 'compatible'
    assert evidence['relative_gap'] == pytest.approx(.02)
    assert reading['fight_status'] == 'compatible'


def test_raised_guard_uses_its_own_wrist_and_forearm_instead_of_target_torso():
    people, depth = scene('guard', source_level=.44, target_level=.42)
    reading = fight_depth_evidence(depth, people)[0]
    evidence = contact(reading, kind='guard', target_joint=9)
    assert evidence['geometry_match']
    assert evidence['status'] == 'compatible'
    assert reading['fight_status'] == 'compatible'
    # A different target wrist is a separate observation, never the same key.
    assert not contact(reading, kind='guard', target_joint=10)['geometry_match']


def test_projected_wrist_overlap_does_not_hide_separated_distal_forearms():
    people, depth = scene('guard')
    people[0].keypoints[9] = people[1].keypoints[9]
    paint_arm(depth, people[0], 9, .4)
    paint_arm(depth, people[1], 9, .7)
    # Both wrist pixels now show the foreground hand; the two observed arms
    # still have independent distal surfaces on either side of the crossing.
    assert limb_depth(depth, people[0], 9) is None
    assert limb_depth(depth, people[1], 9)[0] == pytest.approx(.7)
    reading = fight_depth_evidence(depth, people)[0]
    assert contact(reading, kind='guard', target_joint=9)['status'] != 'compatible'


def test_smooth_depth_gradient_along_an_outstretched_arm_is_not_contact_noise():
    people, depth = scene(source_level=.4)
    # Three well-defined surfaces along a sloping forearm; the actual wrist
    # meets the .4 target even though the proximal samples are farther in Z.
    hand, elbow = (people[0].keypoints[index] for index in (9, 7))
    for fraction, value in ((0, .4), (.2, .47), (.35, .52)):
        x, y = [round(hand[axis]*(1-fraction)+elbow[axis]*fraction) for axis in (0, 1)]
        depth[y-3:y+4, x-3:x+4] = value
    assert limb_depth(depth, people[0], 9) == pytest.approx((.4, 0))
    assert contact(fight_depth_evidence(depth, people)[0])['status'] == 'compatible'


def test_head_contact_compares_to_head_interior_instead_of_lower_torso():
    people, depth = scene(source_level=.28, target_level=.4)
    people[0].keypoints[9], people[0].keypoints[7] = (242, 78, .99), (170, 85, .99)
    depth[68:83, 230:250] = .3
    paint_arm(depth, people[0], 9, .28)
    reading = fight_depth_evidence(depth, people)[0]
    assert contact(reading)['geometry_match']
    assert contact(reading)['status'] == 'compatible'
    assert contact(reading)['relative_gap'] <= .03


@pytest.mark.parametrize('invalid', ['wrist_confidence', 'elbow_confidence', 'outside', 'unreliable'])
def test_missing_visible_arm_evidence_never_borrows_another_arm_or_torso(invalid):
    people, depth = scene()
    if invalid == 'unreliable':
        people[0] = replace(people[0], pose_reliable=False)
    else:
        joint = 7 if invalid == 'elbow_confidence' else 9
        x, y, confidence = people[0].keypoints[joint]
        people[0].keypoints[joint] = (-10, y, confidence) if invalid == 'outside' else (x, y, .1)
    reading = fight_depth_evidence(depth, people)[0]
    assert contact(reading)['status'] == 'uncertain'
    assert contact(reading)['relative_gap'] is None
    assert not contact(reading)['geometry_match']
    assert reading['fight_status'] != 'compatible'


def test_compatible_arm_depth_without_contact_geometry_cannot_override_separated_torsos():
    people, depth = scene()
    people[0].keypoints[9], people[0].keypoints[7] = (140, 140, .99), (110, 160, .99)
    paint_arm(depth, people[0], 9, .42)
    reading = fight_depth_evidence(depth, people)[0]
    assert contact(reading)['status'] == 'compatible'
    assert not contact(reading)['geometry_match']
    assert reading['status'] == 'separated'
    assert reading['fight_status'] == 'uncertain'


def test_same_plane_torso_evidence_remains_available_for_traditional_grappling():
    people, depth = scene(torso_gap=.01)
    for person in people:
        person.keypoints[9] = (*person.keypoints[9][:2], .1)
    reading = fight_depth_evidence(depth, people)[0]
    assert reading['status'] == reading['fight_status'] == 'compatible'


def test_low_internal_surface_agreement_cannot_be_hidden_by_matching_medians():
    people, depth = scene(source_level=.4)
    gradient = np.tile(np.linspace(.1, .7, 7), (7, 1))
    paint_arm(depth, people[0], 9, gradient)
    reading = fight_depth_evidence(depth, people)[0]
    assert limb_depth(depth, people[0], 9)[1] > .035
    assert contact(reading)['status'] == 'uncertain'
    assert reading['fight_status'] == 'uncertain'


def test_gross_torso_separation_vetoes_even_apparently_matching_contact_pixels():
    people, depth = scene(torso_gap=.4)
    reading = fight_depth_evidence(depth, people)[0]
    assert reading['status'] == reading['fight_status'] == 'separated'
    assert contact(reading)['geometry_match']
    assert all(item['status'] == 'separated' for item in reading['contacts'])


def test_depth_observations_keep_both_actor_and_wrist_provenance():
    people, depth = scene()
    reading = fight_depth_evidence(depth, people)[0]
    assert contact(reading)['status'] == 'compatible'
    assert contact(reading, wrist=10)['status'] != 'compatible'
    assert {item['actor'] for item in reading['contacts']} == {1, 2}
    assert all(item['target_track'] != item['actor'] for item in reading['contacts'])


@pytest.mark.parametrize('value', [np.nan, np.inf, .5])
def test_invalid_or_flat_maps_cannot_supply_any_contact_evidence(value):
    people, depth = scene()
    depth.fill(value)
    assert fight_depth_evidence(depth, people) == []
