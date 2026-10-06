"""Generic depth surfaces: reliable interiors, uncertainty boundaries and Z separation."""
from dataclasses import replace

import numpy as np
import pytest

from vmd.heuristics import Person
from vmd.spatial import box_overlap, fight_depth_evidence, torso_depth


def pair():
    people = []
    for track, x in ((1, 80), (2, 150)):
        joints = [(x, 30, .99)] * 17
        joints[5:7] = [(x-25, 45, .99), (x+25, 45, .99)]
        joints[11:13] = [(x-25, 125, .99), (x+25, 125, .99)]
        people.append(Person(track, (x-40, 25, x+40, 180), joints))
    return people


def surface(gap=0, spreads=(0, 0)):
    """Two nonoverlapping torso surfaces, with upper/lower depth gradients."""
    people = pair()
    depth = np.full((220, 240), .05, dtype=np.float64)
    for person, level, spread in zip(people, (.4, .4+gap), spreads):
        left, right = int(person.keypoints[5][0]), int(person.keypoints[6][0])+1
        depth[45:85, left:right] = level-spread
        depth[85:126, left:right] = level+spread
    return people, depth


def status(people, depth):
    return fight_depth_evidence(depth, people)[0]['status']


def test_interior_depth_retains_same_plane_despite_opposite_boundary_contamination():
    people, depth = surface(gap=.04, spreads=(.025, .03))
    for person, edge in zip(people, (.92, .08)):
        for index in (5, 6, 11, 12):
            x, y, _ = person.keypoints[index]
            depth[int(y)-5:int(y)+6, int(x)-5:int(x)+6] = edge
    first, second = (torso_depth(depth, person) for person in people)
    assert first == pytest.approx((.4, .025))
    assert second == pytest.approx((.44, .03))
    assert status(people, depth) == 'compatible'


@pytest.mark.parametrize(('gap', 'spreads', 'expected'), [
    (.119, (0, 0), 'compatible'),
    (.121, (0, 0), 'uncertain'),
    (.099, (.02, .02), 'compatible'),
    (.101, (.02, .02), 'uncertain'),
    (.083, (.015, .035), 'compatible'),  # Use the larger spread, not their sum.
    (.086, (.015, .035), 'uncertain'),
    (.05, (.0395, 0), 'compatible'),
    (0, (.0405, 0), 'uncertain'),  # A small gap cannot hide a noisy torso.
    (0, (0, .06), 'uncertain'),
])
def test_gap_and_each_torso_spread_must_pass_independent_limits(gap, spreads, expected):
    people, depth = surface(gap, spreads)
    assert status(people, depth) == expected


@pytest.mark.parametrize('gap', [.19, .25, .4])
def test_nearby_overlapping_people_at_different_depth_never_qualify(gap):
    people, depth = surface(gap, spreads=(.02, .015))
    assert box_overlap(people[0].box, people[1].box) > .05
    assert status(people, depth) == 'separated'


@pytest.mark.parametrize('kind', ['flat', 'nan', 'infinity', 'empty', 'wrong_dimensions'])
def test_absent_or_invalid_depth_never_supplies_compatibility(kind):
    people, depth = surface()
    if kind == 'flat':
        depth.fill(.5)
    elif kind == 'nan':
        depth[0, 0] = np.nan
    elif kind == 'infinity':
        depth[0, 0] = np.inf
    elif kind == 'empty':
        depth = np.empty((0, 0))
    else:
        depth = depth[..., None]
    assert fight_depth_evidence(depth, people) == []


@pytest.mark.parametrize('kind', ['unreliable_pose', 'missing_joints', 'outside_image'])
def test_inadequate_pose_does_not_sample_background_as_a_second_torso(kind):
    people, depth = surface()
    if kind == 'unreliable_pose':
        people[1] = replace(people[1], pose_reliable=False)
    else:
        people[1].keypoints = list(people[1].keypoints)
        for index in (5, 6):
            x, y, _ = people[1].keypoints[index]
            people[1].keypoints[index] = (x, y, .1) if kind == 'missing_joints' else (-20, y, .99)
    assert status(people, depth) == 'uncertain'


def test_one_hidden_hip_still_uses_three_visible_interior_anchors():
    people, depth = surface(gap=.03)
    x, y, _ = people[1].keypoints[12]
    people[1].keypoints[12] = (x, y, .1)
    assert status(people, depth) == 'compatible'


def test_one_occluded_depth_patch_does_not_override_three_agreeing_samples():
    people, depth = surface(gap=.03)
    # A foreign foreground patch covers one anatomical quadrant, not the torso.
    depth[45:85, 55:80] = .9
    assert torso_depth(depth, people[0]) == pytest.approx((.4, 0))
    assert status(people, depth) == 'compatible'
