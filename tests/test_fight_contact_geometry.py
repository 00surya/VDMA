"""Detection-box overlap is not contact with another person's body."""
import math

import pytest

from vmd.heuristics import FightHeuristic, Person
from vmd.behavior import BehaviorHeuristic
from vmd.stabilization import PoseStabilizer


def person(track, x):
    points = [(x, 75, .99)] * 17
    points[5:7] = [(x-25, 110, .99), (x+25, 110, .99)]
    points[11:13] = [(x-20, 200, .99), (x+20, 200, .99)]
    points[9:11] = [(x-35, 155, .99), (x+35, 155, .99)]
    points[15:17] = [(x-20, 290, .99), (x+20, 290, .99)]
    return Person(track, (x-80, 45, x+80, 300), points, body_scale=250,
                  local_flow=.1, limb_speeds={9: 1.4, 10: 1.4, 15: 0, 16: 0},
                  limb_motion={9: .8, 10: .8})


@pytest.mark.parametrize('confirmation', [False, True])
@pytest.mark.parametrize('scene', ['conversation', 'wide_boxes', 'missing_shoulders', 'stationary_hands'])
def test_fast_hands_in_empty_detection_box_space_are_not_body_contact(scene, confirmation):
    detector = BehaviorHeuristic(confirmation=True) if confirmation else FightHeuristic()
    for n in range(65):
        a, b = person(1, 180), person(2, 300)
        # People gesture between them: both hands are inside the old expanded
        # bounding-box region but remain outside the other person's body.
        for actor in (a, b):
            actor.keypoints[9:11] = [(238+4*math.sin(n), 155, .99),
                                    (242+4*math.cos(n), 155, .99)]
        if scene == 'wide_boxes':
            a, b = person(1, 130), person(2, 330)
            a.box = b.box = (30, 45, 430, 300)
        elif scene == 'missing_shoulders':
            # High-confidence wrists do not justify inventing a torso region.
            for actor in (a, b):
                actor.keypoints[5] = (*actor.keypoints[5][:2], .1)
        elif scene == 'stationary_hands':
            for actor in (a, b):
                actor.keypoints[9:11] = [(238, 155, .99), (242, 155, .99)]
                actor.limb_speeds = {}
                actor.limb_motion = {}
                actor.local_flow = 0
        result = detector.update([a, b], n*.1, .1)
        assert not result.events
        assert result.state not in {'possible_fight', 'fight_detected'}
        assert not any(candidate.signals['candidate'] for candidate in result.candidates)


@pytest.mark.parametrize('contact', ['torso', 'head', 'kick', 'withdrawal'])
def test_observed_body_contact_and_withdrawal_paths_still_support_strikes(contact):
    detector = FightHeuristic()
    events = []
    for n in range(35):
        a, b = person(1, 180), person(2, 300)
        a.limb_speeds = {9: 1.4, 10: 0, 15: 0, 16: 0}
        b.limb_speeds = {}
        a.keypoints[9] = (285, 155, .99)
        if contact == 'head':
            a.keypoints[9] = (300, 75, .99)
        elif contact == 'kick':
            a.keypoints[9] = (145, 155, .99)
            a.keypoints[15] = (285, 155, .99)
            a.limb_speeds = {15: 1.4}
            a.limb_motion = {15: .8}
        elif contact == 'withdrawal':
            a.keypoints[9] = (240 if n % 2 else 285, 155, .99)
            a.limb_speeds[9] = 1.4 if n % 2 else 0
        events.extend(detector.update([a, b], n*.1, .1).events)
    assert len(events) == 1
    assert events[0].signals['candidate']


def test_shared_translation_does_not_invent_path_through_torso():
    detector = FightHeuristic()
    for n in range(35):
        a, b = person(1, 180+n*40), person(2, 300+n*40)
        for actor in (a, b):
            actor.keypoints[9:11] = [(238+n*40, 155, .99), (242+n*40, 155, .99)]
        result = detector.update([a, b], n*.1, .1)
        assert not result.events
        assert not any(candidate.signals['candidate'] for candidate in result.candidates)


def profile_person(track, x, shoulder_width=4):
    actor = person(track, x)
    actor.keypoints[5:7] = [(x-shoulder_width/2, 110, .99), (x+shoulder_width/2, 110, .99)]
    actor.keypoints[11:13] = [(x-1, 200, .99), (x+1, 200, .99)]
    actor.keypoints[:5] = [(x, 75, .99)] * 5
    return actor


@pytest.mark.parametrize('shoulder_width', [0, 4, 12])
@pytest.mark.parametrize('framing', ['standing', 'upper_body'])
@pytest.mark.parametrize('target', ['torso', 'head'])
def test_visible_profile_body_keeps_thickness_when_projected_shoulders_collapse(shoulder_width, framing, target):
    detector = FightHeuristic()
    events = []
    for n in range(24):
        a, b = person(1, 180), profile_person(2, 300, shoulder_width)
        if framing == 'upper_body':
            a.box, b.box = (100, 45, 260, 225), (220, 45, 380, 225)
            a.body_scale = b.body_scale = 225
        a.keypoints[9] = (285, 155 if target == 'torso' else 75, .99)
        a.limb_speeds = {9: 1.4}
        b.limb_speeds = {}
        result = detector.update([a, b], n/8, .1)
        events.extend(result.events)
    assert len(events) == 1
    assert events[0].signals['body_contact']


@pytest.mark.parametrize('scene', ['near_empty_space', 'distant_wide_boxes'])
def test_profile_allowance_does_not_expand_to_hands_between_separate_bodies(scene):
    detector = FightHeuristic()
    for n in range(40):
        a, b = profile_person(1, 180), profile_person(2, 300)
        if scene == 'distant_wide_boxes':
            b = profile_person(2, 500)
            a.box = b.box = (80, 45, 600, 300)
            hand_x = 340
        else:
            hand_x = 240
        for actor in (a, b):
            actor.keypoints[9:11] = [(hand_x+4*math.sin(n), 155, .99),
                                    (hand_x+4*math.cos(n), 155, .99)]
        result = detector.update([a, b], n/8, .1)
        assert not result.events
        assert not any(candidate.signals['body_contact'] for candidate in result.candidates)
        assert not any(candidate.signals['candidate'] for candidate in result.candidates)


@pytest.mark.parametrize('framing', ['standing', 'upper_body'])
def test_image_supported_profile_strikes_survive_real_pose_stabilization(framing):
    detector, stabilizer = FightHeuristic(), PoseStabilizer()
    events = []
    for n in range(48):
        a, b = person(1, 180), profile_person(2, 300)
        if framing == 'upper_body':
            a.box, b.box = (100, 45, 260, 225), (220, 45, 380, 225)
        a.keypoints[9] = (270 if n % 2 else 305, 155, .99)
        people = stabilizer.update([a, b], n/8,
            lambda actor, index: .8 if actor.track_id == 1 and index == 9 else 0)
        events.extend(detector.update(people, n/8, .1).events)
    assert len(events) == 1
    assert events[0].signals['body_contact']
    assert events[0].signals['contact_motion'] > 0
