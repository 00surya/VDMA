"""Down posture checks use observed landmarks and time, not injury claims."""
from dataclasses import replace

import pytest

from vmd.behavior import BehaviorHeuristic
from vmd.heuristics import Person
from vmd.stabilization import PoseStabilizer


def lying(track=1, *, hidden=(), diagonal=False, x=0):
    points = [(100+x, 230, .99)]*17
    box = (80+x, 180, 400+x, 290)
    points[5:7] = [(110+x, 220, .99), (110+x, 245, .99)]
    points[11:13] = [(215+x, 220, .99), (215+x, 245, .99)]
    points[13:17] = [(280+x, 220, .99), (280+x, 245, .99),
                     (375+x, 220, .99), (375+x, 245, .99)]
    if diagonal:
        box = (80+x, 170, 320+x, 390)
        points[5:7] = [(110+x, 195, .99), (110+x, 215, .99)]
        points[11:13] = [(190+x, 250, .99), (190+x, 270, .99)]
        points[13:17] = [(250+x, 270, .99), (250+x, 290, .99),
                         (300+x, 280, .99), (300+x, 300, .99)]
    for index in hidden:
        points[index] = (0, 0, .1)
    return Person(track, box, points, body_scale=250)


def collect(detector, frames=65, source=None):
    events = []
    for n in range(frames):
        result = detector.update([source(n) if source else lying()], n*.1)
        events.extend(result.events)
    return events, result


@pytest.mark.parametrize('hidden', [(12,), (11,), (12, 14, 16), (13, 14, 15, 16)])
def test_horizontal_person_with_partial_hips_or_cropped_legs_is_reported_once(hidden):
    events, result = collect(BehaviorHeuristic(), source=lambda _: lying(hidden=hidden))
    assert [event.event_type for event in events] == ['person_down']
    assert result.state == 'person_down' and not result.trigger
    assert events[0].signals['required_stationary_seconds'] == 3
    assert set(events[0].signals['visible_hips']) == {11, 12}-set(hidden)


def test_one_hidden_hip_survives_actual_pose_filtering():
    detector, stabilizer = BehaviorHeuristic(), PoseStabilizer()
    events = []
    for n in range(65):
        filtered = stabilizer.update([lying(hidden=(12, 14, 16))], n*.1)
        assert filtered[0].pose_reliable
        events.extend(detector.update(filtered, n*.1).events)
    assert [event.event_type for event in events] == ['person_down']


def test_diagonal_compact_body_needs_a_visible_aligned_straight_leg():
    events, _ = collect(BehaviorHeuristic(), source=lambda _: lying(diagonal=True))
    assert [event.event_type for event in events] == ['person_down']
    assert events[0].signals['posture_evidence'] == 'torso_and_leg'
    assert events[0].signals['body_aspect_ratio'] < 1.2
    assert events[0].signals['horizontal_legs'] == [11, 12]
    events, _ = collect(BehaviorHeuristic(), source=lambda _: lying(diagonal=True, hidden=(13, 14, 15, 16)))
    assert events == []


def test_diagonal_bent_seated_legs_are_not_corroborating_down_evidence():
    p = lying(diagonal=True)
    p.keypoints[13:17] = [(260, 200, .99), (260, 220, .99),
                          (220, 350, .99), (220, 370, .99)]
    events, _ = collect(BehaviorHeuristic(), source=lambda _: p)
    assert events == []


def test_bent_over_person_with_observed_vertical_legs_is_not_down():
    p = lying()
    p.box = (80, 180, 400, 420)
    p.keypoints[13:17] = [(215, 330, .99), (235, 335, .99),
                          (215, 410, .99), (235, 410, .99)]
    events, _ = collect(BehaviorHeuristic(), source=lambda _: p)
    assert events == []


@pytest.mark.parametrize('hidden', [(5,), (6,), (11, 12)])
def test_missing_torso_anchors_do_not_fabricate_down_posture(hidden):
    events, _ = collect(BehaviorHeuristic(), source=lambda _: lying(hidden=hidden))
    assert events == []


def test_box_stationary_but_body_moving_cannot_finish_down_timer():
    def moving(n):
        p = lying()
        p.keypoints = [(x+n*1.5, y, confidence) for x, y, confidence in p.keypoints]
        return p
    events, _ = collect(BehaviorHeuristic(), source=moving)
    assert events == []


def test_operator_duration_changes_the_required_continuous_still_period():
    detector = BehaviorHeuristic(person_down_seconds=5)
    events = []
    for n in range(54):
        events.extend(detector.update([lying()], n*.1).events)
    assert events == []
    for n in range(54, 65):
        events.extend(detector.update([lying()], n*.1).events)
    assert len(events) == 1
    assert events[0].signals['stationary_seconds'] >= 5
    assert events[0].signals['required_stationary_seconds'] == 5


@pytest.mark.parametrize('duration', [0, -1, float('nan'), float('inf')])
def test_invalid_durations_are_rejected(duration):
    with pytest.raises(ValueError):
        BehaviorHeuristic(person_down_seconds=duration)


def test_recycled_id_jump_drops_recent_fight_and_upright_transition():
    detector = BehaviorHeuristic()
    upright = lying()
    upright.box = (80, 70, 180, 350)
    upright.keypoints[5:7] = [(110, 120, .99), (150, 120, .99)]
    upright.keypoints[11:13] = [(110, 200, .99), (150, 200, .99)]
    for n in range(6):
        detector.update([upright], n*.1)
    detector.recent_fights[1] = .5
    events = []
    for n in range(65):
        events.extend(detector.update([lying(x=500)], .6+n*.1).events)
    assert [event.event_type for event in events] == ['person_down']
    assert 1 not in detector.recent_fights


def test_down_timer_cannot_cross_track_loss_gap_or_camera_motion():
    for disruption in ('loss', 'gap', 'camera'):
        detector = BehaviorHeuristic()
        for n in range(25):
            assert not detector.update([lying()], n*.1).events
        offset = 0
        if disruption == 'loss':
            detector.update([], 2.5)
        elif disruption == 'gap':
            offset = 1
            detector.update([lying()], 3.5)
        else:
            detector.update([lying()], 2.5, camera_motion=.2)
        for n in range(25):
            assert not detector.update([lying()], 2.6+offset+n*.1).events


def test_unreliable_pose_and_nonfinite_geometry_never_trigger_down():
    for p in (replace(lying(), pose_reliable=False), replace(lying(), body_scale=float('nan'))):
        events, _ = collect(BehaviorHeuristic(), source=lambda _: p)
        assert events == []


def test_duplicate_or_unassigned_tracks_do_not_share_a_down_timer():
    for people in ([lying(), lying(x=500)], [lying(track=-1)]):
        detector = BehaviorHeuristic()
        for n in range(65):
            assert not detector.update(people, n*.1).events
