"""Regression scenes for interaction evidence lost by the former hard gates."""
from vmd.heuristics import FightHeuristic, Person, Rules
from vmd.stabilization import PoseStabilizer


def actor(track, x, wrist_x=None):
    points = [(x, 90, .99)] * 17
    points[5:7] = [(x - 25, 120, .99), (x + 25, 120, .99)]
    points[11:13] = [(x - 20, 180, .99), (x + 20, 180, .99)]
    points[9:11] = [(wrist_x if wrist_x is not None else x + 30, 155, .99), (x - 30, 155, .99)]
    return Person(track, (x - 50, 75, x + 50, 340), points, body_scale=250,
                  limb_speeds={9: 0, 10: 0, 15: 0, 16: 0}, local_flow=.02)


def test_moderate_supported_interaction_alerts_within_one_second():
    detector = FightHeuristic()
    events = []
    for n in range(12):
        a, b = actor(1, 200, 300), actor(2, 330)
        a.limb_speeds[9] = .72
        result = detector.update([a, b], n * .1, .02)
        if result.trigger:
            events.append(n * .1)
    assert len(events) == 1
    assert .6 <= events[0] <= 1.0


def test_withdrawal_path_retains_same_wrist_contact_between_frames():
    detector = FightHeuristic()
    events = []
    for n in range(35):
        a, b = actor(1, 200, 330 if n % 2 else 385), actor(2, 400)
        # Extension was undersampled; measured velocity arrives during withdrawal.
        a.limb_speeds[9] = 1.2 if n % 2 else 0
        a.local_flow = b.local_flow = .1
        events.extend(detector.update([a, b], n * .1, .1).events)
    assert len(events) == 1


def test_small_image_motion_around_striking_wrist_survives_box_motion_gate():
    detector = FightHeuristic()
    stabilizer = PoseStabilizer()
    events = []
    for n in range(45):
        a, b = actor(1, 200, 315 if n % 2 else 345), actor(2, 355)
        # Most body pixels stay still; only the striking wrist has measured flow.
        a.local_flow = b.local_flow = .002
        people = stabilizer.update([a, b], n * .1,
            lambda person, joint: .8 if person.track_id == 1 and joint == 9 else 0)
        events.extend(detector.update(people, n * .1, .002).events)
    assert len(events) == 1


def test_wrist_motion_cannot_borrow_flow_from_another_stationary_joint():
    detector = FightHeuristic()
    stabilizer = PoseStabilizer()
    for n in range(45):
        a, b = actor(1, 200, 315 if n % 2 else 345), actor(2, 355)
        a.local_flow = b.local_flow = .002
        people = stabilizer.update([a, b], n * .1,
            lambda person, joint: .8 if joint == 10 else 0)
        assert not detector.update(people, n * .1, .002).events


def test_old_contact_is_not_reused_after_a_frame_gap_or_track_change():
    for discontinuity in ('gap', 'track'):
        detector = FightHeuristic()
        a, b = actor(1, 200, 330), actor(2, 400)
        detector.update([a, b], 0, .1)
        a = actor(1 if discontinuity == 'gap' else 3, 200, 290)
        a.limb_speeds[9] = 1.5
        a.local_flow = b.local_flow = .1
        result = detector.update([a, b], 2 if discontinuity == 'gap' else .1, .1)
        assert not result.trigger
        assert result.signals['wrist_speed'] == 0


def test_score_below_custom_threshold_explains_the_blocker():
    detector = FightHeuristic(Rules(threshold=.9))
    for n in range(15):
        a, b = actor(1, 200, 300), actor(2, 330)
        a.limb_speeds[9] = .72
        result = detector.update([a, b], n * .1, .02)
    assert not result.trigger
    assert any('score' in blocker.lower() for blocker in result.signals['blockers'])


def test_pixels_to_flow_to_filtered_pose_can_accumulate_moderate_punches():
    import math
    import cv2
    import numpy as np
    from vmd.vision import Motion

    motion, stabilizer, detector = Motion(), PoseStabilizer(), FightHeuristic()
    background = np.random.default_rng(3).integers(0, 30, (400, 640, 3), dtype=np.uint8)
    events = []
    for n in range(60):
        wrist_x = 330 + 20 * math.sin(n * .7)
        people = [actor(1, 200, wrist_x), actor(2, 355)]
        image = background.copy()
        cv2.rectangle(image, (int(wrist_x) - 7, 148), (int(wrist_x) + 7, 162), (245, 245, 245), -1)
        flow, camera = motion.infer(image, people, n * .1)
        for person in people:
            person.local_flow = motion.person_flow(person)
        people = stabilizer.update(people, n * .1, motion.joint_motion)
        events.extend(detector.update(people, n * .1, flow, camera).events)
    assert len(events) == 1
    assert events[0].signals['contact_motion'] > 0
    assert events[0].signals['strike_bursts'] >= 2


def test_short_torso_scale_does_not_exclude_people_within_striking_distance():
    detector = FightHeuristic()
    events = []
    for n in range(20):
        a, b = actor(1, 200, 360), actor(2, 380)
        a.body_scale = b.body_scale = 150
        a.local_flow = b.local_flow = .1
        a.limb_speeds[9] = 1.2
        events.extend(detector.update([a, b], n * .1, .1).events)
    assert len(events) == 1
