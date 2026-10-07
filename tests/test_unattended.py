"""Temporal bag fixtures test guards, not general-object model accuracy."""
import math

import pytest

from vmd.unattended import UnattendedObjects


SHAPE = (480, 640, 3)


def bag(x=250, y=315, label='backpack', confidence=.8):
    return {'label': label, 'confidence': confidence, 'box': [x, y, x + 40, y + 45], 'context_only': True}


def person(x=170, confidence=.8):
    return {'label': 'person', 'confidence': confidence, 'box': [x, 80, x + 65, 355], 'context_only': True}


def update(detector, time, items, **kwargs):
    return detector.update(items, time, frame_shape=SHAPE, **kwargs)


def attend(detector, start=0, item=None):
    for time in (start, start + 1, start + 2):
        assert update(detector, time, [item or bag(), person()]) == []


def absent(detector, start=3, end=25, item=None):
    return [event for time in range(start, end + 1) for event in update(detector, time, [item or bag()])]


@pytest.mark.parametrize('label', ['backpack', 'handbag', 'suitcase'])
def test_stationary_previously_attended_bag_alerts_at_threshold_once(label):
    detector = UnattendedObjects(threshold_seconds=10)
    attend(detector, item=bag(label=label))
    events = absent(detector, item=bag(label=label))
    assert len(events) == 1
    event = events[0]
    assert event.trigger and event.event_type == 'unattended_object'
    assert event.signals['source_time'] == 13
    assert event.signals['absent_seconds'] == 10
    assert event.signals['stationary_seconds'] == 13
    assert event.signals['prior_attendant_seen'] is True
    assert event.signals['attendant_first_seen'] == 0
    assert event.signals['attendant_last_seen'] == 2
    assert event.signals['attendant_samples'] == 3
    assert event.signals['object_label'] == label
    assert event.signals['ownership_verified'] is False
    assert event.signals['contents_verified'] is False
    assert 'bomb' not in ' '.join(event.reasons).lower()
    assert detector.snapshot()['tracks'][0]['episode_id'] == event.signals['episode_id']


def test_default_is_sixty_seconds_of_absence_not_time_since_first_seen():
    detector = UnattendedObjects()
    attend(detector)
    events = absent(detector, end=63)
    assert len(events) == 1
    assert events[0].signals['source_time'] == 63
    assert events[0].signals['absent_seconds'] == 60


def test_snapshot_countdown_is_source_time_and_reading_it_never_advances_evidence():
    detector = UnattendedObjects(10)
    attend(detector)
    absent(detector, end=8)
    first = detector.snapshot()
    assert first['source_time'] == 8
    assert first['tracks'][0]['status'] == 'counting'
    assert first['tracks'][0]['absent_seconds'] == 5
    assert detector.snapshot() == first
    first['tracks'][0]['box'][0] = -999
    assert detector.snapshot()['tracks'][0]['box'][0] == 250


def test_object_never_observed_with_attendant_does_not_alert():
    detector = UnattendedObjects(10)
    assert absent(detector, start=0, end=100) == []
    assert detector.snapshot()['tracks'][0]['status'] == 'waiting_for_attendant'


def test_nearby_person_keeps_stationary_bag_attended():
    detector = UnattendedObjects(10)
    assert [event for time in range(40) for event in update(detector, time, [bag(), person()])] == []
    assert detector.snapshot()['tracks'][0]['status'] == 'attended'


def test_far_people_do_not_arm_static_bag():
    detector = UnattendedObjects(10)
    assert [event for time in range(40) for event in update(detector, time, [bag(), person(500)])] == []


def test_nearby_return_interrupts_absence_and_rearms_only_with_new_attendance():
    detector = UnattendedObjects(10)
    attend(detector)
    assert absent(detector, end=11) == []
    attend(detector, start=12)
    events = absent(detector, start=15, end=26)
    assert len(events) == 1
    assert events[0].signals['source_time'] == 25
    assert events[0].signals['attendant_first_seen'] == 12


def test_noncontinuous_one_frame_visitors_cannot_accumulate_attendance():
    detector = UnattendedObjects(10)
    for time in range(30):
        items = [bag(), person()] if time % 3 == 0 else [bag()]
        assert update(detector, time, items) == []
    assert detector.snapshot()['tracks'][0]['prior_attendant_seen'] is False


def test_multiple_nearby_people_make_association_unarmed():
    detector = UnattendedObjects(10)
    for time in range(3):
        assert update(detector, time, [bag(), person(), person(310)]) == []
    assert absent(detector) == []


def test_abrupt_alternating_attendants_do_not_accumulate_attendance():
    detector = UnattendedObjects(10)
    for time in range(5):
        assert update(detector, time, [bag(), person(150 if time % 2 else 310)]) == []
    assert absent(detector, start=5) == []


def test_weak_nearby_person_blocks_claim_of_continuous_absence():
    detector = UnattendedObjects(10)
    attend(detector)
    for time in range(3, 30):
        assert update(detector, time, [bag(), person(confidence=.3)]) == []


def test_carried_bag_inside_person_box_never_arms():
    detector = UnattendedObjects(10)
    item = bag(x=180, y=180)
    attend(detector, item=item)
    assert absent(detector, item=item) == []


def test_walking_bag_and_person_never_establish_stationarity():
    detector = UnattendedObjects(10)
    for time in range(50):
        x = 100 + time * 4
        assert update(detector, time, [bag(x=x), person(x=x - 80)]) == []
    assert detector.snapshot()['tracks'][0]['prior_attendant_seen'] is False


def test_slow_accumulated_bag_drift_uses_fixed_anchor_not_last_frame():
    detector = UnattendedObjects(10)
    attend(detector)
    for time in range(3, 35):
        assert update(detector, time, [bag(x=250 + (time - 2))]) == []


def test_moving_bag_breaks_old_absence_even_if_nearest_box_can_match():
    detector = UnattendedObjects(10)
    attend(detector)
    assert absent(detector, end=10) == []
    assert update(detector, 11, [bag(x=260)]) == []
    assert absent(detector, start=12, item=bag(x=260)) == []


def test_detector_jitter_within_anchor_tolerance_preserves_stationary_episode():
    detector = UnattendedObjects(10)
    attend(detector)
    events = []
    for time in range(3, 25):
        events.extend(update(detector, time, [bag(x=250 + (time % 2) * 2)]))
    assert len(events) == 1
    assert events[0].signals['source_time'] == 13


def test_bag_occlusion_is_not_counted_as_unattended_evidence():
    detector = UnattendedObjects(10)
    attend(detector)
    assert absent(detector, end=10) == []
    assert update(detector, 11, []) == []
    assert absent(detector, start=12) == []


def test_duplicate_results_do_not_advance_timer_or_count_as_attendance():
    detector = UnattendedObjects(10)
    for _ in range(20):
        assert update(detector, 0, [bag(), person()]) == []
    assert absent(detector, start=1) == []
    detector.reset()
    attend(detector)
    absent(detector, end=8)
    for _ in range(20):
        assert update(detector, 8, [bag()]) == []
        assert detector.snapshot()['tracks'][0]['absent_seconds'] == 5


@pytest.mark.parametrize('kind', ['backwards', 'gap', 'camera_motion', 'shape', 'error', 'stale', 'missing'])
def test_discontinuities_cannot_complete_old_absence(kind):
    detector = UnattendedObjects(10)
    attend(detector)
    assert absent(detector, end=11) == []
    if kind == 'backwards':
        update(detector, 2, [bag()])
        start = 3
    elif kind == 'gap':
        update(detector, 16, [bag()])
        start = 17
    elif kind == 'camera_motion':
        update(detector, 12, [bag()], camera_motion=.10)
        start = 13
    elif kind == 'shape':
        detector.update([bag()], 12, frame_shape=(500, 640))
        start = 13
    elif kind in ('error', 'stale'):
        update(detector, 12, [bag()], status=kind)
        start = 13
    else:
        update(detector, 12, None)
        start = 13
    assert absent(detector, start=start, end=start + 20) == []


@pytest.mark.parametrize('invalid', [
    {'label': 'person', 'confidence': .9, 'box': [math.nan, 0, 10, 20]},
    {'label': 'person', 'confidence': .9, 'box': [0, 0, 10, math.inf]},
    {'label': 'person', 'confidence': .9, 'box': [20, 0, 10, 20]},
    {'label': 'person', 'confidence': .9, 'box': [0, 0, 700, 20]},
    {'label': 'person', 'confidence': True, 'box': [0, 0, 10, 20]},
    {'label': [], 'confidence': .9, 'box': [0, 0, 10, 20]},
    'invalid detection',
])
def test_invalid_observations_cannot_turn_unknown_people_into_absence(invalid):
    detector = UnattendedObjects(10)
    attend(detector)
    absent(detector, end=11)
    assert update(detector, 12, [bag(), invalid]) == []
    assert absent(detector, start=13) == []


@pytest.mark.parametrize('value', [True, 9, 3601, math.nan, math.inf, None])
def test_threshold_validation(value):
    with pytest.raises(ValueError):
        UnattendedObjects(value)


@pytest.mark.parametrize('value', [0, -1, True, 129, 2.5])
def test_bounded_track_configuration(value):
    with pytest.raises(ValueError):
        UnattendedObjects(max_tracks=value)


def test_new_tracks_are_bounded_without_truncating_people_presence():
    detector = UnattendedObjects(10, max_tracks=2)
    items = [bag(x=10, y=10), bag(x=180, y=10), bag(x=350, y=10), person()]
    for time in range(5):
        update(detector, time, items)
        assert len(detector.tracks) <= 2
    for time in range(5, 10):
        update(detector, time, [])
    assert detector.tracks == {}


def test_overfull_sample_is_unknown_not_truncated_to_bags():
    detector = UnattendedObjects(10)
    attend(detector)
    absent(detector, end=11)
    assert update(detector, 12, [bag()] * 101 + [person()]) == []
    assert detector.tracks == {}


def test_duplicate_overlapping_bags_do_not_reassign_or_complete_timer():
    detector = UnattendedObjects(10)
    attend(detector)
    absent(detector, end=11)
    assert update(detector, 12, [bag(x=248), bag(x=252)]) == []
    assert absent(detector, start=13) == []


def test_other_object_classes_and_low_confidence_bags_are_not_candidates():
    detector = UnattendedObjects(10)
    for time in range(30):
        assert update(detector, time, [bag(label='bench'), bag(confidence=.4), person()]) == []
    assert detector.tracks == {}


def test_two_spatially_separate_bags_keep_independent_episodes():
    detector = UnattendedObjects(10)
    for time in range(3):
        update(detector, time, [bag(x=70), bag(x=420), person(0), person(500)])
    events = []
    for time in range(3, 25):
        events.extend(update(detector, time, [bag(x=70), bag(x=420)]))
    assert len(events) == 2
    assert len({event.signals['episode_id'] for event in events}) == 2


def test_reported_latch_survives_brief_occlusion_and_reattendance():
    detector = UnattendedObjects(10)
    attend(detector)
    events = absent(detector, end=14)
    episode_id = events[0].signals['episode_id']
    update(detector, 15, [])
    attend(detector, start=16)
    assert absent(detector, start=19, end=40) == []
    assert detector.snapshot()['tracks'][0]['episode_id'] == episode_id


def test_half_hertz_samples_use_source_duration_and_three_attendance_observations():
    detector = UnattendedObjects(10)
    for time in (0, 2, 4):
        update(detector, time, [bag(), person()])
    events = []
    for time in range(6, 25, 2):
        events.extend(update(detector, time, [bag()]))
    assert len(events) == 1
    assert events[0].signals['source_time'] == 16
    assert events[0].signals['absent_seconds'] == 10
