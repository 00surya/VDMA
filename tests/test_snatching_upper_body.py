"""A cropped camera can observe departure without inventing hidden ankle motion."""
import pytest

from vmd.heuristics import Rules
from vmd.snatching import SnatchingHeuristic, neck_region
from test_snatching import person, scene


def cropped_sequence(action='run', depth='compatible', *, visible_legs=False, torso_flow=True):
    detector, events = SnatchingHeuristic(Rules()), []
    for n in range(36):
        timestamp = n/8
        people = scene(timestamp, action)
        actor = people[0]
        # The tracked torso remains visible; legs are outside the image.
        if not visible_legs:
            for index in (15, 16):
                x, y, _ = actor.keypoints[index]
                actor.keypoints[index] = (x, y, .05)
                actor.limb_speeds.pop(index, None)
                actor.limb_motion.pop(index, None)
        if not torso_flow and timestamp > .25:
            actor.local_flow = 0
        if action == 'disappear' and timestamp >= .5:
            people = people[1:]
        if n == 3 and depth:
            detector.observe_depth(.125, [{'tracks': [1, 2], 'status': depth}], timestamp)
        events.extend(event for event in detector.update(people, timestamp) if event.trigger)
    return events


def test_upper_body_rapid_departure_upgrades_only_with_original_grab_depth():
    events = cropped_sequence()
    assert [event.event_type for event in events] == ['possible_snatching', 'snatching_detected']
    assert events[0].signals['episode_id'] == events[1].signals['episode_id']
    assert events[1].signals['escape_support'] == 'visible_torso'
    assert events[1].signals['depth_source_time'] == .125
    assert any('legs were outside' in reason for reason in events[1].reasons)


@pytest.mark.parametrize('depth', [None, 'uncertain', 'separated'])
def test_cropped_departure_cannot_bypass_missing_or_unreliable_grab_depth(depth):
    assert [event.event_type for event in cropped_sequence(depth=depth)] == ['possible_snatching']


@pytest.mark.parametrize('action', ['walk', 'stand', 'victim_runs', 'other_runs', 'disappear'])
def test_cropped_scene_needs_observed_rapid_departure_of_the_original_actor(action):
    assert [event.event_type for event in cropped_sequence(action)] == ['possible_snatching']


def test_visible_stationary_legs_or_unsupported_torso_do_not_use_cropped_fallback():
    assert [event.event_type for event in cropped_sequence('no_legs', visible_legs=True)] == ['possible_snatching']
    assert [event.event_type for event in cropped_sequence(torso_flow=False)] == ['possible_snatching']


def test_collar_allowance_covers_visible_shoulder_without_expanding_to_full_box():
    detector, events = SnatchingHeuristic(Rules()), []
    for n, wrist_x in enumerate((640, 577, 627)):
        actor, target = person(1, 680, (wrist_x, 102, .99)), person(2, 540)
        if n == 1:
            region, _ = neck_region(target)
            assert region[2] < target.box[2]
        events.extend(event for event in detector.update([actor, target], n/8) if event.trigger)
    assert [event.event_type for event in events] == ['possible_snatching']


def test_departure_depth_does_not_replace_the_original_grab_observation():
    detector, events = SnatchingHeuristic(Rules()), []
    for n in range(20):
        timestamp = n/8
        people = scene(timestamp)
        if n == 3:
            detector.observe_depth(.125, [{'tracks': [1, 2], 'status': 'compatible'}], timestamp)
        if n == 4:
            # This is the first depth frame after the pull. Separation while
            # leaving cannot retroactively invalidate observed grab contact.
            detector.observe_depth(.375, [{'tracks': [1, 2], 'status': 'uncertain'}], timestamp)
        events.extend(event for event in detector.update(people, timestamp) if event.trigger)
    assert [event.event_type for event in events] == ['possible_snatching', 'snatching_detected']
    assert events[-1].signals['depth_source_time'] == .125


@pytest.mark.parametrize('quality', ['uncertain', 'separated'])
def test_known_incompatible_grab_depth_suppresses_review_and_escape(quality):
    detector, events = SnatchingHeuristic(Rules()), []
    for n in range(24):
        timestamp = n/8
        if n == 2:
            detector.observe_depth(.125, [{'tracks': [1, 2], 'status': quality}], timestamp)
        if n == 4:
            # Compatible depth during withdrawal is not evidence at the grab.
            detector.observe_depth(.5, [{'tracks': [1, 2], 'status': 'compatible'}], timestamp)
        choices = detector.update(scene(timestamp), timestamp)
        events.extend(event for event in choices if event.trigger)
        assert choices == []
    assert events == []
    state = detector.pending[(1, 2)]
    assert state['possible'] and not state.get('possible_reported')
    assert state['running_at'] is not None
    assert state['depth_status'] == quality


def test_late_compatible_original_grab_can_release_one_review_then_one_upgrade():
    detector, events = SnatchingHeuristic(Rules()), []
    for n in range(24):
        timestamp = n/8
        if n == 2:
            detector.observe_depth(0, [{'tracks': [1, 2], 'status': 'uncertain'}], timestamp)
        if n == 4:
            detector.observe_depth(.125, [{'tracks': [1, 2], 'status': 'compatible'}], timestamp)
        choices = detector.update(scene(timestamp), timestamp)
        if n < 4:
            assert not choices
        events.extend((timestamp, event) for event in choices if event.trigger)
    assert [event.event_type for _, event in events] == ['possible_snatching', 'snatching_detected']
    assert events[0][0] == .5
    assert events[0][1].signals['episode_id'] == events[1][1].signals['episode_id']
    assert events[1][1].signals['depth_source_time'] == .125


def test_late_incompatible_grab_removes_current_warning_without_reemitting_history():
    detector, events = SnatchingHeuristic(Rules()), []
    for n in range(16):
        timestamp = n/8
        if n == 3:
            detector.observe_depth(.125, [{'tracks': [1, 2], 'status': 'separated'}], timestamp)
        if n == 5:
            detector.observe_depth(.25, [{'tracks': [1, 2], 'status': 'compatible'}], timestamp)
        choices = detector.update(scene(timestamp, 'stand'), timestamp)
        if 3 <= n < 5:
            assert not choices
        events.extend(event for event in choices if event.trigger)
    assert [event.event_type for event in events] == ['possible_snatching']


def test_depth_worker_failure_cannot_remove_a_known_grab_separation_veto():
    detector = SnatchingHeuristic(Rules())
    for n in range(12):
        timestamp = n/8
        if n == 2:
            detector.observe_depth(.125, [{'tracks': [1, 2], 'status': 'separated'}], timestamp)
        if n == 3:
            detector.depth_unavailable()
        assert detector.update(scene(timestamp), timestamp) == []
