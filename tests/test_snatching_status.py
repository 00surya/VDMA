"""Standing-rule status explains waiting gates without changing alert criteria."""
from copy import deepcopy
from dataclasses import replace

import pytest

from test_snatching import person, scene
from vmd.behavior import BehaviorHeuristic
from vmd.heuristics import Assessment, Rules
from vmd.snatching import SnatchingHeuristic


def update(detector, people, time, **kwargs):
    result = detector.update(people, time, **kwargs)
    return result, detector.snapshot()


def test_initial_status_is_plain_bounded_and_inspection_does_not_advance_time():
    detector = SnatchingHeuristic(Rules())
    initial = detector.snapshot()
    assert initial['phase'] == 'waiting_input'
    assert initial['visible_people'] == initial['pending_pairs'] == 0
    assert initial['timestamp'] is None and initial['pairs'] == []
    initial['blockers'].clear()
    initial['visible_people'] = 99
    assert detector.diagnostics['blockers']
    assert detector.diagnostics['visible_people'] == 0
    assert detector.last_time is None and detector.previous == {}


def test_visible_seated_person_is_distinguished_from_missing_people():
    detector = SnatchingHeuristic(Rules())
    seated = replace(person(1, 400), box=(350, 60, 800, 250))
    _, status = update(detector, [seated, person(2, 540)], 0)
    assert status['visible_people'] == status['tracked_people'] == 2
    assert status['eligible_standing_people'] == 1
    assert status['nonstanding_people'] == 1
    assert status['phase'] == 'needs_people'
    assert any('Seated or rider' in reason for reason in status['blockers'])


def test_unreliable_torso_and_unassigned_duplicate_tracks_are_explained():
    detector = SnatchingHeuristic(Rules())
    bad = person(1, 400)
    bad.keypoints[5] = (*bad.keypoints[5][:2], .1)
    _, status = update(detector, [bad, person(2, 540), person(-1, 100)], 0)
    assert status['body_pose_missing_people'] == 1
    assert status['unassigned_or_duplicate_people'] == 1
    assert status['eligible_standing_people'] == 1
    assert any('Both shoulders' in reason for reason in status['blockers'])
    _, status = update(detector, [person(1, 400), person(1, 540)], .125)
    assert status['tracked_people'] == status['eligible_standing_people'] == 0
    assert status['unassigned_or_duplicate_people'] == 2


def test_warmup_then_normal_waiting_reach_is_visible_without_false_alerts():
    detector = SnatchingHeuristic(Rules())
    still = [person(1, 400), person(2, 540)]
    for p in still:
        p.limb_speeds, p.limb_motion = {}, {}
    events, first = update(detector, still, 0)
    assert events == [] and first['phase'] == 'warming_up'
    events, second = update(detector, still, .125)
    assert events == [] and second['phase'] == 'warming_wrist_motion'
    events, third = update(detector, still, .25)
    assert events == [] and third['phase'] == 'waiting_reach'
    assert third['continuous_people'] == 2 and third['warming_people'] == 0
    assert third['wrist_motion_ready_people'] == 0
    assert any('No supported wrist movement' in reason for reason in third['blockers'])


def test_reach_then_pull_reports_actor_target_and_separate_escape_depth_gates():
    detector = SnatchingHeuristic(Rules())
    update(detector, scene(0), 0)
    events, reached = update(detector, scene(.125), .125)
    assert events == [] and reached['phase'] == 'reach_seen'
    pair = reached['pairs'][0]
    assert pair['actor_track'] == 1 and pair['target_track'] == 2
    assert pair['grab_seconds'] == .125 and pair['pull_seconds'] is None
    assert pair['reach_seen'] and not pair['pull_seen']
    events, pulled = update(detector, scene(.25), .25)
    assert [event.event_type for event in events] == ['possible_snatching']
    assert pulled['phase'] == 'waiting_escape_and_depth'
    assert pulled['pairs'][0]['pull_seen'] and not pulled['pairs'][0]['escape_detected']
    assert pulled['pairs'][0]['pull_seconds'] == .25
    assert len(pulled['blockers']) == 2


@pytest.mark.parametrize('depth', ['compatible', 'uncertain', 'separated'])
def test_original_depth_status_explains_confirmation_or_veto(depth):
    detector = SnatchingHeuristic(Rules())
    update(detector, scene(0), 0)
    update(detector, scene(.125), .125)
    detector.observe_depth(.125, [{'tracks': [1, 2], 'status': depth}], .25)
    events, status = update(detector, scene(.25), .25)
    pair = status['pairs'][0]
    assert pair['depth_status'] == depth
    assert pair['depth_compatible'] is (depth == 'compatible')
    if depth == 'compatible':
        assert status['phase'] == 'waiting_escape'
        assert pair['depth_source_time'] == .125
        assert [event.event_type for event in events] == ['possible_snatching']
    else:
        assert status['phase'] == 'depth_blocked'
        assert depth in status['blockers'][0]
        assert pair['depth_source_time'] is None and events == []


def test_supported_escape_without_original_depth_stays_waiting_depth():
    detector = SnatchingHeuristic(Rules())
    events = []
    for n in range(12):
        choices, status = update(detector, scene(n/8), n/8)
        events.extend(choice for choice in choices if choice.trigger)
    assert [event.event_type for event in events] == ['possible_snatching']
    assert status['phase'] == 'waiting_depth'
    pair = status['pairs'][0]
    assert pair['escape_detected'] and pair['depth_status'] == 'waiting'
    assert pair['departure_observations'] >= 4
    assert not pair['reported']


def test_diagnostics_are_fresh_after_depth_failure_without_advancing_the_rule():
    detector = SnatchingHeuristic(Rules())
    for n in range(3):
        update(detector, scene(n/8), n/8)
    detector.depth_unavailable()
    before = detector.last_time
    status = detector.snapshot()
    assert status['pairs'][0]['depth_status'] == 'unavailable'
    assert not status['pairs'][0]['depth_compatible']
    assert detector.last_time == before


@pytest.mark.parametrize('reset', ['gap', 'rewind', 'camera'])
def test_gap_rewind_and_camera_motion_explain_cleared_pending_evidence(reset):
    detector = SnatchingHeuristic(Rules())
    for n in range(3):
        update(detector, scene(n/8), n/8)
    assert detector.pending
    timestamp = 2 if reset == 'gap' else .25 if reset == 'rewind' else .375
    events, status = update(detector, scene(timestamp), timestamp,
                            camera_motion=.2 if reset == 'camera' else 0)
    assert events == [] and status['pairs'] == [] and status['pending_pairs'] == 0
    if reset == 'camera':
        assert status['phase'] == 'camera_moving' and status['camera_moving']
        assert not status['source_gap']
    else:
        assert status['phase'] == 'source_gap' and status['source_gap']
        assert status['source_gap_seconds'] == timestamp-.25
        assert not status['camera_moving']


def test_pair_status_is_bounded_and_returned_lists_cannot_mutate_detector():
    detector = SnatchingHeuristic(Rules())
    for n in range(12):
        detector.pending[(n, n+100)] = {'actor': n, 'target': n+100, 'contact': 0}
    status = detector.snapshot()
    assert status['pending_pairs'] == 12 and len(status['pairs']) == 8
    assert status['pairs_truncated']
    before = deepcopy(detector.pending)
    status['pairs'][0]['pair'].clear()
    status['pairs'][0]['blockers'].clear()
    status['pairs'][0]['actor_track'] = 999
    assert detector.pending == before
    assert detector.snapshot()['pairs'][0]['actor_track'] == 0


def test_recent_visible_pair_cooldown_is_explained_after_episode_expires():
    detector = SnatchingHeuristic(Rules())
    for n in range(45):
        detector.update(scene(n/8, 'stand'), n/8)
    status = detector.snapshot()
    assert not status['pairs'] and status['phase'] == 'cooldown'
    assert status['eligible_standing_people'] == 2
    assert any('cooldown' in reason for reason in status['blockers'])


def test_repeated_status_reads_do_not_change_any_emitted_event_criteria():
    baseline, observed = SnatchingHeuristic(Rules()), SnatchingHeuristic(Rules())
    for n in range(36):
        timestamp = n/8
        if n == 3:
            for detector in (baseline, observed):
                detector.observe_depth(.125, [{'tracks': [1, 2], 'status': 'compatible'}], timestamp)
        expected = baseline.update(scene(timestamp), timestamp)
        observed.snapshot()
        actual = observed.update(scene(timestamp), timestamp)
        status = observed.diagnostics
        observed.snapshot()
        simplify = lambda choices: [(event.event_type, event.trigger, event.pair,
            event.signals['depth_status'], event.signals['escape_detected']) for event in choices]
        assert simplify(actual) == simplify(expected)
        if any(event.event_type == 'snatching_detected' for event in actual):
            assert status['phase'] == 'confirmed'
            assert status['pairs'][0]['reported'] and status['pairs'][0]['depth_compatible']


def test_behavior_retains_independent_snatch_diagnostics_when_fight_wins_display():
    detector = BehaviorHeuristic()
    detector.fight.update = lambda *args: Assessment(.8, 'fight_detected',
        ['Synthetic simultaneous fight'], {'episode_id': 'fixture-fight', 'fight_marker': 'retained'},
        (1, 2), False, 'fight')
    for n in range(3):
        result = detector.update(scene(n/8), n/8)
    assert result.state == 'fight_detected' and result.event_type == 'fight'
    assert result.signals['fight_marker'] == 'retained'
    status = result.signals['snatching']
    assert status == detector.snatching.snapshot()
    assert status['phase'] == 'waiting_escape_and_depth'
    assert status['pairs'][0]['actor_track'] == 1 and status['pairs'][0]['pull_seen']
    assert [event.event_type for event in result.events] == ['possible_snatching']
