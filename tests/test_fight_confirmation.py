from dataclasses import replace

import numpy as np
import pytest

from vmd.behavior import BehaviorHeuristic, EVENT_LABELS
from vmd.heuristics import Person, Rules
from vmd.spatial import fight_depth_evidence


def interacting_people():
    people = []
    for track, x in ((1, 180), (2, 300)):
        points = [(x, 80, .99)] * 17
        points[5:7] = [(x-25, 110, .99), (x+25, 110, .99)]
        points[11:13] = [(x-20, 200, .99), (x+20, 200, .99)]
        points[9:11] = [(285 if track == 1 else 195, 155, .99), (x-30, 155, .99)]
        points[15:17] = [(x-25, 290, .99), (x+25, 290, .99)]
        people.append(Person(track, (x-80, 50, x+80, 300), points,
                             body_scale=250, local_flow=.1,
                             limb_speeds={9: 1.4, 10: 0, 15: 0, 16: 0},
                             limb_motion={9: .8}))
    return people


def depth_map(separated=False):
    depth = np.full((360, 440), .05, dtype=np.float32)
    depth[50:300, 100:260] = .3 if separated else .6
    depth[50:300, 220:380] = .85 if separated else .6
    return depth


def step(behavior, n, people=None, depth=True, camera=0, timestamp=None):
    timestamp = round(n * .1, 4) if timestamp is None else timestamp
    people = interacting_people() if people is None else people
    if depth and n % 5 == 0:
        behavior.confirmation.observe_depth(timestamp, fight_depth_evidence(depth_map(), people), timestamp)
    return behavior.update(people, timestamp, .1, camera)


def test_default_three_second_confirmation_is_independent_of_review_observation():
    behavior = BehaviorHeuristic(Rules(hold_seconds=.1), confirmation=True)
    events, pending = [], []
    for n in range(200):
        result = step(behavior, n)
        if n < 31:  # Accepted motion starts at .1s; its three-second clock ends at 3.1s.
            assert not any(e.event_type == 'fight' for e in result.events) and result.state != 'fight_detected'
        if result.state in {'checking_interaction', 'possible_fight'}:
            pending.append(result.signals['pending_pairs'])
        events += [e for e in result.events if e.event_type == 'fight']
    assert pending
    assert len(events) == 1
    assert events[0].pair == (1, 2)
    assert events[0].signals['fight_timer_seconds'] >= 3
    assert events[0].signals['depth_samples'] >= 2
    assert events[0].signals['motion_supported_seconds'] >= 3
    assert result.state == 'fight_detected'


@pytest.mark.parametrize('case', ['one_person', 'same_id', 'untracked', 'distant_people',
                                 'bad_pose', 'one_still', 'wrist_only', 'no_depth', 'short_wave'])
def test_missing_gate_never_creates_a_fight(case):
    behavior = BehaviorHeuristic(confirmation=True)
    for n in range(100):
        people = interacting_people()
        if case == 'one_person':
            people = people[:1]
        elif case == 'same_id':
            people[1].track_id = 1
        elif case == 'untracked':
            people[1].track_id = -1
        elif case == 'distant_people':
            people[1].box = tuple(v+500 if i%2 == 0 else v for i,v in enumerate(people[1].box))
            people[1].keypoints = [(x+500,y,c) for x,y,c in people[1].keypoints]
        elif case == 'bad_pose':
            people[1].pose_reliable = False
        elif case == 'one_still':
            people[1].local_flow = 0
        elif case == 'wrist_only':
            for person in people:
                person.local_flow = .002
        elif case == 'short_wave' and n >= 15:
            for person in people:
                person.local_flow = 0
                person.limb_speeds = {}
        result = step(behavior, n, people, depth=case != 'no_depth')
        assert not any(e.event_type == 'fight' for e in result.events) and result.state != 'fight_detected'
        assert behavior.recent_fights == {}


@pytest.mark.parametrize('break_kind', ['lost', 'new_id', 'nonoverlap', 'camera', 'motion', 'gap'])
def test_tracking_or_evidence_break_restarts_confirmation(break_kind):
    behavior = BehaviorHeuristic(Rules(cooldown_seconds=0), confirmation=True)
    events = []
    for n in range(130):
        people = interacting_people()
        timestamp = round(n*.1, 4)
        camera = 0
        if break_kind == 'new_id' and n >= 30:
            people[1].track_id = 3
        if n == 30:
            if break_kind == 'lost':
                people = people[:1]
            elif break_kind == 'nonoverlap':
                # Move the actual person out of reach, not just the box while
                # leaving a valid contacting wrist on their torso.
                people[1].box = tuple(v+500 if i%2 == 0 else v for i,v in enumerate(people[1].box))
                people[1].keypoints = [(x+500, y, c) for x,y,c in people[1].keypoints]
            elif break_kind == 'camera':
                camera = .2
        if break_kind == 'motion' and 30 <= n <= 47:
            for person in people:
                person.local_flow = 0
                person.limb_motion = {}
        if break_kind == 'gap' and n >= 30:
            timestamp += 1
        result = step(behavior, n, people, camera=camera, timestamp=timestamp)
        if n < 60:  # No pre-break motion can shorten the new three-second timer.
            assert not any(e.event_type == 'fight' for e in result.events)
        events += [e for e in result.events if e.event_type == 'fight']
    assert len(events) == 1
    assert events[0].signals['fight_timer_seconds'] >= 3
    assert events[0].signals['depth_samples'] >= 2


@pytest.mark.parametrize('depth_kind', ['missing', 'flat', 'separated', 'repeated', 'future', 'unknown'])
def test_unusable_depth_or_repolling_one_sample_cannot_confirm(depth_kind):
    behavior = BehaviorHeuristic(confirmation=True)
    for n in range(100):
        timestamp = n*.1
        people = interacting_people()
        if n % 5 == 0:
            readings = fight_depth_evidence(depth_map(depth_kind == 'separated'), people)
            if depth_kind == 'flat':
                readings = fight_depth_evidence(np.full((360, 440), .5), people)
            elif depth_kind == 'missing':
                readings = []
            elif depth_kind == 'unknown':
                readings = [dict(item, status='uncertain') for item in readings]
            source_time = .5 if depth_kind == 'repeated' else timestamp + 1 if depth_kind == 'future' else timestamp
            behavior.confirmation.observe_depth(source_time, readings, timestamp)
        result = step(behavior, n, people, depth=False)
        assert not any(e.event_type == 'fight' for e in result.events) and result.state != 'fight_detected'


@pytest.mark.parametrize('status', ['separated', 'unavailable', 'stale'])
def test_depth_failure_restarts_evidence_instead_of_resuming_old_timer(status):
    # The six-second timer leaves time for depth to become stale before the
    # initial interaction could confirm; interruptions must precede confirmation.
    behavior = BehaviorHeuristic(Rules(fight_confirmation_seconds=6), confirmation=True)
    events = []
    reset_at = None
    for n in range(180):
        timestamp = n*.1
        if n == 20:
            if status == 'unavailable':
                behavior.confirmation.depth_unavailable()
            elif status != 'stale':
                behavior.confirmation.observe_depth(timestamp, [{'tracks': [1, 2], 'status': status}], timestamp)
        # Stop samples long enough to exceed the age limit for the stale case.
        depth = not (20 <= n < (50 if status == 'stale' else 30))
        result = step(behavior, n, depth=depth)
        if n >= 20 and reset_at is None and result.signals.get('fight_timer_seconds') == 0:
            reset_at = timestamp
        if reset_at is None or timestamp < reset_at+6-1e-6:
            assert not any(e.event_type == 'fight' for e in result.events)
        events += [e for e in result.events if e.event_type == 'fight']
    assert reset_at is not None
    assert len(events) == 1
    assert events[0].signals['fight_timer_seconds'] >= 6
    assert events[0].signals['depth_samples'] >= 2
    assert events[0].signals['depth_source_time'] >= (5 if status == 'stale' else 3)


@pytest.mark.parametrize('ambiguity_end,expected_first_fight', [(40, 4.0), (65, 9.4)])
def test_depth_ambiguity_pauses_confirmation_and_only_preserves_still_fresh_support(ambiguity_end, expected_first_fight):
    behavior = BehaviorHeuristic(confirmation=True)
    events = []
    for n in range(110):
        timestamp = n*.1
        ambiguous = 25 <= n < ambiguity_end
        if ambiguous:
            behavior.confirmation.observe_depth(timestamp, [{'tracks': [1, 2], 'status': 'uncertain'}], timestamp)
        result = step(behavior, n, depth=not ambiguous)
        if ambiguous:
            assert result.state != 'fight_detected'
            assert not any(event.event_type == 'fight' for event in result.events)
            state = behavior.confirmation.pending[(1, 2)]
            assert state.depth_status == 'uncertain'
            # Ambiguity supplies no additional samples. Beyond freshness it
            # clears the old interval and its motion timer. Recovery still
            # needs two fresh samples, not a second full depth-span timer.
            assert state.depth_samples == (4 if n <= 45 else 0)
        events += [(timestamp, event) for event in result.events if event.event_type == 'fight']
    assert len(events) == 1
    assert events[0][0] == pytest.approx(expected_first_fight)


def test_short_pauses_retain_pair_but_do_not_count_as_supported_time():
    behavior = BehaviorHeuristic(confirmation=True)
    events = []
    for n in range(140):
        people = interacting_people()
        if n % 10 == 8:
            for person in people:
                person.local_flow = 0
        result = step(behavior, n, people)
        if n < 33:
            assert not any(e.event_type == 'fight' for e in result.events)
        events += [e for e in result.events if e.event_type == 'fight']
    assert len(events) == 1
    assert events[0].signals['motion_supported_seconds'] >= 3


def test_simultaneous_pairs_have_independent_confirmation_windows():
    behavior = BehaviorHeuristic(confirmation=True)
    events = []
    for n in range(120):
        people = interacting_people()
        if n >= 30:
            people += [replace(person, track_id=person.track_id+2,
                               box=tuple(value+600 if i%2 == 0 else value for i, value in enumerate(person.box)),
                               keypoints=[(x+600, y, c) for x, y, c in person.keypoints])
                       for person in interacting_people()]
        if n % 5 == 0:
            observations = [{'tracks': [1, 2], 'status': 'compatible'}]
            if n >= 30:
                observations += [{'tracks': [3, 4], 'status': 'compatible'}]
            behavior.confirmation.observe_depth(n*.1, observations, n*.1)
        result = step(behavior, n, people, depth=False)
        for event in result.events:
            if event.event_type != 'fight':
                continue
            events.append((n*.1, event.pair))
    assert [pair for _, pair in events] == [(1, 2), (3, 4)]
    assert events[1][0] >= 6


def test_delayed_depth_before_reacquisition_cannot_attach_to_reused_track_ids():
    behavior = BehaviorHeuristic(Rules(cooldown_seconds=0), confirmation=True)
    for n in range(30):
        step(behavior, n)
    step(behavior, 30, people=[])
    step(behavior, 31, depth=False)
    behavior.confirmation.observe_depth(2.9, [{'tracks': [1, 2], 'status': 'compatible'}], 3.2)
    assert behavior.confirmation.pending[(1, 2)].depth_last is None


def test_same_frame_depth_quality_distinguishes_compatibility_from_uncertainty():
    people = interacting_people()
    assert fight_depth_evidence(depth_map(), people)[0]['status'] == 'compatible'
    assert fight_depth_evidence(depth_map(True), people)[0]['status'] == 'separated'
    people[0].keypoints[5] = (155, 110, .1)
    people[0].keypoints[6] = (205, 110, .1)
    assert fight_depth_evidence(depth_map(), people)[0]['status'] == 'uncertain'


def test_single_supported_strike_is_review_only_and_deduplicated_without_depth():
    behavior = BehaviorHeuristic(confirmation=True)
    events = []
    for n in range(120):
        people = interacting_people()
        # One short action, then still; one warning, never a confirmed fight.
        if n > 8:
            for p in people:
                p.limb_speeds = {}
                p.limb_motion = {}
                p.local_flow = 0
        events += step(behavior, n, people, depth=False).events
    assert [e.event_type for e in events] == ['possible_fight']
    assert behavior.recent_fights == {}


def test_three_second_fight_escalation_reuses_possible_incident_id():
    behavior = BehaviorHeuristic(confirmation=True)
    events = []
    for n in range(90):
        events += step(behavior, n).events
    assert [e.event_type for e in events] == ['possible_fight', 'fight']
    assert events[0].signals['episode_id'] == events[1].signals['episode_id']
    assert events[1].signals['fight_timer_seconds'] >= 3


def test_repeated_bursts_with_brief_pauses_can_confirm_only_with_depth():
    for depth in (False, True):
        behavior = BehaviorHeuristic(confirmation=True)
        events = []
        for n in range(100):
            people = interacting_people()
            for p in people:
                # Avoid bilateral grappling; exercise repeated-strike bursts.
                p.local_flow = .1 if n % 10 < 4 else 0
                p.limb_motion = {9: .8} if n % 10 < 4 else {}
                p.limb_speeds = {9: 1.4} if n % 10 < 4 else {}
            # Prevent the close-grappling geometry while retaining overlap.
            people[1].box = (230, 50, 390, 300)
            events += step(behavior, n, people, depth=depth).events
        fights = [e for e in events if e.event_type == 'fight']
        assert bool(fights) is depth
        if fights:
            assert len(fights) == 1 and fights[0].signals['strike_bursts'] >= 2
