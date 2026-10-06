"""Separate strikes must survive withdrawal frames without becoming continuous grappling."""
import pytest

from vmd.behavior import BehaviorHeuristic
from vmd.heuristics import Rules
from vmd.stabilization import PoseStabilizer
from test_fight_confirmation import interacting_people


@pytest.mark.parametrize('depth', ['compatible', 'missing', 'separated', 'stale'])
@pytest.mark.parametrize('hold', [.3, .7, 1.4])
def test_one_supported_contact_frame_per_second_withdrawals_and_required_depth(depth, hold):
    detector = BehaviorHeuristic(Rules(threshold=.6, hold_seconds=hold), confirmation=True)
    events, ids = [], set()
    for n in range(96):
        t = n/8
        people = interacting_people()
        for person in people:
            person.local_flow = .002
            person.limb_speeds = {}
            person.limb_motion = {}
        # Validated limb evidence occupies only one frame, followed by .875s
        # of withdrawal. A still wrist without speed/image support is not a hit.
        if n % 8 == 1:
            people[0].limb_speeds = {9: 1.4}
            people[0].limb_motion = {9: .8}
        if n % 8 == 0 and depth != 'missing':
            detector.observe_depth(t, [{'tracks': [1, 2], 'status': 'compatible' if depth == 'stale' else depth}],
                                   t+3 if depth == 'stale' else t)
        result = detector.update(people, t, .002)
        if result.signals.get('episode_id'):
            ids.add(result.signals['episode_id'])
        events += [(t, event) for event in result.events]
    possible = [event for t,event in events if event.event_type == 'possible_fight']
    fights = [(t,event) for t,event in events if event.event_type == 'fight']
    if depth == 'separated':
        assert not events
        return
    assert len(ids) == 1 and len(possible) == 1
    assert bool(fights) is (depth == 'compatible')
    if fights:
        assert len(fights) == 1
        assert fights[0][1].signals['fight_timer_seconds'] >= 3
        assert fights[0][1].signals['strike_bouts'] >= 2
        assert fights[0][1].signals['depth_samples'] >= 2
        assert possible[0].signals['episode_id'] == fights[0][1].signals['episode_id']


def test_actual_stabilizer_can_follow_a_punch_and_withdrawal_each_second():
    detector, stabilizer = BehaviorHeuristic(confirmation=True), PoseStabilizer()
    events = []
    for n in range(96):
        t = n/8
        people = interacting_people()
        people[0].keypoints[9] = (285 if n >= 16 and n%8 == 2 else 185, 155, .99)
        people[1].keypoints[9] = (330, 155, .99)
        for person in people:
            person.local_flow = .003
        if n%8 == 0:
            detector.observe_depth(t, [{'tracks': [1,2], 'status': 'compatible'}], t)
        people = stabilizer.update(people, t, lambda person,joint: .8 if person.track_id == 1 and joint == 9 else 0)
        events += [(t,e) for e in detector.update(people,t,.003).events]
    assert [e.event_type for t,e in events] == ['possible_fight', 'fight']
    assert events[0][0] == 2.25  # First body-directed contact after the two-second warmup.
    assert events[1][0] == events[0][0]+3  # Three seconds of ongoing striking.
    assert events[1][1].signals['depth_samples'] >= 2


@pytest.mark.parametrize('depth', ['compatible', 'missing', 'separated'])
def test_reaching_body_without_five_percent_box_overlap_requires_matching_depth(depth):
    detector = BehaviorHeuristic(confirmation=True)
    events = []
    for n in range(80):
        t = n/8
        people = interacting_people()
        # Boxes are tight around torsos, but the tracked hand reaches the body.
        people[0].box = (100,50,250,300)
        people[1].box = (245,50,380,300)
        for person in people:
            person.local_flow = .002
            person.limb_speeds = {}
            person.limb_motion = {}
        if n%8 == 1:
            people[0].limb_speeds = {9:1.4}
            people[0].limb_motion = {9:.8}
        if n%8 == 0 and depth != 'missing':
            detector.observe_depth(t, [{'tracks': [1,2], 'status': depth}],t)
        events += detector.update(people,t,.1).events
    assert [e.event_type for e in events] == (['possible_fight','fight'] if depth == 'compatible' else [])


def test_low_threshold_does_not_turn_one_brief_contact_into_repeated_strikes():
    detector = BehaviorHeuristic(Rules(threshold=.4), confirmation=True)
    events = []
    for n in range(80):
        people = interacting_people()
        for person in people:
            person.limb_speeds = {9:1.4} if n == 8 else {}
            person.limb_motion = {9:.8} if n == 8 else {}
            person.local_flow = .002
        if n%8 == 0:
            detector.observe_depth(n/8, [{'tracks':[1,2],'status':'compatible'}],n/8)
        result = detector.update(people,n/8,.002)
        events.extend(result.events)
        assert not any(event.event_type == 'fight' for event in result.events)
    assert [event.event_type for event in events] == ['possible_fight']


def test_continuous_fast_contact_is_not_counted_as_separate_strikes_at_low_fps():
    detector = BehaviorHeuristic(confirmation=True)
    for n in range(24):
        people = interacting_people()
        people[1].limb_speeds = {}
        people[1].local_flow = 0
        detector.observe_depth(n/2,[{'tracks':[1,2],'status':'compatible'}],n/2)
        result = detector.update(people,n/2,.1)
        assert result.signals.get('strike_bouts',0) <= 1
        assert not any(event.event_type == 'fight' for event in result.events)


@pytest.mark.parametrize('fps', [2, 2.5, 8])
def test_sustained_bilateral_grapple_counts_observed_time_at_low_fps(fps):
    detector = BehaviorHeuristic(confirmation=True)
    fights = []
    last_depth = -1
    for n in range(int(10*fps)):
        t = n/fps
        if t-last_depth >= 1:
            detector.observe_depth(t,[{'tracks':[1,2],'status':'compatible'}],t)
            last_depth = t
        result = detector.update(interacting_people(),t,.1)
        fights += [event for event in result.events if event.event_type == 'fight']
        if t < 3:
            assert not fights
    assert len(fights) == 1
    assert fights[0].signals['fight_timer_seconds'] >= 3


def test_resumed_strikes_collect_fresh_depth_without_provisional_cooldown_blindness():
    detector = BehaviorHeuristic(confirmation=True)
    events = []
    for n in range(96):
        t = n/8
        people = interacting_people()
        for person in people:
            person.limb_speeds = {}
            person.limb_motion = {}
            person.local_flow = .002
        if n%8 == 1 and not 2.2 < t < 4:
            people[0].limb_speeds = {9:1.4}
            people[0].limb_motion = {9:.8}
        if n%8 == 0:
            detector.observe_depth(t,[{'tracks':[1,2],'status':'compatible'}],t)
        events += [(t,event) for event in detector.update(people,t,.002).events]
    assert [event.event_type for t,event in events] == ['possible_fight','fight']
    assert events[1][0] == 7.125  # First resumed contact 4.125s + a fresh three-second timer.
    assert events[0][1].signals['episode_id'] == events[1][1].signals['episode_id']
    assert events[1][1].signals['fight_timer_seconds'] >= 3
