"""Operator timer follows accepted striking, never wall time or a punch quota."""
import pytest

from vmd.behavior import BehaviorHeuristic
from vmd.confirmation import FightConfirmation
from vmd.heuristics import Assessment, Rules
from test_fight_confirmation import interacting_people


def candidate_at(timestamp, *, active=True, cycles=True):
    """Already validated wrist-to-body evidence; isolate temporal confirmation."""
    return Assessment(.85 if active else .1, signals={
        'candidate': active, 'body_contact': active, 'grappling': False,
        # Two observed extension/withdrawal cycles, rather than three bouts.
        'strike_cycle_times': [stamp for stamp in (.25, .5) if cycles and stamp <= timestamp],
    }, pair=(1, 2))


def tick(gate, timestamp, *, active=True, depth=None, people=None, cycles=True,
         camera_motion=0):
    if depth is not None:
        gate.observe_depth(timestamp, [{'tracks': [1, 2], 'status': depth}], timestamp)
    assessment = Assessment(candidates=[candidate_at(timestamp, active=active, cycles=cycles)])
    return gate.update(assessment, interacting_people() if people is None else people,
                       timestamp, camera_motion)


def timer(result, required):
    signals = result.signals
    assert signals['fight_timer_required_seconds'] == required
    assert signals['fight_timer_state'] in {'counting', 'paused', 'waiting_depth',
                                           'waiting_strikes', 'confirmed', 'idle'}
    assert isinstance(signals['fight_timer_reason'], str) and signals['fight_timer_reason']
    assert signals['fight_timer_seconds'] >= 0
    return signals['fight_timer_seconds']


def test_actual_body_directed_fast_strike_starts_possible_immediately_despite_long_review_hold():
    behavior = BehaviorHeuristic(Rules(hold_seconds=5), confirmation=True)
    for timestamp in (0, .125):
        people = interacting_people()
        for person in people:
            person.local_flow = .002
        # One person's real supported contacting wrist suffices for a review.
        # It is not enough by itself to confirm a fight.
        people[1].limb_speeds = {}
        people[1].limb_motion = {}
        result = behavior.update(people, timestamp, .002)
    assert result.state == 'possible_fight'
    assert [event.event_type for event in result.events] == ['possible_fight']
    assert timer(result, 3) == 0


@pytest.mark.parametrize('required', [.5, 1, 3])
def test_strike_timer_reaches_configured_seconds_without_fixed_three_bout_quota(required):
    gate = FightConfirmation(Rules(fight_confirmation_seconds=required, hold_seconds=5))
    events, values = [], []
    for n in range(int((required+2)*8)+1):
        timestamp = n/8
        result = tick(gate, timestamp, depth='compatible' if n >= 2 and n % 2 == 0 else None)
        values.append(timer(result, required))
        events.extend((timestamp, event) for event in result.events)
        if .125 <= timestamp < required+.125:
            assert not any(event.event_type == 'fight' for event in result.events)
    assert [event.event_type for _, event in events] == ['possible_fight', 'fight']
    assert events[0][0] == .125
    assert events[1][0] == pytest.approx(required+.125)
    assert events[1][1].signals['fight_timer_state'] == 'confirmed'
    assert events[1][1].signals['depth_samples'] >= 2
    assert events[0][1].signals['episode_id'] == events[1][1].signals['episode_id']
    assert values[1:5] == pytest.approx([0, .125, .25, .375], abs=.005)
    assert events[1][1].signals['strike_bouts'] < 3


@pytest.mark.parametrize('depth_kind', ['missing', 'single_repeated', 'stale', 'uncertain', 'separated'])
def test_elapsed_striking_never_confirms_without_two_fresh_compatible_depth_samples(depth_kind):
    gate = FightConfirmation(Rules(fight_confirmation_seconds=.5))
    events = []
    for n in range(33):
        timestamp = n/8
        if n >= 2:
            if depth_kind == 'single_repeated':
                gate.observe_depth(.25, [{'tracks': [1, 2], 'status': 'compatible'}], timestamp)
            elif depth_kind == 'stale':
                gate.observe_depth(timestamp-3, [{'tracks': [1, 2], 'status': 'compatible'}], timestamp)
            elif depth_kind != 'missing':
                gate.observe_depth(timestamp, [{'tracks': [1, 2], 'status': depth_kind}], timestamp)
        result = tick(gate, timestamp)
        timer(result, .5)
        events.extend(result.events)
        assert result.signals['fight_timer_state'] != 'confirmed'
    assert not any(event.event_type == 'fight' for event in events)
    if depth_kind != 'separated':
        assert result.signals['fight_timer_state'] == 'waiting_depth'


def test_short_withdrawal_freezes_timer_then_supported_return_bridges_the_gap():
    gate = FightConfirmation(Rules(fight_confirmation_seconds=1))
    for n in range(5):
        result = tick(gate, n/8, depth='compatible' if n >= 2 else None)
    before_pause = timer(result, 1)
    assert before_pause == pytest.approx(.375, abs=.005)
    for timestamp in (.625, .75, .875):
        result = tick(gate, timestamp, active=False, depth='compatible')
        assert timer(result, 1) == before_pause
        assert result.signals['fight_timer_state'] == 'paused'
        assert not any(event.event_type == 'fight' for event in result.events)
    result = tick(gate, 1, depth='compatible')
    assert timer(result, 1) == pytest.approx(.875, abs=.005)
    assert result.signals['fight_timer_state'] == 'counting'
    result = tick(gate, 1.125, depth='compatible')
    assert result.signals['fight_timer_state'] == 'confirmed'
    assert [event.event_type for event in result.events] == ['fight']


def test_motion_stopping_cannot_finish_timer_just_because_depth_keeps_arriving():
    gate = FightConfirmation(Rules(fight_confirmation_seconds=1))
    for n in range(5):
        result = tick(gate, n/8, depth='compatible' if n >= 2 else None)
    elapsed = timer(result, 1)
    for n in range(5, 40):
        result = tick(gate, n/8, active=False, depth='compatible')
        assert timer(result, 1) <= elapsed
        assert result.signals['fight_timer_state'] != 'confirmed'
        assert not any(event.event_type == 'fight' for event in result.events)
    assert result.signals['fight_timer_state'] == 'idle'
    assert result.signals['fight_timer_seconds'] == 0


@pytest.mark.parametrize('break_kind', ['separation', 'tracking_loss', 'frame_gap', 'camera_motion'])
def test_spatial_or_tracking_break_clears_timer_and_requires_a_new_depth_window(break_kind):
    gate = FightConfirmation(Rules(fight_confirmation_seconds=1, cooldown_seconds=0))
    for n in range(6):
        result = tick(gate, n/8, depth='compatible' if n >= 2 else None)
    assert timer(result, 1) > 0
    timestamp = 2 if break_kind == 'frame_gap' else .75
    result = tick(gate, timestamp,
                  depth='separated' if break_kind == 'separation' else None,
                  people=interacting_people()[:1] if break_kind == 'tracking_loss' else None,
                  camera_motion=.2 if break_kind == 'camera_motion' else 0)
    assert timer(result, 1) == 0
    assert result.signals['fight_timer_state'] != 'confirmed'
    # The old compatible samples must not satisfy the recovered interaction.
    for index in range(1, 8):
        result = tick(gate, timestamp+index/8,
                      depth='compatible' if break_kind == 'separation' and index == 1 else None)
        assert not any(event.event_type == 'fight' for event in result.events)
        assert result.signals.get('depth_samples', 0) <= 1


def test_activity_without_accepted_body_contact_has_no_timer_or_possible_incident():
    gate = FightConfirmation(Rules(fight_confirmation_seconds=.5))
    for n in range(25):
        result = tick(gate, n/8, active=False, depth='compatible')
        assert timer(result, .5) == 0
        assert result.signals['fight_timer_state'] == 'idle'
        assert not result.events


def slow_tick(gate, timestamp, strikes=(), *, striking=True, depth='compatible'):
    if depth is not None:
        gate.observe_depth(timestamp, [{'tracks': [1, 2], 'status': depth}], timestamp)
    candidate = Assessment(.75, signals={
        'candidate': False, 'body_contact': False, 'slow_punch_activity': True,
        'slow_punch_score': .75, 'slow_punch_strikes': list(strikes),
        'slow_punch_striking': striking, 'bilateral_punch_cycles': [],
    }, pair=(1, 2))
    return gate.update(Assessment(candidates=[candidate]), interacting_people(), timestamp, 0)


def test_accepted_slow_contact_starts_timer_and_reciprocal_strikes_need_no_completed_cycle_quota():
    gate = FightConfirmation(Rules(fight_confirmation_seconds=1, hold_seconds=5))
    events = []
    for n in range(25):
        timestamp = n/8
        strikes = []
        # An approach opens the pending pair at .125. Exact contacting-frame
        # depth arrives before each contact is evaluated, as in file analysis.
        if n >= 2:
            strikes.append({'actor': 1, 'wrist': 9, 'start': .125, 'contact': .25})
        if n >= 7:
            strikes.append({'actor': 2, 'wrist': 9, 'start': .5, 'contact': .875})
        result = slow_tick(gate, timestamp, strikes, striking=n >= 2)
        events.extend((timestamp, event) for event in result.events)
        if n == 2:
            assert timer(result, 1) == 0
            assert result.state == 'possible_fight'
    assert [event.event_type for _, event in events] == ['possible_fight', 'fight']
    assert events[0][0] == .25 and events[1][0] == 1.25
    assert events[1][1].signals['bilateral_punch_cycles'] == []
    assert events[0][1].signals['episode_id'] == events[1][1].signals['episode_id']


@pytest.mark.parametrize('scene', ['approach_only', 'one_sided', 'one_simultaneous_contact'])
def test_slow_activity_or_one_contact_action_cannot_confirm_on_elapsed_time(scene):
    gate = FightConfirmation(Rules(fight_confirmation_seconds=.5))
    for n in range(25):
        timestamp = n/8
        strikes = []
        if n and scene != 'approach_only':
            strikes.append({'actor': 1, 'wrist': 9, 'start': 0., 'contact': .125})
        if n and scene == 'one_simultaneous_contact':
            strikes.append({'actor': 2, 'wrist': 9, 'start': 0., 'contact': .125})
        result = slow_tick(gate, timestamp, strikes, striking=scene != 'approach_only')
        assert not any(event.event_type == 'fight' for event in result.events)
        if scene == 'approach_only':
            assert timer(result, .5) == 0
            assert result.signals['fight_timer_state'] == 'idle'
            assert not result.events


@pytest.mark.parametrize('break_kind', ['unavailable', 'stale'])
def test_missing_or_stale_depth_discards_timer_progress(break_kind):
    gate = FightConfirmation(Rules(fight_confirmation_seconds=6))
    for n in range(6):
        result = tick(gate, n/8, depth='compatible' if n >= 2 else None)
    assert timer(result, 6) > 0
    if break_kind == 'unavailable':
        gate.depth_unavailable()
        result = tick(gate, .75)
    else:
        for n in range(6, 27):
            result = tick(gate, n/8)
    assert timer(result, 6) == 0
    assert result.signals['fight_timer_state'] == 'waiting_depth'
    assert result.signals['depth_samples'] == 0
    assert not result.events


@pytest.mark.parametrize('contact_depth', ['uncertain', 'separated'])
def test_known_bad_contact_depth_cannot_reuse_older_compatible_samples_for_slow_strikes(contact_depth):
    gate = FightConfirmation(Rules(fight_confirmation_seconds=.5))
    events = []
    for n in range(25):
        timestamp = n/8
        strikes = []
        if n >= 2:
            strikes.append({'actor': 1, 'wrist': 9, 'start': .125, 'contact': .25})
        if n >= 7:
            strikes.append({'actor': 2, 'wrist': 9, 'start': .5, 'contact': .875})
        result = slow_tick(gate, timestamp, strikes, striking=n >= 2,
                           depth=contact_depth if n == 2 else 'compatible')
        events.extend(result.events)
        if n < 7:
            assert timer(result, .5) == 0
            assert not result.events
    assert not any(event.event_type == 'fight' for event in events)


def test_depth_reset_does_not_reuse_an_old_fast_cycle_to_confirm_continuous_one_way_motion():
    gate = FightConfirmation(Rules(fight_confirmation_seconds=.5))
    events = []
    for n in range(17):
        timestamp = n/8
        if n == 4:
            gate.depth_unavailable()
        if n >= 2 and n != 4:
            gate.observe_depth(timestamp, [{'tracks': [1, 2], 'status': 'compatible'}], timestamp)
        candidate = candidate_at(timestamp, cycles=False)
        candidate.signals['strike_cycle_times'] = [.25] if n >= 2 else []
        result = gate.update(Assessment(candidates=[candidate]), interacting_people(), timestamp, 0)
        events.extend(result.events)
    # The only completed cycle happened before depth failed at .5s. Fresh
    # depth plus continuous one-direction speed cannot create new repetition.
    assert [event.event_type for event in events] == ['possible_fight']
    assert result.signals['fight_timer_state'] == 'waiting_strikes'


def confirmed_gate():
    gate = FightConfirmation(Rules(fight_confirmation_seconds=.5))
    events = []
    for n in range(9):
        result = tick(gate, n/8, depth='compatible' if n >= 2 else None)
        events.extend(result.events)
    assert [event.event_type for event in events] == ['possible_fight', 'fight']
    return gate, result


def test_confirmed_timer_stays_visible_during_brief_withdrawals_without_duplicate_events():
    gate, confirmed = confirmed_gate()
    for n in range(9, 13):
        result = tick(gate, n/8, active=False, depth='compatible')
        assert result.signals['fight_timer_state'] == 'confirmed'
        assert timer(result, .5) == confirmed.signals['fight_timer_seconds']
        assert not result.events
    for n in range(13, 23):
        result = tick(gate, n/8, active=False, depth='compatible')
        assert not result.events
    assert result.signals['fight_timer_state'] == 'idle'
    assert timer(result, .5) == 0


@pytest.mark.parametrize('depth', ['uncertain', 'separated', 'unavailable'])
def test_live_confirmed_timer_clears_when_depth_no_longer_supports_the_pair(depth):
    gate, _ = confirmed_gate()
    if depth == 'unavailable':
        gate.depth_unavailable()
    result = tick(gate, 1.125, active=False, depth=None if depth == 'unavailable' else depth)
    assert result.signals['fight_timer_state'] != 'confirmed'
    assert not result.events


def test_both_depth_samples_must_still_be_fresh_when_the_motion_timer_finishes():
    gate = FightConfirmation(Rules(fight_confirmation_seconds=3))
    events = []
    for n in range(26):
        timestamp = n/8
        result = tick(gate, timestamp,
                      depth='compatible' if timestamp in (.25, 1.0) else None)
        events.extend(result.events)
        assert not any(event.event_type == 'fight' for event in result.events)
    # At 3.125s the .25s sample has expired, although the 1s sample is fresh.
    assert timer(result, 3) == 3
    assert result.signals['fresh_depth_samples'] == 1
    assert result.signals['fight_timer_state'] == 'waiting_depth'
    result = tick(gate, 3.25, depth='compatible')
    assert result.signals['fresh_depth_samples'] == 2
    assert result.signals['fight_timer_state'] == 'confirmed'
    assert [event.event_type for event in result.events] == ['fight']
    assert events[0].signals['episode_id'] == result.events[0].signals['episode_id']
