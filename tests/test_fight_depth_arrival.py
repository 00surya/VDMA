"""Delayed exact contact depth may validate motion, but cannot invent more motion."""
import pytest

from test_fight_confirmation import interacting_people
from vmd.confirmation import FightConfirmation
from vmd.heuristics import Assessment, Rules


def strike(actor, timestamp):
    return {'actor': actor, 'wrist': 9, 'target_track': 3-actor,
            'contact_kind': 'guard', 'target_joint': 10,
            'start': timestamp-.125, 'contact': timestamp, 'score': .8}


def observe(gate, timestamp, strikes=(), *, status='compatible',
            directed='compatible', current_time=None):
    contacts = [{key: value for key, value in item.items()
                 if key in {'actor', 'wrist', 'target_track', 'contact_kind', 'target_joint'}}
                | {'status': directed, 'geometry_match': False} for item in strikes]
    gate.observe_depth(timestamp, [{'tracks': [1, 2], 'status': status,
                                   'contacts': contacts}],
                       timestamp if current_time is None else current_time)


def tick(gate, timestamp, strikes=(), *, current=()):
    candidate = Assessment(.8, pair=(1, 2), signals={
        'candidate': False, 'body_contact': False, 'slow_punch_activity': True,
        'slow_punch_score': .8, 'slow_punch_strikes': list(strikes),
        'slow_punch_striking': bool(current), 'slow_punch_striking_score': .8,
        'slow_punch_striking_contacts': list(current), 'bilateral_punch_cycles': [],
    })
    return gate.update(Assessment(candidates=[candidate]), interacting_people(), timestamp, 0)


def prepared_gate(required=1.5):
    gate = FightConfirmation(Rules(fight_confirmation_seconds=required))
    tick(gate, 0)
    tick(gate, .125)
    return gate


def test_delayed_exact_guard_depth_credits_one_observed_contact_after_the_arm_withdraws():
    gate = prepared_gate()
    contact = strike(1, .25)
    before = tick(gate, .25, [contact], current=[contact])
    assert not before.events and not before.signals['fight_timer_started']
    observe(gate, .25, [contact], current_time=.375)
    result = tick(gate, .375, [contact])
    assert [event.event_type for event in result.events] == ['possible_fight']
    assert gate.pending[(1, 2)].strike_since == .25
    assert result.signals['fight_timer_seconds'] == 0
    # The retained contact list is historical data, not renewed movement.
    for index in range(4, 25):
        timestamp = index/8
        observe(gate, timestamp)
        result = tick(gate, timestamp, [contact])
        assert result.signals['fight_timer_seconds'] == 0
        assert not result.events
    assert not result.signals['fight_timer_started']


def test_exact_depth_arriving_after_motion_grace_cannot_start_a_historical_timer():
    gate = prepared_gate()
    contact = strike(1, .25)
    tick(gate, .25, [contact], current=[contact])
    for index in range(3, 15):
        tick(gate, index/8, [contact])
    observe(gate, .25, [contact], current_time=1.875)
    result = tick(gate, 1.875, [contact])
    assert not result.events
    assert not result.signals['fight_timer_started']


@pytest.mark.parametrize('directed', ['uncertain', 'separated', 'missing'])
def test_pair_depth_alone_cannot_start_a_guard_timer_without_its_exact_contact(directed):
    gate = prepared_gate()
    contact = strike(1, .25)
    observe(gate, .25, [] if directed == 'missing' else [contact], directed=directed)
    for index in range(2, 12):
        result = tick(gate, index/8, [contact], current=[contact])
        assert not result.events
        assert not result.signals['fight_timer_started']


@pytest.mark.parametrize('pair_status', ['compatible', 'uncertain'])
def test_supported_uncertain_contact_freezes_existing_timer_until_reciprocal_exact_depth(pair_status):
    gate = prepared_gate()
    initial, uncertain, reciprocal = strike(1, .25), strike(1, 1), strike(2, 2.25)
    observe(gate, .25, [initial])
    result = tick(gate, .25, [initial], current=[initial])
    assert [event.event_type for event in result.events] == ['possible_fight']
    for index in range(3, 8):
        result = tick(gate, index/8, [initial])
    frozen = result.signals['fight_timer_seconds']
    observe(gate, 1, [uncertain], status=pair_status, directed='uncertain')
    for index in range(8, 18):
        result = tick(gate, index/8, [initial, uncertain], current=[uncertain])
        assert result.signals['fight_timer_started']
        assert result.signals['fight_timer_seconds'] == frozen
        assert result.signals['fight_timer_state'] == 'waiting_depth'
        assert 'depth' in result.signals['fight_timer_reason'].lower()
        assert not result.events
    observe(gate, 2.25, [reciprocal])
    result = tick(gate, 2.25, [initial, uncertain, reciprocal], current=[reciprocal])
    assert [event.event_type for event in result.events] == ['fight']
    assert result.signals['fight_timer_seconds'] == 2


def test_uncertain_motion_cannot_preserve_a_timer_beyond_compatible_depth_freshness():
    gate = prepared_gate()
    initial, uncertain = strike(1, .25), strike(1, 1)
    observe(gate, .25, [initial])
    tick(gate, .25, [initial], current=[initial])
    for index in range(3, 8):
        tick(gate, index/8, [initial])
    observe(gate, 1, [uncertain], status='uncertain', directed='uncertain')
    for index in range(8, 26):
        timestamp = index/8
        if index == 20:
            uncertain = strike(1, timestamp)
            observe(gate, timestamp, [uncertain], status='uncertain', directed='uncertain')
        result = tick(gate, timestamp, [initial, uncertain], current=[uncertain])
        assert not any(event.event_type == 'fight' for event in result.events)
    assert not result.signals['fight_timer_started']
    assert result.signals['fight_timer_seconds'] == 0


def test_known_separated_directed_contact_cannot_bridge_an_existing_guard_timer():
    gate = prepared_gate()
    initial, separated, reciprocal = strike(1, .25), strike(1, 1), strike(2, 2.25)
    observe(gate, .25, [initial])
    tick(gate, .25, [initial], current=[initial])
    for index in range(3, 8):
        tick(gate, index/8, [initial])
    # Torso/another arm may match, but this contacting wrist is on another plane.
    observe(gate, 1, [separated], directed='separated')
    for index in range(8, 18):
        tick(gate, index/8, [initial, separated], current=[separated])
    observe(gate, 2.25, [reciprocal])
    result = tick(gate, 2.25, [initial, separated, reciprocal], current=[reciprocal])
    assert not any(event.event_type == 'fight' for event in result.events)
    assert result.signals['fight_timer_seconds'] == 0
