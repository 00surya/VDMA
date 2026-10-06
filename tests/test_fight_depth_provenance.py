"""A local arm-depth exception must not validate unrelated fast movement."""
import pytest

from test_fight_confirmation import interacting_people
from test_fight_depth_arrival import strike, tick
from test_fight_timer import candidate_at
from vmd.behavior import BehaviorHeuristic
from vmd.confirmation import FightConfirmation
from vmd.heuristics import Assessment, Rules


def contacts(moving_status):
    return [{'actor': actor, 'wrist': 9, 'target_track': 3-actor,
             'contact_kind': 'body', 'target_joint': None,
             'status': 'compatible' if actor == 1 else moving_status,
             'geometry_match': True} for actor in (1, 2)]


def run_fast_motion(torso_status):
    behavior = BehaviorHeuristic(Rules(fight_confirmation_seconds=.5, hold_seconds=.1),
                                 confirmation=True)
    events = []
    for index in range(24):
        timestamp = index/8
        people = interacting_people()
        for person in people:
            person.local_flow = .002
        # The still person's projected hand has compatible local depth. Only
        # the other person's hand supplies repeated supported fast movement.
        people[0].limb_speeds, people[0].limb_motion = {}, {}
        active = index % 8 in (1, 2, 3)
        people[1].limb_speeds = {9: 1.4 if active else 0}
        people[1].limb_motion = {9: .8 if active else 0}
        behavior.observe_depth(timestamp, [{
            'tracks': [1, 2], 'status': torso_status, 'fight_status': 'compatible',
            'contacts': contacts('compatible' if torso_status == 'compatible' else 'separated'),
        }], timestamp)
        result = behavior.update(people, timestamp, .002)
        events.extend(result.events)
    return events


@pytest.mark.parametrize('torso_status', ['separated', 'uncertain'])
def test_fast_motion_cannot_borrow_another_static_arms_local_depth_exception(torso_status):
    events = run_fast_motion(torso_status)
    assert not any(event.event_type == 'fight' for event in events)
    if torso_status == 'separated':
        assert not events, 'Known torso separation also vetoes ordinary fast review warnings'


def test_fast_motion_retains_confirmation_with_two_fresh_compatible_original_torso_samples():
    events = run_fast_motion('compatible')
    assert [event.event_type for event in events] == ['possible_fight', 'fight']
    assert events[0].signals['episode_id'] == events[1].signals['episode_id']


def test_exact_reciprocal_guard_contacts_can_bridge_different_torso_planes():
    behavior = BehaviorHeuristic(Rules(fight_confirmation_seconds=.5), confirmation=True)
    gate = behavior.confirmation
    tick(gate, 0)
    tick(gate, .125)
    initial, reciprocal = strike(1, .25), strike(2, .875)
    events = []
    for index in range(2, 8):
        timestamp = index/8
        accepted = [initial] if index < 7 else [initial, reciprocal]
        current = initial if index < 7 else reciprocal
        if index in (2, 7):
            # Actual spatial output: torsos differ, but these specific raised
            # guard contacts were independently observed on matching surfaces.
            details = {key: value for key, value in current.items()
                       if key in {'actor', 'wrist', 'target_track', 'contact_kind', 'target_joint'}}
            behavior.observe_depth(timestamp, [{
                'tracks': [1, 2], 'status': 'separated', 'fight_status': 'compatible',
                'contacts': [dict(details, status='compatible', geometry_match=True)],
            }], timestamp)
        result = tick(gate, timestamp, accepted, current=[current])
        events.extend(result.events)
    assert [event.event_type for event in events] == ['possible_fight', 'fight']
    assert events[-1].signals['fight_timer_seconds'] == .625
    # A later original-torso separation must retain the confirmed local-contact
    # presentation during an ordinary brief withdrawal, with no second event.
    behavior.observe_depth(1, [{
        'tracks': [1, 2], 'status': 'separated', 'fight_status': 'compatible',
        'contacts': [],
    }], 1)
    result = tick(gate, 1, [initial, reciprocal])
    assert result.signals['fight_timer_state'] == 'confirmed'
    assert result.signals['fight_timer_seconds'] == .625
    assert not result.events


def test_original_torso_separation_discards_fast_cycles_even_if_an_unrelated_arm_matches():
    gate = FightConfirmation(Rules(fight_confirmation_seconds=1))
    for index in range(20):
        timestamp = index/8
        gate.observe_depth(timestamp, [{
            'tracks': [1, 2], 'status': 'compatible',
            'torso_status': 'separated' if index == 6 else 'compatible',
            'contacts': contacts('separated'),
        }], timestamp)
        # The only directional cycles happened at .25 and .5, before the
        # separated frame. A continuous velocity estimate after it is not a
        # fresh observed strike cycle, and cannot reuse the earlier quota.
        result = gate.update(Assessment(candidates=[candidate_at(timestamp)]),
                             interacting_people(), timestamp, 0)
        assert not any(event.event_type == 'fight' for event in result.events)
        if index == 7:
            assert result.signals['fight_timer_seconds'] == 0


def test_torso_separation_rebases_mixed_timer_to_the_first_accepted_slow_contact():
    gate = FightConfirmation(Rules(fight_confirmation_seconds=2))
    accepted = strike(1, .25)
    for index in range(7):
        timestamp = index/8
        detail = {key: value for key, value in accepted.items()
                  if key in {'actor', 'wrist', 'target_track', 'contact_kind', 'target_joint'}}
        gate.observe_depth(timestamp, [{
            'tracks': [1, 2], 'status': 'compatible',
            'torso_status': 'separated' if index == 6 else 'compatible',
            'contacts': [dict(detail, status='compatible')],
        }], timestamp)
        candidate = candidate_at(timestamp)
        candidate.signals.update({
            'slow_punch_activity': index >= 2, 'slow_punch_score': .8,
            'slow_punch_strikes': [accepted] if index >= 2 else [],
            'slow_punch_striking': index >= 2, 'slow_punch_striking_score': .8,
            'slow_punch_striking_contacts': [accepted] if index >= 2 else [],
        })
        gate.update(Assessment(candidates=[candidate]), interacting_people(), timestamp, 0)
    state = gate.pending[(1, 2)]
    assert state.strike_since == .25  # Fast onset at .125 has been discarded.
    assert state.fight_seconds == .5
    assert not state.samples and not state.strike_bouts
    assert state.fast_depth_after == .75


def test_recovered_fast_motion_requires_two_original_depth_samples_after_separation():
    gate = FightConfirmation(Rules(fight_confirmation_seconds=.5))
    for index in range(13):
        timestamp = index/8
        if index <= 7 or index == 12:
            gate.observe_depth(timestamp, [{
                'tracks': [1, 2], 'status': 'compatible',
                'torso_status': 'separated' if index == 6 else 'compatible',
                'contacts': [],
            }], timestamp)
        candidate = candidate_at(timestamp, cycles=False)
        candidate.signals['strike_cycle_times'] = [1.0] if timestamp >= 1.0 else []
        result = gate.update(Assessment(candidates=[candidate]), interacting_people(), timestamp, 0)
        if index < 12:
            assert not any(event.event_type == 'fight' for event in result.events)
        if index == 11:
            assert result.signals['fight_timer_seconds'] == .5
            assert result.signals['fight_timer_state'] == 'waiting_depth'
    assert [event.event_type for event in result.events] == ['fight']
    assert gate.pending[(1, 2)].fast_depth_after == .75
