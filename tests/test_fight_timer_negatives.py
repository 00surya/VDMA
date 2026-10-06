"""Onset timing must retain contact, arm identity, and exact-frame depth gates."""
import math

import pytest

from test_bilateral_punches import slow_actor
from test_fight_timer import slow_tick
from test_fight_confirmation import interacting_people
from vmd.behavior import BehaviorHeuristic
from vmd.confirmation import FightConfirmation
from vmd.heuristics import Assessment, Rules
from vmd.stabilization import PoseStabilizer


@pytest.mark.parametrize('required', [.5, 1, 3])
@pytest.mark.parametrize('second_actor_delay', [0, .5, .75, 1])
def test_one_two_arm_embrace_does_not_start_timer_even_when_people_move_at_different_times(
        required, second_actor_delay):
    behavior = BehaviorHeuristic(Rules(threshold=.55, fight_confirmation_seconds=required),
                                 confirmation=True)
    stabilizer = PoseStabilizer()
    events, results = [], []
    last_depth = float('-inf')
    for n in range(64):
        timestamp = n/8
        a, b = slow_actor(1, 190), slow_actor(2, 300)
        fraction = lambda stamp: (1-math.cos(max(0, min(stamp, 3))*2*math.pi/3))/2
        a.keypoints[9] = (215+70*fraction(timestamp), 155, .99)
        b.keypoints[9] = (275-70*fraction(timestamp-second_actor_delay), 155, .99)
        for actor in (a, b):
            actor.keypoints[10] = actor.keypoints[9]
        people = stabilizer.update([a, b], timestamp,
                                   lambda actor, index: .25 if index in (9, 10) else 0)
        sampled = timestamp-last_depth >= 1
        if sampled:
            behavior.observe_depth(timestamp, [{'tracks': [1, 2], 'status': 'compatible'}], timestamp)
            last_depth = timestamp
        result = behavior.update(people, timestamp, .002)
        priority = (any(state.since == timestamp for state in behavior.confirmation.pending.values())
                    or any(strike['contact'] == timestamp for strikes in behavior.fight.slow_punch_strikes.values()
                           for strike in strikes))
        if priority and not sampled:
            behavior.observe_depth(timestamp, [{'tracks': [1, 2], 'status': 'compatible'}], timestamp)
            last_depth = timestamp
        events.extend(result.events)
        results.append(result)
    assert not any(result.signals.get('candidate') for result in results)
    assert any(strike.get('two_handed') for result in results
               for strike in result.signals.get('slow_punch_strikes', []))
    assert not [event for event in events if event.event_type in {'possible_fight', 'fight'}]
    assert max(result.signals.get('fight_timer_seconds', 0) for result in results) == 0


@pytest.mark.parametrize('preceding_offset', [.125, .05])
def test_previous_compatible_frame_cannot_preempt_unknown_or_bad_exact_contact_depth(preceding_offset):
    gate = FightConfirmation(Rules(fight_confirmation_seconds=.5))
    slow_tick(gate, 0, depth=None, striking=False)
    slow_tick(gate, .125, depth=None, striking=False)
    gate.observe_depth(.25-preceding_offset, [{'tracks': [1, 2], 'status': 'compatible'}], .25)
    strike = {'actor': 1, 'wrist': 9, 'start': .125, 'contact': .25, 'score': .75}
    result = slow_tick(gate, .25, [strike], depth=None)
    # Engine only learns the contact after this update, then requests its exact
    # frame. Saving here would race an uncertain/separated response for .25.
    assert not result.events
    assert result.signals['fight_timer_seconds'] == 0
    gate.observe_depth(.25, [{'tracks': [1, 2], 'status': 'uncertain'}], .25)
    result = slow_tick(gate, .375, [strike], depth=None)
    assert not result.events
    assert result.signals['fight_timer_seconds'] == 0


def test_late_exact_contact_depth_can_start_review_on_next_supported_frame():
    gate = FightConfirmation(Rules(fight_confirmation_seconds=1))
    slow_tick(gate, 0, depth=None, striking=False)
    slow_tick(gate, .125, depth=None, striking=False)
    strike = {'actor': 1, 'wrist': 9, 'start': .125, 'contact': .25, 'score': .75}
    result = slow_tick(gate, .25, [strike], depth=None)
    assert not result.events
    gate.observe_depth(.25, [{'tracks': [1, 2], 'status': 'compatible'}], .375)
    result = slow_tick(gate, .375, [strike], depth=None)
    assert [event.event_type for event in result.events] == ['possible_fight']
    assert result.signals['fight_timer_seconds'] == .125


def test_low_score_current_arm_cannot_borrow_motion_score_from_an_arm_with_bad_contact_depth():
    gate = FightConfirmation(Rules(threshold=.6, fight_confirmation_seconds=1))
    slow_tick(gate, 0, depth=None, striking=False)
    slow_tick(gate, .125, depth=None, striking=False)
    first = {'actor': 1, 'wrist': 9, 'start': .125, 'contact': .25, 'score': .75}
    result = slow_tick(gate, .25, [first])
    assert result.state == 'possible_fight'
    assert result.signals['fight_timer_seconds'] == 0
    gate.observe_depth(.375, [{'tracks': [1, 2], 'status': 'uncertain'}], .375)
    second = {'actor': 2, 'wrist': 9, 'start': .125, 'contact': .375, 'score': .95}
    candidate = Assessment(.95, pair=(1, 2), signals={
        'candidate': False, 'body_contact': False, 'slow_punch_activity': True,
        'slow_punch_score': .95, 'slow_punch_striking': True,
        'slow_punch_striking_score': .95, 'slow_punch_strikes': [first, second],
        'slow_punch_striking_contacts': [
            {'actor': 1, 'wrist': 9, 'contact': .25, 'score': .3},
            {'actor': 2, 'wrist': 9, 'contact': .375, 'score': .95}],
    })
    result = gate.update(Assessment(candidates=[candidate]), interacting_people(), .375, 0)
    # The supported older arm is now below threshold; the high-score arm has
    # uncertain depth at its own contact. Neither can advance the clock.
    assert result.signals['fight_timer_seconds'] == 0
    assert result.signals['fight_timer_state'] == 'waiting_depth'
    assert result.signals['fight_timer_reason'] == 'Checking depth at the striking contact frame'
    assert not result.events
