"""Slower bilateral punches still require temporal, pose and depth evidence."""
import pytest

from test_bilateral_punches import slow_people
from vmd.behavior import BehaviorHeuristic
from vmd.heuristics import Rules
from vmd.stabilization import PoseStabilizer


def observe_slow(*, period=3, scene='punch', depth='compatible', threshold=.6,
                 hold=.7, confirmation=3, duration=18, interruption=None):
    behavior = BehaviorHeuristic(Rules(threshold=threshold, hold_seconds=hold,
        fight_confirmation_seconds=confirmation), confirmation=True)
    stabilizer = PoseStabilizer()
    results, events = [], []
    last_depth = float('-inf')

    def sample(timestamp):
        status = interruption if interruption and 8 <= timestamp < 9 else depth
        if status == 'unavailable':
            behavior.depth_unavailable()
        elif status != 'missing':
            source_time = 1 if status == 'repeated' else timestamp
            status = 'compatible' if status in {'repeated', 'review_only'} else status
            behavior.observe_depth(source_time, [{'tracks': [1, 2], 'status': status}], timestamp,
                                   confirm_fight=depth != 'review_only')

    for n in range(int(duration*8)):
        timestamp = n/8
        people = stabilizer.update(slow_people(timestamp, period, scene), timestamp,
                                   lambda actor, index: .25 if index == 9 else 0)
        # Match file Engine ordering: scheduled depth before rules, new contact
        # priority after rules. Never manufacture a second same-frame update.
        sampled = timestamp-last_depth >= 1 or bool(interruption and timestamp == 8)
        if sampled:
            sample(timestamp)
            last_depth = timestamp
        result = behavior.update(people, timestamp, .002)
        priority = (any(state.since == timestamp for state in behavior.confirmation.pending.values())
                    or any(strike['contact'] == timestamp for strikes in behavior.fight.slow_punch_strikes.values()
                           for strike in strikes))
        if priority and not sampled:
            sample(timestamp)
            last_depth = timestamp
        results.append((timestamp, result))
        events.extend((timestamp, event) for event in result.events)
    return behavior, results, events


@pytest.mark.parametrize('period', [2.5, 3, 4, 5])
def test_slow_reciprocal_punches_confirm_once_through_real_stabilizer(period):
    _, results, events = observe_slow(period=period)
    fights = [(time, event) for time, event in events if event.event_type == 'fight']
    assert len(fights) == 1  # Direct confirmation must not emit again as a review upgrade.
    timestamp, event = fights[0]
    assert not any(result.signals.get('candidate') for _, result in results)
    strikes = [strike for strike in event.signals['slow_punch_strikes'] if not strike.get('two_handed')]
    assert {strike['actor'] for strike in strikes} == {1, 2}
    assert max(strike['contact'] for strike in strikes)-min(strike['contact'] for strike in strikes) >= .5
    assert event.signals['fight_timer_seconds'] >= 3
    assert event.signals['depth_samples'] >= 2
    assert event.score >= .6
    assert timestamp >= 3
    assert len({event.signals['episode_id'] for _, event in events}) == 1


@pytest.mark.parametrize('depth', ['missing', 'uncertain', 'separated', 'repeated', 'review_only'])
def test_slow_motion_never_bypasses_required_depth(depth):
    _, _, events = observe_slow(depth=depth)
    assert not any(event.event_type == 'fight' for _, event in events)
    if depth == 'separated':
        assert not events


@pytest.mark.parametrize('scene', ['one_sided', 'wave', 'handshake', 'hug', 'shoulder_pat', 'translation', 'static'])
def test_same_depth_and_ordinary_or_unilateral_motion_are_not_slow_fights(scene):
    _, _, events = observe_slow(scene=scene)
    assert not [event for _, event in events if event.event_type == 'fight']
    if scene == 'one_sided':
        assert [event.event_type for _, event in events] == ['possible_fight']
    else:
        assert not [event for _, event in events if event.event_type == 'possible_fight']


@pytest.mark.parametrize('status', ['separated', 'uncertain', 'unavailable'])
def test_depth_break_blocks_confirmation_and_reliable_recovery_preserves_incident(status):
    _, results, events = observe_slow(interruption=status, duration=25)
    fights = [event for _, event in events if event.event_type == 'fight']
    # A brief veto resets evidence while retaining the same episode's handling.
    assert len(fights) == 1
    assert len({event.signals['episode_id'] for event in fights}) == len(fights)
    resumed = [(time, result) for time, result in results if time >= 8 and result.state == 'fight_detected']
    assert resumed
    timestamp, result = resumed[0]
    if status == 'uncertain':
        # Brief ambiguity adds no support and cannot confirm while uncertain,
        # but the preceding compatible samples can remain fresh until recovery.
        assert not any(result.state == 'fight_detected' for time, result in results if 8 <= time < 9)
        assert timestamp >= 9
    else:
        assert timestamp >= 12
    assert result.signals['fight_timer_seconds'] >= 3
    assert result.signals['depth_samples'] >= 2
    if status != 'uncertain':
        assert all(any(strike['actor'] == actor and strike['contact'] >= 9
                       for strike in result.signals['slow_punch_strikes']) for actor in (1, 2))


def test_threshold_and_separate_confirmation_setting_control_slow_path():
    _, _, events = observe_slow(threshold=.95)
    assert not events
    _, _, events = observe_slow(threshold=.4, hold=5, confirmation=6)
    fights = [event for _, event in events if event.event_type == 'fight']
    assert len(fights) == 1
    assert fights[0].signals['required_seconds'] == 6
    assert fights[0].signals['fight_timer_seconds'] >= 6
