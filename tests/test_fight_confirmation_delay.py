"""Configured confirmation time remains separate from review and sampling time."""
import numpy as np
import pytest

from vmd.behavior import BehaviorHeuristic
from vmd.engine import Engine
from vmd.heuristics import Rules
from test_fight_confirmation import depth_map, interacting_people


def matched_sample(timestamp, sequence, case='matched'):
    people = interacting_people()
    depth = depth_map(case == 'separated')
    sample = {'sequence': sequence, 'source_time': timestamp, 'depth': depth}
    poses = {sequence: {'time': timestamp, 'shape': depth.shape, 'people': people}}
    if case == 'wrong_time':
        sample['source_time'] += .01
    elif case == 'wrong_sequence':
        sample['sequence'] += 1
    elif case == 'wrong_shape':
        sample['depth'] = np.ones((2, 2))
    return Engine.matched_depth_evidence(sample, poses)


def run_grapple(delay, review=.7, depth_every=.5, case='matched', duration=None):
    behavior = BehaviorHeuristic(Rules(hold_seconds=review, fight_confirmation_seconds=delay), confirmation=True)
    events = []
    for n in range(int((duration or delay+3)*8)+1):
        timestamp = n/8
        if n and timestamp % depth_every == 0:
            behavior.observe_depth(timestamp, matched_sample(timestamp, n, case), timestamp)
        result = behavior.update(interacting_people(), timestamp, .1)
        events.extend((timestamp, event) for event in result.events)
    return behavior, events


@pytest.mark.parametrize('delay', [.5, 1, 3, 6, 10])
def test_confirmation_uses_configured_duration_instead_of_fixed_three_seconds(delay):
    behavior, events = run_grapple(delay, review=.3)
    fights = [(timestamp, event) for timestamp, event in events if event.event_type == 'fight']
    assert len(fights) == 1
    timestamp, fight = fights[0]
    # Motion starts at .125s; two distinct depth samples exist by 1s. The
    # selected duration applies to motion, not another depth-span countdown.
    assert timestamp == pytest.approx(max(delay+.125, 1.0))
    assert behavior.confirmation.required_seconds == delay
    assert fight.signals['required_seconds'] == delay
    assert fight.signals['fight_timer_seconds'] >= delay
    assert fight.signals['depth_samples'] >= 2


def test_review_observation_does_not_silently_override_confirmation_setting():
    _, events = run_grapple(.5, review=5)
    fights = [(timestamp, event) for timestamp, event in events if event.event_type == 'fight']
    assert len(fights) == 1 and fights[0][0] == 1.0
    assert fights[0][1].signals['possible_required_seconds'] == 5


def test_default_confirmation_remains_three_seconds_independent_of_review_hold():
    for review in (.3, .7, 5):
        behavior = BehaviorHeuristic(Rules(hold_seconds=review), confirmation=True)
        assert behavior.confirmation.required_seconds == 3


def test_two_fresh_distinct_samples_can_satisfy_half_second_setting():
    _, events = run_grapple(.5, duration=1)
    fight = next(event for _, event in events if event.event_type == 'fight')
    assert fight.signals['depth_samples'] == 2
    assert fight.signals['confirmation_seconds'] == .5


def test_depth_sampling_can_extend_actual_delay_beyond_configured_half_second():
    _, events = run_grapple(.5, depth_every=1)
    fights = [(timestamp, event) for timestamp, event in events if event.event_type == 'fight']
    assert len(fights) == 1 and fights[0][0] == 2
    assert fights[0][1].signals['depth_samples'] == 2
    assert fights[0][1].signals['confirmation_seconds'] == 1


@pytest.mark.parametrize('case', ['wrong_time', 'wrong_sequence', 'wrong_shape', 'separated'])
def test_short_delay_does_not_bypass_exact_frame_or_depth_agreement(case):
    _, events = run_grapple(.5, case=case, duration=6)
    assert not any(event.event_type == 'fight' for _, event in events)


@pytest.mark.parametrize('case', ['repeated', 'future', 'stale', 'widely_spaced'])
def test_single_or_unusable_sample_cannot_satisfy_short_confirmation(case):
    behavior = BehaviorHeuristic(Rules(fight_confirmation_seconds=.5), confirmation=True)
    events = []
    for n in range(65):
        timestamp = n/8
        if timestamp >= .5:
            if case == 'repeated':
                source_time = .5
            elif case == 'future':
                source_time = timestamp+1
            elif case == 'stale':
                source_time = timestamp-3
            elif timestamp in (.5, 3.5, 6.5):
                source_time = timestamp
            else:
                source_time = None
            if source_time is not None:
                behavior.observe_depth(source_time, [{'tracks': [1, 2], 'status': 'compatible'}], timestamp)
        events.extend(behavior.update(interacting_people(), timestamp, .1).events)
    assert not any(event.event_type == 'fight' for event in events)
