"""False-positive regressions for the review stage, not just confirmed fights."""
import pytest

from vmd.behavior import BehaviorHeuristic
from vmd.heuristics import Rules
from test_fight_confirmation import interacting_people, step


@pytest.mark.parametrize('hold', [.7, 1.4])
def test_possible_fight_honors_camera_persistence(hold):
    behavior = BehaviorHeuristic(Rules(hold_seconds=hold), confirmation=True)
    events = []
    for n in range(30):
        result = step(behavior, n, depth=False)
        if n*.1 < hold-1e-6:
            assert result.state != 'possible_fight' and not result.events
        events += result.events
    assert [event.event_type for event in events] == ['possible_fight']


def test_brief_motion_cannot_create_review_alert():
    behavior = BehaviorHeuristic(confirmation=True)
    for n in range(30):
        people = interacting_people()
        if n > 2:
            for person in people:
                person.limb_speeds = {}
                person.limb_motion = {}
                person.local_flow = 0
        result = step(behavior, n, people, depth=False)
        assert not result.events
        assert result.state != 'possible_fight'


@pytest.mark.parametrize('separation', ['image', 'depth'])
def test_known_separation_prevents_possible_as_well_as_confirmed_fight(separation):
    behavior = BehaviorHeuristic(confirmation=True)
    for n in range(100):
        people = interacting_people()
        if separation == 'image':
            people[1].box = (270, 50, 430, 300)
        elif n % 5 == 0:
            behavior.observe_depth(n*.1, [{'tracks': [1, 2], 'status': 'separated'}], n*.1)
        result = step(behavior, n, people, depth=False)
        assert not result.events
        assert result.state not in {'possible_fight', 'fight_detected'}


def test_late_separation_stops_current_warning_and_resets_motion_evidence():
    behavior = BehaviorHeuristic(confirmation=True)
    for n in range(12):
        result = step(behavior, n, depth=False)
    assert result.state == 'possible_fight'
    behavior.observe_depth(1.2, [{'tracks': [1, 2], 'status': 'separated'}], 1.2)
    result = step(behavior, 12, depth=False)
    assert result.state == 'observing' and not result.events
    assert result.signals['motion_supported_seconds'] == 0
    assert result.signals['rejected_fight_pairs'] == [(1, 2)]


def test_review_only_depth_still_vetoes_separation_without_confirming():
    behavior = BehaviorHeuristic(confirmation=True)
    for n in range(80):
        if n % 5 == 0:
            behavior.observe_depth(n*.1, [{'tracks': [1, 2], 'status': 'separated'}], n*.1, confirm_fight=False)
        result = step(behavior, n, depth=False)
        assert not result.events


def test_depth_before_motion_vetoes_new_episode_even_after_image_separation():
    behavior = BehaviorHeuristic(confirmation=True)
    behavior.observe_depth(0, [{'tracks': [1, 2], 'status': 'separated'}], 0)
    for n in range(20):
        people = interacting_people()
        if n == 3:
            people[1].box = (270, 50, 430, 300)
        result = step(behavior, n, people, depth=False)
        assert not result.events
        assert result.state not in {'possible_fight', 'fight_detected'}
    behavior.observe_depth(2, [{'tracks': [1, 2], 'status': 'compatible'}], 2)
    events = []
    for n in range(20, 29):
        events += step(behavior, n, depth=False).events
    assert [event.event_type for event in events] == ['possible_fight']
    # The compatible sample predated this new episode; it does not prove duration.
    assert behavior.confirmation.pending[(1, 2)].depth_samples == 0


def test_pair_leaving_view_releases_current_warning():
    behavior = BehaviorHeuristic(confirmation=True)
    for n in range(12):
        result = step(behavior, n, depth=False)
    assert result.state == 'possible_fight'
    result = step(behavior, 12, people=[], depth=False)
    assert result.state == 'observing' and not result.events
    assert result.signals['rejected_fight_pairs'] == [(1, 2)]
