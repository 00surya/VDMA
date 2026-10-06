"""A supported punch can meet a raised defensive arm before reaching the torso."""
from copy import deepcopy

import pytest

from test_bilateral_punches import slow_actor
from vmd.behavior import BehaviorHeuristic
from vmd.heuristics import FightHeuristic, Rules, guard_contact


def guard_pair(wrist_x=283):
    actor, target = slow_actor(1, 200), slow_actor(2, 360)
    actor.keypoints[9] = (wrist_x, 125, .99)
    actor.keypoints[7] = (225, 140, .99)
    target.keypoints[10] = (290, 125, .99)
    target.keypoints[8] = (320, 145, .99)
    actor.limb_speeds, actor.limb_motion = {9: .25}, {9: .4}
    target.limb_speeds, target.limb_motion = {}, {}
    return actor, target


def transform(person, scale=1, mirrored=False):
    sign = -1 if mirrored else 1
    person.keypoints = [(sign*x*scale, y*scale, confidence) for x, y, confidence in person.keypoints]
    x1, y1, x2, y2 = person.box
    left, right = sorted((sign*x1*scale, sign*x2*scale))
    person.box = (left, y1*scale, right, y2*scale)
    person.body_scale *= scale


@pytest.mark.parametrize('scale', [.5, 1, 2])
@pytest.mark.parametrize('mirrored', [False, True])
def test_supported_forward_path_can_meet_a_bent_upper_guard(scale, mirrored):
    actor, target = guard_pair()
    old_actor, old_target = guard_pair(265)
    for person in (actor, target, old_actor, old_target):
        transform(person, scale, mirrored)
    contact = guard_contact(actor, target, 9, Rules(), old_actor=old_actor, old_target=old_target)
    assert contact == {'contact_kind': 'guard', 'target_track': 2, 'target_joint': 10}


@pytest.mark.parametrize('case', ['low_handshake', 'overhead_wave', 'straight_arm', 'far_from_body',
                                 'guard_hidden', 'elbow_hidden', 'nose_hidden', 'retreating', 'stationary'])
def test_nearby_hands_alone_do_not_establish_a_defensive_guard_contact(case):
    actor, target = guard_pair()
    old_actor, old_target = guard_pair(265)
    if case == 'low_handshake':
        target.keypoints[10], actor.keypoints[9] = (290, 170, .99), (283, 170, .99)
    elif case == 'overhead_wave':
        target.keypoints[10], actor.keypoints[9] = (290, 60, .99), (283, 60, .99)
    elif case == 'straight_arm':
        target.keypoints[8] = (337.5, 117.5, .99)
    elif case == 'far_from_body':
        target.keypoints[10], actor.keypoints[9] = (245, 125, .99), (238, 125, .99)
    elif case in {'guard_hidden', 'elbow_hidden', 'nose_hidden'}:
        index = {'guard_hidden': 10, 'elbow_hidden': 8, 'nose_hidden': 0}[case]
        x, y, _ = target.keypoints[index]
        target.keypoints[index] = (x, y, .1)
    elif case == 'retreating':
        old_actor.keypoints[9] = (286, 125, .99)
    elif case == 'stationary':
        old_actor = deepcopy(actor)
    assert guard_contact(actor, target, 9, Rules(), old_actor=old_actor, old_target=old_target) is None


def test_shared_translation_cannot_create_a_forward_guard_strike():
    actor, target = guard_pair()
    old_actor, old_target = deepcopy(actor), deepcopy(target)
    for person in (actor, target):
        person.keypoints = [(x+30, y+10, confidence) for x, y, confidence in person.keypoints]
        person.box = tuple(value+(30 if index % 2 == 0 else 10) for index, value in enumerate(person.box))
    assert guard_contact(actor, target, 9, Rules(), old_actor=old_actor, old_target=old_target) is None


def run_guard_path(xs, *, image_support=True):
    heuristic = FightHeuristic(Rules(threshold=.55))
    results, accepted = [], []
    for index, x in enumerate(xs):
        timestamp = index/8
        actor, target = guard_pair(x)
        if not image_support:
            # A moving wrist must not borrow image motion from the other arm.
            actor.limb_motion = {10: .4}
        result = heuristic.update([actor, target], timestamp, .002)
        results.append(result)
        accepted.extend(strike for strike in result.signals.get('slow_punch_strikes', [])
                        if strike['contact'] == timestamp)
    return results, accepted


def test_coherent_inward_motion_accepts_guard_contact_with_specific_target_joint():
    results, accepted = run_guard_path([220, 235, 253, 270, 283, 270, 253, 235, 220])
    assert not any(result.signals.get('candidate') for result in results), 'Fast striking must not explain this case'
    assert len(accepted) == 1
    strike = accepted[0]
    assert strike['actor'] == 1 and strike['wrist'] == 9
    assert strike['contact_kind'] == 'guard'
    assert strike['target_track'] == 2 and strike['target_joint'] == 10
    assert strike['start'] < strike['contact']


def test_specific_guard_interception_takes_precedence_over_padded_body_region():
    heuristic = FightHeuristic(Rules(threshold=.55))
    accepted = []
    for index, x in enumerate((220, 303)):
        actor, target = guard_pair(x)
        # The raised arm overlaps the padded head/torso region in the image;
        # the contacting guard joint must still identify the depth comparison.
        target.keypoints[10] = (310, 125, .99)
        target.keypoints[8] = (330, 145, .99)
        result = heuristic.update([actor, target], index/8, .002)
        accepted.extend(strike for strike in result.signals.get('slow_punch_strikes', [])
                        if strike['contact'] == index/8)
    assert len(accepted) == 1
    assert accepted[0]['contact_kind'] == 'guard'
    assert accepted[0]['target_joint'] == 10


@pytest.mark.parametrize('xs', [[220, 283, 283, 283], [220, 253, 220, 220, 283, 283]])
def test_one_supported_guard_contact_can_be_observed_but_cannot_confirm_a_fight(xs):
    _, accepted = run_guard_path(xs)
    # At low FPS a real punch may occupy one sampled inward frame. A specific
    # raised guard plus independent same-wrist image motion permits that contact.
    assert len(accepted) == 1 and accepted[0]['contact_kind'] == 'guard'
    behavior = BehaviorHeuristic(Rules(threshold=.55, fight_confirmation_seconds=.5), confirmation=True)
    for index, x in enumerate(xs+[xs[-1]]*30):
        timestamp = index/8
        result = behavior.update(list(guard_pair(x)), timestamp, .002)
        behavior.observe_depth(timestamp, [{'tracks': [1, 2], 'status': 'compatible',
            'contacts': [{'actor': 1, 'wrist': 9, 'target_track': 2, 'target_joint': 10,
                          'contact_kind': 'guard', 'status': 'compatible'}]}], timestamp)
        assert not any(event.event_type == 'fight' for event in result.events)


@pytest.mark.parametrize('depth', ['compatible', 'missing', 'uncertain', 'wrong_joint', 'separated'])
def test_repeated_one_sided_guard_strikes_need_exact_contact_depth_and_reciprocal_motion(depth):
    behavior = BehaviorHeuristic(Rules(threshold=.55, fight_confirmation_seconds=.5), confirmation=True)
    events = []
    for index in range(64):
        timestamp = index/8
        x = (220, 283, 265, 240, 220, 220, 220, 220)[index % 8]
        result = behavior.update(list(guard_pair(x)), timestamp, .002)
        events.extend(result.events)
        if depth != 'missing':
            behavior.observe_depth(timestamp, [{'tracks': [1, 2],
                'status': 'separated' if depth == 'separated' else 'compatible',
                'contacts': [{'actor': 1, 'wrist': 9, 'target_track': 2,
                    'target_joint': 9 if depth == 'wrong_joint' else 10, 'contact_kind': 'guard',
                    'status': 'uncertain' if depth == 'uncertain' else 'compatible'}]}], timestamp)
    assert not any(event.event_type == 'fight' for event in events)
    if depth == 'compatible':
        assert [event.event_type for event in events] == ['possible_fight']
    else:
        assert not events


def test_guard_geometry_without_same_wrist_image_motion_is_not_a_strike():
    _, accepted = run_guard_path([220, 235, 253, 270, 283, 270], image_support=False)
    assert not accepted


def test_single_pose_jump_into_body_cannot_become_an_accepted_slow_strike():
    heuristic = FightHeuristic(Rules(threshold=.55))
    accepted = []
    for index, x in enumerate((215, 285, 285, 285)):
        actor, target = slow_actor(1, 190), slow_actor(2, 300)
        actor.keypoints[9] = (x, 155, .99)
        actor.limb_speeds, actor.limb_motion = {9: .25}, {9: .4}
        target.limb_speeds, target.limb_motion = {}, {}
        result = heuristic.update([actor, target], index/8, .002)
        accepted.extend(strike for strike in result.signals.get('slow_punch_strikes', [])
                        if strike['contact'] == index/8)
    assert not accepted


@pytest.mark.parametrize(('field', 'wrong_value'), [
    ('actor', 2), ('wrist', 10), ('target_track', 1),
    ('contact_kind', 'body'), ('target_joint', 9),
])
def test_guard_contact_depth_cannot_be_borrowed_from_a_different_directed_surface(field, wrong_value):
    # Accepted filtered contact geometry and original-frame depth have different
    # coordinates, but their track/joint identities must match exactly.
    from test_fight_depth_arrival import observe, prepared_gate, strike, tick

    gate = prepared_gate()
    accepted = strike(1, .25)
    different_surface = dict(accepted, **{field: wrong_value})
    observe(gate, .25, [different_surface])
    result = tick(gate, .25, [accepted], current=[accepted])
    assert not result.events
    assert not result.signals['fight_timer_started']
