"""Temporal scene checks use deterministic poses, not model-accuracy claims."""
from dataclasses import replace

import pytest

from vmd.heuristics import Person, Rules
from vmd.snatching import SnatchingHeuristic


def person(track, x, wrist=None, *, moving=False):
    points = [(x, 75, .99)] * 17
    points[5:7] = [(x-25, 100, .99), (x+25, 100, .99)]
    points[11:13] = [(x-20, 180, .99), (x+20, 180, .99)]
    points[9:11] = [wrist or (x+30, 160, .99), (x-30, 165, .99)]
    points[15:17] = [(x-20, 345, .99), (x+20, 345, .99)]
    return Person(track, (x-50, 60, x+50, 360), points, body_scale=250,
                  limb_speeds={9: 1.4, 15: 1.2 if moving else 0},
                  limb_motion={9: .8, 15: .4 if moving else 0}, local_flow=.1)


def scene(t, action='run'):
    run = max(0, t-.5)*300 if action in {'run', 'no_legs'} else max(0, t-.5)*60 if action == 'walk' else 0
    victim_run = max(0, t-.5)*300 if action == 'victim_runs' else 0
    x = 400-run
    wrist = (430, 102, .99) if t == 0 else (525, 102, .99) if t <= .125 else (475-run, 102, .99)
    if action == 'handshake':
        wrist = (*wrist[:1], 170, .99)
    if action == 'own_neck':
        wrist = (x, 100, .99)
    if action == 'hug':
        wrist = (525, 102, .99)
    actor = person(1, x, wrist, moving=t>.5 and action=='run')
    victim = person(2, 540+victim_run)
    if action == 'unsupported':
        actor.limb_motion = {}
    people = [actor, victim]
    if action == 'other_runs':
        people.append(person(3, 850+max(0,t-.5)*300, moving=True))
    return people


def detect(action='run', depth='compatible', break_kind=None):
    detector = SnatchingHeuristic(Rules())
    events = []
    for n in range(36):
        t = n/8
        people = scene(t, action)
        # Asynchronous result from the exact approach frame arrives later.
        if n == 3 and depth:
            source = .125 if depth != 'old' else 0-1
            detector.observe_depth(source, [{'tracks': [1,2], 'status': depth}], t)
        if break_kind == 'track' and n >= 4:
            people[0] = replace(people[0], track_id=4)
        if break_kind == 'lost' and n == 4:
            people = people[1:]
        if break_kind == 'gap' and n >= 4:
            t += 1
        choices = detector.update(people, t, camera_motion=.2 if break_kind=='camera' and n==4 else 0)
        events.extend(choice for choice in choices if choice.trigger)
    return events


def test_pull_standing_is_possible_then_same_person_escape_upgrades_same_episode():
    standing = detect('stand')
    assert [e.event_type for e in standing] == ['possible_snatching']
    running = detect()
    assert [e.event_type for e in running] == ['possible_snatching', 'snatching_detected']
    assert running[0].signals['episode_id'] == running[1].signals['episode_id']
    assert running[1].signals['depth_source_time'] == .125
    assert running[1].signals['actor_track'] == 1


@pytest.mark.parametrize('depth', [None, 'uncertain', 'separated', 'old'])
def test_no_matching_grab_depth_never_confirms_even_with_escape(depth):
    assert [e.event_type for e in detect(depth=depth)] == ['possible_snatching']


@pytest.mark.parametrize('action', ['walk', 'no_legs', 'victim_runs', 'other_runs'])
def test_only_same_actor_with_supported_rapid_departure_can_escalate(action):
    assert [e.event_type for e in detect(action)] == ['possible_snatching']


@pytest.mark.parametrize('action', ['handshake', 'own_neck', 'hug', 'unsupported'])
def test_normal_gestures_or_unsupported_pose_motion_do_not_create_snatching(action):
    assert detect(action) == []


@pytest.mark.parametrize('break_kind', ['track', 'lost', 'gap', 'camera'])
def test_discontinuities_cannot_complete_a_previous_grab(break_kind):
    assert [e.event_type for e in detect(break_kind=break_kind)] == ['possible_snatching']


def test_snatching_sequence_survives_real_pose_stabilization():
    import math
    from vmd.stabilization import PoseStabilizer
    detector, stabilizer = SnatchingHeuristic(Rules()), PoseStabilizer()
    events = []
    for n in range(30):
        t = n/8
        relative = max(0, t-.75)
        people = scene(relative)
        if relative > .5:
            # Actual relative ankle swing, rather than just body translation.
            for joint, sign in ((15,1),(16,-1)):
                x,y,c = people[0].keypoints[joint]
                people[0].keypoints[joint] = (x+sign*35*math.sin(t*9),y,c)
        filtered = stabilizer.update(people, t, lambda *args: .8)
        if n == 9:
            detector.observe_depth(.875, [{'tracks':[1,2],'status':'compatible'}],t)
        events.extend(e for e in detector.update(filtered,t) if e.trigger)
    assert [e.event_type for e in events] == ['possible_snatching','snatching_detected']
