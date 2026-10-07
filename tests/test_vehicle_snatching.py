"""Rider/drag review fixtures; these are not theft-model accuracy tests."""
from dataclasses import replace

import pytest

from vmd.vehicle_snatching import VehicleSnatchingHeuristic
from test_snatching import person


SHAPE = (640, 960)


def translate(p, dx=0, dy=0):
    return replace(p, box=(p.box[0]+dx, p.box[1]+dy, p.box[2]+dx, p.box[3]+dy),
                   keypoints=[(x+dx, y+dy, c) for x, y, c in p.keypoints])


def scene(timestamp, action='drag'):
    elapsed = max(0, timestamp-1)
    actor_dx = -elapsed*100 if action != 'victim_only' else 0
    target_dx = -elapsed*130 if action not in {'no_reaction', 'wave', 'walking'} else -elapsed*25
    wrist = (430, 102, .99) if timestamp < .5 else (480, 102, .99) if timestamp == .5 else (425, 102, .99)
    if action == 'handshake': wrist = (wrist[0], 170, .99)
    if action == 'hug' and timestamp >= .5: wrist = (480, 102, .99)
    if action == 'own_neck': wrist = (400, 100, .99)
    if action == 'high_five': wrist = (wrist[0], 80, .99)
    actor, target = translate(person(1, 400, wrist), dx=actor_dx), translate(person(2, 540), dx=target_dx)
    actor.local_flow = .2 if elapsed else .05
    target.local_flow = .2 if elapsed and action not in {'walking', 'no_reaction', 'wave'} else .03
    if action == 'high_five':
        target.keypoints[10] = (480+target_dx, 80, .99)
    if action == 'missing_pull' and timestamp >= .625:
        actor.keypoints[9] = (*actor.keypoints[9][:2], .1)
    if action == 'unsupported_pull' and timestamp >= .625:
        actor.limb_motion = {9: 0}
    if action == 'unsupported_rider' and elapsed:
        actor.local_flow = .02
    if action == 'unsupported_reaction' and elapsed:
        target.local_flow = .02
    if action == 'no_reaction':
        target = person(2, 540)
        target.local_flow = .02
    if action == 'fall' and timestamp >= 1.25:
        points = list(target.keypoints)
        points[5:7] = [(525, 200, .99), (555, 200, .99)]
        points[11:13] = [(655, 200, .99), (665, 200, .99)]
        target = replace(target, box=(500, 150, 740, 250), keypoints=points, local_flow=.2)
    return [actor, target]


def vehicles(people, timestamp, action='drag'):
    actor, target = people
    dx = -max(0, timestamp-1)*100 if action not in {'victim_only', 'stationary_bike'} else 0
    box = [320+dx, 220, 480+dx, 390]
    result = [{'label': 'motorcycle', 'confidence': .8, 'box': box}]
    if action == 'duplicate_bike':
        result.append({'label': 'motorcycle', 'confidence': .6, 'box': [box[0]+5, 230, box[2]-5, 385]})
        result.append({'label': 'motorcycle', 'confidence': .33, 'box': box})
    if action == 'ambiguous_bikes':
        result = [{'label': 'motorcycle', 'confidence': .8, 'box': [310+dx, 220, 445+dx, 390]},
                  {'label': 'motorcycle', 'confidence': .7, 'box': [355+dx, 220, 490+dx, 390]}]
    if action == 'different_bike' and timestamp >= .75:
        result = [{'label': 'motorcycle', 'confidence': .95,
                   'box': [target.box[0]-30, 220, target.box[2]+30, 390]}]
    result.extend({'label': 'person', 'confidence': .9, 'box': list(p.box)} for p in people)
    return result


def run(action='drag', *, object_rate=8, pose_rate=8, break_kind=None):
    detector, events, snapshots = VehicleSnatchingHeuristic(), [], []
    last_sample = float('-inf')
    for n in range(int(4*pose_rate)):
        timestamp = n/pose_rate
        people = scene(timestamp, action)
        if break_kind == 'track' and n >= 7:
            people[0] = replace(people[0], track_id=4)
        if break_kind == 'target_loss' and n >= 7:
            people = people[:1]
        if break_kind == 'duplicate_rider':
            people.append(person(3, 430))
        camera = .3 if break_kind == 'camera' and n == 7 else 0
        shape = (800, 960) if break_kind == 'shape' and n >= 7 else SHAPE
        current = timestamp+1 if break_kind == 'gap' and n >= 7 else timestamp
        if current-last_sample >= 1/object_rate-1e-9:
            items = vehicles(people[:2], timestamp, action) if len(people) > 1 else []
            if break_kind == 'duplicate_rider':
                items.append({'label': 'person', 'confidence': .9, 'box': list(people[-1].box)})
            detector.observe_vehicles(items, people, current, current, n+1, camera, shape)
            last_sample = current
        choices = detector.update(people, people, current, shape, camera)
        events.extend(choice for choice in choices if choice.trigger)
        snapshots.append(detector.snapshot())
    return detector, events, snapshots


def test_vehicle_drag_is_one_uncertain_manual_review_event_without_leg_motion():
    _, events, _ = run()
    event, = events
    assert event.event_type == event.state == 'possible_snatching'
    assert event.signals['vehicle_context'] and event.signals['review_only']
    assert event.signals['contact_uncertain'] and not event.signals['property_removal_verified']
    assert event.signals['raw_neck_gap_body_heights'] > 0
    assert event.signals['rider_movement_body_heights'] >= .2
    assert event.signals['vehicle_movement_body_heights'] >= .2
    assert event.signals['target_reaction']['net_body_heights'] >= .25
    assert event.signals['target_reaction']['toward_body_heights'] >= .2
    assert any('does not establish theft' in reason for reason in event.reasons)


@pytest.mark.parametrize('rate', [1, 2, 8])
def test_vehicle_samples_have_their_own_source_clock_and_must_be_seen_after_pull(rate):
    _, events, _ = run(object_rate=rate)
    event, = events
    assert event.signals['vehicle_source_time'] > event.signals['pull_seconds']
    assert event.signals['vehicle_sequence'] >= 1


def test_two_fps_retains_four_supported_observations_within_bounded_history():
    detector, events, _ = run(object_rate=1, pose_rate=2)
    event, = events
    assert event.signals['rider_speed_body_heights_s'] >= .2
    assert event.signals['target_reaction']['speed_body_heights_s'] >= .4
    for state in detector.pending.values():
        for name in ('rider_path', 'target_path'):
            if name in state:
                assert state[name][-1][0]-state[name][0][0] <= 2


@pytest.mark.parametrize('support', [.8, 0])
def test_actual_pose_stabilization_preserves_supported_review_and_rejects_unmeasured_arm_motion(support):
    from vmd.stabilization import PoseStabilizer
    detector, stabilizer, events = VehicleSnatchingHeuristic(), PoseStabilizer(), []
    for n in range(32):
        timestamp = n/8
        raw = scene(timestamp)
        stable = stabilizer.update(raw, timestamp, lambda *args: support)
        if n % 8 == 0:
            detector.observe_vehicles(vehicles(raw, timestamp), raw, timestamp, timestamp, n+1, 0, SHAPE)
        events.extend(e for e in detector.update(raw, stable, timestamp, SHAPE) if e.trigger)
    assert [e.event_type for e in events] == (['possible_snatching'] if support else [])


def test_supported_filtered_approach_may_finish_after_raw_hand_already_reaches_neck_band():
    detector = VehicleSnatchingHeuristic()
    for n in range(3):
        timestamp = n/8
        raw = scene(.5)
        stable = scene(0) if n < 2 else scene(.5)
        detector.observe_vehicles(vehicles(raw, timestamp), raw, timestamp, timestamp, n+1, 0, SHAPE)
        detector.update(raw, stable, timestamp, SHAPE)
    candidate, = detector.snapshot()['candidates']
    assert candidate['phase'] == 'checking_pull'
    assert candidate['grab_seconds'] == .25


def test_defensive_hand_near_collar_is_not_mistaken_for_both_hands_raised_high_five():
    detector, events = VehicleSnatchingHeuristic(), []
    for n in range(32):
        timestamp = n/8
        raw = scene(timestamp)
        if timestamp <= .5:
            raw[1].keypoints[9] = (480, 85, .99)
        detector.observe_vehicles(vehicles(raw, timestamp), raw, timestamp, timestamp, n+1, 0, SHAPE)
        events.extend(e for e in detector.update(raw, raw, timestamp, SHAPE) if e.trigger)
    assert [e.event_type for e in events] == ['possible_snatching']


def test_duplicate_same_motorcycle_predictions_do_not_make_rider_ambiguous():
    detector, events, _ = run('duplicate_bike')
    assert len(events) == 1
    assert detector.snapshot()['vehicles'] == 1


@pytest.mark.parametrize('action', ['no_reaction', 'wave', 'walking', 'handshake', 'hug', 'own_neck', 'high_five',
                                    'missing_pull', 'unsupported_pull', 'unsupported_rider',
                                    'unsupported_reaction', 'victim_only', 'stationary_bike',
                                    'ambiguous_bikes', 'different_bike'])
def test_vehicle_context_alone_cannot_turn_gestures_or_unsupported_movement_into_an_alert(action):
    _, events, snapshots = run(action)
    assert events == []
    if action == 'missing_pull':
        assert any('clear pulling wrist' in ' '.join(s['blockers']+[
            blocker for candidate in s['candidates'] for blocker in candidate['blockers']]) for s in snapshots)


@pytest.mark.parametrize('kind', ['track', 'target_loss', 'duplicate_rider', 'camera', 'shape', 'gap'])
def test_discontinuity_or_ambiguous_association_cannot_finish_an_earlier_pull(kind):
    assert run(break_kind=kind)[1] == []


def test_target_may_fall_after_observed_pull_without_remaining_standing():
    _, events, _ = run('fall')
    event, = events
    assert event.signals['target_reaction']['falling_posture']
    assert event.event_type == 'possible_snatching'


def test_one_motorcycle_observation_cannot_be_reused_to_prove_its_departure():
    detector, events = VehicleSnatchingHeuristic(), []
    for n in range(24):
        timestamp = n/8
        people = scene(timestamp)
        if n == 3:
            detector.observe_vehicles(vehicles(people, timestamp), people, timestamp, timestamp, n+1, 0, SHAPE)
        events.extend(e for e in detector.update(people, people, timestamp, SHAPE) if e.trigger)
    assert events == []


def test_snapshot_is_read_only_and_histories_remain_bounded():
    detector, _, _ = run()
    first = detector.snapshot()
    assert detector.snapshot() == first
    first['blockers'].append('changed')
    assert 'changed' not in detector.snapshot()['blockers']
    assert len(detector.vehicles) <= detector.max_vehicles
    assert len(detector.pending) <= detector.max_pairs
    assert all(len(value['samples']) <= 16 for value in detector.vehicles.values())


@pytest.mark.parametrize('damage', ['jump', 'scale', 'unreliable', 'duplicate', 'raw_only'])
def test_visible_target_discontinuity_cannot_finish_previously_observed_reaction(damage):
    detector, events = VehicleSnatchingHeuristic(), []
    for n in range(25):
        timestamp = n/8
        raw = scene(timestamp)
        items = vehicles(raw, timestamp)
        if timestamp >= 2:
            if damage == 'jump': raw[1] = translate(raw[1], dx=300)
            if damage == 'scale': raw[1] = replace(raw[1], body_scale=60)
            if damage == 'unreliable': raw[1] = replace(raw[1], pose_reliable=False)
            if damage == 'duplicate': raw.append(raw[1])
        stable = raw[:1] if damage == 'raw_only' and timestamp >= 2 else raw
        if n % 8 == 0:
            detector.observe_vehicles(items, raw, timestamp, timestamp, n+1, 0, SHAPE)
        events.extend(e for e in detector.update(raw, stable, timestamp, SHAPE) if e.trigger)
        if timestamp == 1.875:
            state, = detector.pending.values()
            assert state['reaction'] is not None and not state.get('reported')
    assert events == []


def test_genuinely_missing_target_has_only_short_reference_grace_after_supported_reaction():
    detector, events = VehicleSnatchingHeuristic(), []
    for n in range(27):
        timestamp = n/8
        raw = scene(timestamp)
        items = vehicles(raw, timestamp)
        if timestamp >= 2: raw = raw[:1]
        if n % 8 == 0:
            detector.observe_vehicles(items, raw, timestamp, timestamp, n+1, 0, SHAPE)
        events.extend(e for e in detector.update(raw, raw, timestamp, SHAPE) if e.trigger)
    event, = events
    assert 0 < event.signals['target_reference_age_seconds'] <= detector.target_grace_seconds
    assert detector.pending == {}


@pytest.mark.parametrize('box', [(450, 60, 350, 360), (350, 360, 450, 60),
                                 (350, 60, 350, 360), (-1, 60, 450, 360),
                                 (350, 60, 450, 900), (350, 60, float('nan'), 360)])
def test_malformed_pose_boxes_cannot_break_object_association_or_arm_a_rider(box):
    detector = VehicleSnatchingHeuristic()
    raw = scene(0)
    items = vehicles(raw, 0)
    raw[0] = replace(raw[0], box=box)
    detector.observe_vehicles(items, raw, 0, 0, 1, 0, SHAPE)
    assert detector.bindings == {}
    assert detector.update(raw, raw, 0, SHAPE) == []


@pytest.mark.parametrize('damage', ['future', 'stale', 'camera', 'missing_shape', 'bad_box'])
def test_invalid_object_context_cannot_arm_a_rider(damage):
    detector = VehicleSnatchingHeuristic()
    people = scene(0)
    items, source, current, camera, shape = vehicles(people, 0), 0, 0, 0, SHAPE
    if damage == 'future': source = 1
    if damage == 'stale': current = 3
    if damage == 'camera': camera = .3
    if damage == 'missing_shape': shape = None
    if damage == 'bad_box': items[0]['box'][0] = -5
    detector.observe_vehicles(items, people, source, current, 1, camera, shape)
    assert detector.bindings == {}


def test_snapshot_explains_missing_rider_context_without_claiming_a_missing_reach():
    detector = VehicleSnatchingHeuristic()
    raw = scene(0)
    detector.update(raw, raw, 0, SHAPE)
    assert detector.snapshot()['phase'] == 'waiting_rider_context'
    assert detector.snapshot()['riders'] == 0
    assert 'unambiguous rider' in detector.snapshot()['blockers'][0]


def test_snapshot_returns_to_waiting_context_when_last_motorcycle_sample_is_stale():
    detector = VehicleSnatchingHeuristic()
    for n in range(24):
        timestamp = n/8
        raw = scene(0)
        if n == 0:
            detector.observe_vehicles(vehicles(raw, 0), raw, 0, 0, 1, 0, SHAPE)
        detector.update(raw, raw, timestamp, SHAPE)
        if timestamp == 1:
            assert detector.snapshot()['phase'] == 'monitoring'
            assert detector.snapshot()['riders'] == 1
    assert detector.snapshot()['phase'] == 'waiting_rider_context'
    assert detector.snapshot()['riders'] == 0
