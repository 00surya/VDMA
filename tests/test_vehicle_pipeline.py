"""Vehicle review context must join its own raw pose frame; no real models/providers."""
from dataclasses import replace
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import vmd.behavior as behavior
import vmd.engine as module
from vmd.alerts import AlertAgent
from vmd.heuristics import Assessment, Person
from vmd.storage import Store


SHAPE = (200, 320)


def raw_people(sequence):
    people = []
    for track, x in ((1, 50), (2, 190)):
        points = [(x, 35, .99)]*17
        points[5:7] = [(x-10, 50, .99), (x+10, 50, .99)]
        points[11:13] = [(x-10, 95, .99), (x+10, 95, .99)]
        points[9] = (x+sequence, 70, .99)
        points[10] = (x, 80, .99)
        points[15:17] = [(x-10, 175, .99), (x+10, 175, .99)]
        people.append(Person(track, (x-25, 25, x+25, 185), points))
    return people


def install_pipeline(monkeypatch, *, recording=False, damage=None, frames=66):
    state = SimpleNamespace(sequence=0, timestamp=0, pose_calls=0, constructors=[],
        submissions=[], file_requests=[], observing=[], accepted=[], updates=[], resets=0,
        stopped=0, instances=[], analyzing=False)

    class Capture:
        file = recording
        finished, error = False, None
        def __init__(self, *args): self.stop_event = threading.Event()
        def start(self): pass
        def stop(self): pass
        def latest(self, after):
            assert not state.analyzing, 'A recording advanced before finishing its selected object frame'
            if state.sequence >= frames:
                self.finished = True
                return None
            state.timestamp = state.sequence/4
            state.sequence += 1
            return state.sequence, state.timestamp, np.full((*SHAPE, 3), state.sequence*3, np.uint8)

    class Pose:
        def __init__(self, *args): pass
        def infer(self, frame):
            state.pose_calls += 1
            return raw_people(state.sequence)

    class Motion:
        def infer(self, *args):
            moving = ((damage == 'sample_camera' and state.timestamp == 12)
                      or (damage == 'current_camera' and state.timestamp >= (12 if recording else 12.25)))
            return 0, .3 if moving else 0
        def person_flow(self, *args): return 0
        def joint_motion(self, *args): return 0

    class Stabilizer:
        def update(self, people, *args):
            output = []
            for p in people:
                points = list(p.keypoints)
                points[9] = (points[9][0]+40, points[9][1], .1)
                output.append(replace(p, keypoints=points, pose_reliable=False))
            return output

    class Objects:
        def __init__(self, *args, **kwargs):
            state.constructors.append(kwargs)
            if damage == 'startup':
                raise RuntimeError('Fixture rider model unavailable')
            self.result, self.last_submit, self.error = None, float('-inf'), None
        def start(self): pass
        def stop(self): state.stopped += 1
        def sample(self, sequence, timestamp):
            image = np.full((*SHAPE, 3), (205, 0, 0), np.uint8)
            value = {'status': 'ready', 'sequence': sequence, 'source_time': timestamp,
                'submitted_at': time.time(), 'frame_shape': list(SHAPE),
                'jpeg': cv2.imencode('.jpg', image)[1].tobytes(), 'latency_ms': 1,
                'context_objects': [{'label': 'motorcycle', 'confidence': .8, 'box': [20, 100, 140, 185]}],
                # Engine must ignore public weapon payloads when weapons are disabled.
                'detections': [{'label': 'knife', 'confidence': .99, 'box': [100, 40, 120, 70]}],
                'knife_status': 'ready', 'knife_error': None}
            if timestamp >= 12:
                if damage == 'sequence': value['sequence'] = -99
                if damage == 'source': value['source_time'] += .001
                if damage == 'shape': value['frame_shape'][0] += 1
                if damage == 'wall_stale': value['submitted_at'] -= 100
                if damage == 'wall_future': value['submitted_at'] += 100
                if damage == 'missing_context': value.pop('context_objects')
                if damage == 'source_stale': value = self.sample(1, 0)
            return value
        def submit(self, sequence, timestamp, frame):
            state.submissions.append((sequence, timestamp))
            if timestamp-self.last_submit < 1-1e-9:
                return False
            self.last_submit = timestamp
            self.result = self.sample(sequence, timestamp)
            return True
        def poll(self):
            if self.error:
                return 'error', self.result, self.error
            if state.timestamp >= 12:
                if damage == 'error': return 'error', self.result, 'Fixture object worker failed'
                if damage == 'missing': return 'ready', None, None
            return ('ready', self.result, None) if self.result else ('loading', None, None)
        def analyze_file_frame(self, sequence, timestamp, frame, stop_event):
            state.analyzing = True
            try:
                state.file_requests.append((sequence, timestamp))
                assert state.sequence == sequence and state.timestamp == timestamp
                if damage == 'timeout' and timestamp >= 12:
                    self.error = 'Fixture recording object timeout'
                    raise RuntimeError(self.error)
                self.result = self.sample(sequence, timestamp)
                return self.result
            finally:
                state.analyzing = False

    class Rider:
        def __init__(self, rules):
            self.rules, self.latest, self.reported = rules, None, False
            state.instances.append(self)
        def reset(self):
            state.resets += 1
            self.latest = None
        def observe_vehicles(self, objects, people, source_time, current_time, sequence, camera_motion, frame_shape):
            state.observing.append((sequence, source_time, current_time, camera_motion, tuple(frame_shape)))
            if not isinstance(objects, list) or camera_motion > self.rules.camera_speed:
                self.reset()
                return
            assert people[0].keypoints[9] == (50+sequence, 70, .99)
            assert not people[0].pose_reliable, 'Reliability belongs to the exact stabilized source pose'
            self.latest = {'sequence': sequence, 'source_time': source_time}
            state.accepted.append((sequence, source_time, people[0].keypoints[9]))
        def update(self, raw, filtered, timestamp, frame_shape=None, camera_motion=0):
            state.updates.append((timestamp, raw[0].keypoints[9], filtered[0].keypoints[9]))
            assert raw[0].keypoints[9] == (50+state.sequence, 70, .99)
            assert filtered[0].keypoints[9] == (90+state.sequence, 70, .1)
            if camera_motion > self.rules.camera_speed:
                self.reset()
            # Only the observed candidate on source frame 12 may produce this
            # fixture event; later fresh context is not a second test incident.
            if self.reported or self.latest is None or self.latest['source_time'] != 12:
                return []
            self.reported = True
            return [Assessment(.65, 'possible_snatching',
                ['Fixture seated rider reached another person; property removal is unverified'],
                {'episode_id': 'fixture-rider-episode', 'vehicle_context': True,
                 'frame_matched': True, 'vehicle_sequence': self.latest['sequence'],
                 'vehicle_source_time': self.latest['source_time']}, (1, 2), True, 'possible_snatching')]
        def snapshot(self):
            return {'phase': 'review' if self.reported else 'monitoring',
                    'candidates': 1 if self.latest else 0, 'fixture': True}

    monkeypatch.setattr(module, 'Capture', Capture)
    monkeypatch.setattr(module, 'PoseModel', Pose)
    monkeypatch.setattr(module, 'Motion', Motion)
    monkeypatch.setattr(module, 'PoseStabilizer', Stabilizer)
    monkeypatch.setattr(module, 'ObjectWorker', Objects)
    monkeypatch.setattr(behavior, 'VehicleSnatchingHeuristic', Rider)
    monkeypatch.setattr(module, 'choose_device', lambda _: 'cpu')
    monkeypatch.setattr(module, 'annotate', lambda frame, people: frame.copy())
    return state


def settings(**changes):
    return SimpleNamespace(**{'mode': 'live', 'source': 'fixture', 'device': 'cpu', 'depth': 'off',
        'threshold': .6, 'hold_seconds': .7, 'target_fps': 1000, 'object_fps': 1,
        'object_detection': False, 'unattended_objects': False, 'snatching_vehicles': True,
        'eco_mode': False, **changes})


@pytest.mark.parametrize('recording', [False, True])
def test_vehicle_review_uses_exact_raw_context_pose_and_saves_existing_prebuffer_without_auto_response(tmp_path, monkeypatch, recording):
    state = install_pipeline(monkeypatch, recording=recording)
    store = Store(tmp_path)
    alerts = AlertAgent(store, background=False)
    store.on_incident = alerts.register
    try:
        engine = module.Engine(store, camera_id='rider-fixture', name='Fixture gate')
        engine.run(settings())
        store.jobs.join()
        assert engine.state['status'] == 'finished', engine.state
        assert state.pose_calls == 66 and len(state.updates) == 66
        assert state.constructors == [{'fps': 1, 'unattended': False, 'weapons': False, 'riders': True}]
        assert len(state.accepted) == 17
        assert len({item[0] for item in state.accepted}) == len(state.accepted), 'Repeated poll results cannot be rebound'
        assert engine.state['signals']['snatching']['visible_people'] == 2
        assert engine.state['signals']['vehicle_snatching']['fixture']
        assert engine.state['scene_objects'] == [] and engine.state['knife_status'] == 'off'
        incident, = store.incidents()
        assert incident['event_type'] == 'possible_snatching' and incident['review'] == 'unreviewed'
        assert incident['signals']['vehicle_context'] and incident['signals']['frame_matched']
        assert incident['signals']['vehicle_sequence'] == 49 and incident['signals']['vehicle_source_time'] == 12
        assert incident['signals']['source_seconds'] == (12 if recording else 12.25)
        assert incident['signals']['buffer_seconds'] == 10
        assert incident['signals']['live_camera'] is (not recording)
        assert 'seated rider' in incident['reasons'][0]
        assert alerts.get(incident['id']) is None and alerts.snapshot()['alerts'] == []
        assert incident['clip'] and incident['clip_error'] is None and store.error is None
        capture = cv2.VideoCapture(str(store.clips/incident['clip']))
        decoded = []
        try:
            while True:
                ok, frame = capture.read()
                if not ok: break
                decoded.append(frame)
        finally:
            capture.release()
        assert len(decoded) == 101
        # Alert banners can alter the top of the event frame. The untouched
        # lower region must retain the original grayscale camera content.
        assert decoded[0][100:].mean() < 40 and decoded[-1][100:].mean() > 140
        assert all(abs(frame[100:, :, 0].mean()-frame[100:, :, 1].mean()) < 5 for frame in decoded), 'Object previews cannot replace the pose evidence prebuffer'
        if recording:
            assert state.file_requests == [(n*4+1, n) for n in range(17)]
            assert not state.submissions
        else:
            assert not state.file_requests
    finally:
        alerts.close()
        store.close()


@pytest.mark.parametrize('damage', ['missing', 'sequence', 'source', 'shape', 'wall_stale',
                                    'wall_future', 'source_stale', 'error', 'missing_context',
                                    'sample_camera', 'current_camera'])
def test_invalid_or_moving_context_cannot_bind_new_rider_review(tmp_path, monkeypatch, damage):
    state = install_pipeline(monkeypatch, damage=damage)
    store = Store(tmp_path)
    try:
        engine = module.Engine(store)
        engine.run(settings())
        store.jobs.join()
        assert engine.state['status'] == 'finished', engine.state
        assert state.pose_calls == 66
        assert state.resets > 0
        assert store.incidents() == [] and store.error is None
        assert not [item for item in state.accepted if item[1] == 12]
    finally:
        store.close()


@pytest.mark.parametrize('recording,damage', [(False, 'startup'), (True, 'timeout')])
def test_optional_rider_worker_failure_keeps_pose_pipeline_running(tmp_path, monkeypatch, recording, damage):
    state = install_pipeline(monkeypatch, recording=recording, damage=damage)
    store = Store(tmp_path)
    try:
        engine = module.Engine(store)
        engine.run(settings())
        store.jobs.join()
        assert engine.state['status'] == 'finished', engine.state
        assert state.pose_calls == 66
        assert engine.state['object_status'] == 'error'
        assert store.incidents() == []
        if damage == 'timeout':
            assert state.stopped >= 1 and state.resets > 0
    finally:
        store.close()


def test_disabled_rider_mode_never_creates_vehicle_heuristic_or_object_worker(tmp_path, monkeypatch):
    state = install_pipeline(monkeypatch)
    store = Store(tmp_path)
    try:
        engine = module.Engine(store)
        engine.run(settings(snatching_vehicles=False))
        store.jobs.join()
        assert engine.state['status'] == 'finished'
        assert state.pose_calls == 66 and state.instances == [] and state.constructors == []
        assert 'vehicle_snatching' not in engine.state['signals']
        assert store.incidents() == []
    finally:
        store.close()
