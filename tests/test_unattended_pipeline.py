"""Sample provenance, recording backpressure and evidence; no model/providers."""
import queue
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import vmd.engine as module
from vmd.alerts import AlertAgent
from vmd.engine import Engine
from vmd.storage import Store
from test_unattended import bag, person


SHAPE = (400, 640)


def object_sample(sequence, timestamp, *, weapons=False):
    image = np.full((*SHAPE, 3), (205, 0, 0), np.uint8)
    return {'status': 'ready', 'sequence': sequence, 'source_time': timestamp,
            'submitted_at': time.time(), 'frame_shape': list(SHAPE),
            'jpeg': cv2.imencode('.jpg', image)[1].tobytes(), 'latency_ms': 1,
            'context_objects': [bag(), person()] if timestamp <= 2 else [bag()],
            'knife_status': 'ready' if weapons else 'off', 'knife_error': None,
            'detections': [{'label': 'knife', 'confidence': .99, 'box': [20, 30, 50, 60]}] if weapons else []}


@pytest.mark.parametrize('invalid', ['missing', 'sequence', 'source', 'shape', 'uncached',
                                     'source_stale', 'source_future', 'wall_stale', 'wall_future', 'status'])
def test_context_joins_only_exact_fresh_object_sample(invalid):
    original = object_sample(12, 3)
    cached = {12: {'time': 3, 'shape': SHAPE, 'camera_motion': .04}}
    assert Engine.matched_object_context(original, cached, 3.5, 1) is cached[12]
    sample, current, status = dict(original), 3.5, 'ready'
    if invalid == 'missing': sample = None
    if invalid == 'sequence': sample['sequence'] = 11
    if invalid == 'source': sample['source_time'] += .001
    if invalid == 'shape': sample['frame_shape'] = [SHAPE[0], SHAPE[1]+1]
    if invalid == 'uncached': cached = {}
    if invalid == 'source_stale': current = 6.001
    if invalid == 'source_future': current = 2.99
    if invalid == 'wall_stale': sample['submitted_at'] -= 3.01
    if invalid == 'wall_future': sample['submitted_at'] += 1
    if invalid == 'status': status = 'error'
    assert Engine.matched_object_context(sample, cached, current, 1, status) is None


def install_pipeline(monkeypatch, *, recording=False, damage=None, weapons_in_payload=False,
                     frames=66, fps=4):
    """Live samples arrive one frame later; recordings block on the chosen frame."""
    state = SimpleNamespace(sequence=0, timestamp=0, pose_calls=0, constructors=[], submissions=[],
                            file_requests=[], polls=0, worker_stopped=0, attendance_updates=[])
    detector_type = module.UnattendedObjects
    class Detector(detector_type):
        def update(self, objects, timestamp, camera_motion=0, frame_shape=None, **kwargs):
            state.attendance_updates.append((timestamp, camera_motion, frame_shape))
            return super().update(objects, timestamp, camera_motion, frame_shape, **kwargs)

    class Capture:
        finished, error, file = False, None, recording
        def __init__(self, *args): self.stop_event = threading.Event()
        def start(self): pass
        def stop(self): pass
        def latest(self, after):
            # A recording must complete its pending object sample before decoding
            # another frame. The fake worker tracks entry/exit below.
            assert not getattr(state, 'analyzing', False)
            if state.sequence >= frames:
                self.finished = True
                return None
            state.timestamp = state.sequence/fps
            state.sequence += 1
            return state.sequence, state.timestamp, np.full((*SHAPE, 3), 10, np.uint8)

    class Pose:
        def __init__(self, *args): pass
        def infer(self, frame):
            state.pose_calls += 1
            return []

    class Motion:
        def infer(self, *args):
            moving = ((damage == 'sample_camera' and state.timestamp == 5)
                      or (damage == 'current_camera' and state.timestamp == 5.25))
            return 0, .3 if moving else 0
        def person_flow(self, *args): return 0
        def joint_motion(self, *args): return 0

    class Objects:
        def __init__(self, *args, **kwargs):
            state.constructors.append(kwargs)
            self.result, self.last_submit, self.error = None, float('-inf'), None
        def start(self): pass
        def stop(self): state.worker_stopped += 1
        def damaged_sample(self, sequence, timestamp):
            result = object_sample(sequence, timestamp, weapons=weapons_in_payload)
            if 5 <= timestamp < 6:
                if damage == 'sequence': result['sequence'] = -1
                if damage == 'source': result['source_time'] += .001
                if damage == 'shape': result['frame_shape'][0] += 1
                if damage == 'wall_stale': result['submitted_at'] -= 100
                if damage == 'missing_context': result.pop('context_objects')
            return result
        def submit(self, sequence, timestamp, frame):
            state.submissions.append((sequence, timestamp))
            if timestamp-self.last_submit < 1-1e-9:
                return False
            self.last_submit = timestamp
            self.result = self.damaged_sample(sequence, timestamp)
            return True
        def poll(self):
            state.polls += 1
            if self.error:
                return 'error', self.result, self.error
            if damage == 'error' and 5 <= state.timestamp < 6:
                return 'error', self.result, 'fixture object worker failed'
            if damage == 'missing' and 5 <= state.timestamp < 6:
                return 'ready', None, None
            if damage == 'sample_camera' and 5 <= state.timestamp < 6:
                # Deliver the original moving-camera result only after movement
                # has stopped, so the cached-motion gate is exercised separately.
                return 'ready', None, None
            return ('ready', self.result, None) if self.result else ('loading', None, None)
        def analyze_file_frame(self, sequence, timestamp, frame, stop_event):
            state.analyzing = True
            try:
                state.file_requests.append((sequence, timestamp))
                assert timestamp == state.timestamp and sequence == state.sequence
                if damage == 'timeout' and timestamp >= 5:
                    self.error = 'fixture recording object analysis timed out'
                    raise RuntimeError(self.error)
                self.result = self.damaged_sample(sequence, timestamp)
                return self.result
            finally:
                state.analyzing = False

    monkeypatch.setattr(module, 'Capture', Capture)
    monkeypatch.setattr(module, 'PoseModel', Pose)
    monkeypatch.setattr(module, 'ObjectWorker', Objects)
    monkeypatch.setattr(module, 'Motion', Motion)
    monkeypatch.setattr(module, 'UnattendedObjects', Detector)
    monkeypatch.setattr(module, 'choose_device', lambda _: 'cpu')
    monkeypatch.setattr(module, 'annotate', lambda frame, people: frame.copy())
    return state


def settings(**changes):
    return SimpleNamespace(**{'mode': 'live', 'source': 'fixture', 'device': 'cpu', 'depth': 'off',
        'threshold': .6, 'hold_seconds': .7, 'target_fps': 1000, 'object_detection': False,
        'object_fps': 1, 'unattended_objects': True, 'unattended_seconds': 10, 'eco_mode': False,
        **changes})


@pytest.mark.parametrize('recording', [False, True])
def test_independent_bag_monitoring_saves_sample_owned_evidence_without_weapons_or_auto_calls(tmp_path, monkeypatch, recording):
    state = install_pipeline(monkeypatch, recording=recording, weapons_in_payload=True)
    store = Store(tmp_path)
    alerts = AlertAgent(store, background=False)
    store.on_incident = alerts.register
    try:
        engine = Engine(store, camera_id='bags', name='Entrance')
        engine.run(settings())
        store.jobs.join()
        assert engine.state['status'] == 'finished', engine.state
        assert state.pose_calls == 66
        assert state.constructors == [{'fps': 1, 'unattended': True, 'weapons': False}]
        incident, = store.incidents()
        assert incident['event_type'] == 'unattended_object'
        assert incident['camera_id'] == 'bags' and incident['camera_name'] == 'Entrance'
        assert incident['review'] == 'unreviewed' and incident['clip_error'] is None
        assert incident['signals']['source_seconds'] == 13
        assert incident['signals']['sequence'] == 53
        assert incident['signals']['live_camera'] is (not recording)
        assert incident['signals']['frame_matched'] and not incident['signals']['contents_verified']
        assert alerts.get(incident['id']) is None
        capture = cv2.VideoCapture(str(store.clips/incident['clip']))
        frames = []
        try:
            while True:
                ok, image = capture.read()
                if not ok: break
                frames.append(image)
        finally:
            capture.release()
        assert frames and any(image[:, :, 0].mean() > 150 for image in frames)
        assert any(image[:, :, 0].mean() < 40 for image in frames)
        if recording:
            assert state.submissions == []
            assert state.file_requests == [(1+4*n, float(n)) for n in range(17)]
            assert engine.state['object_submissions'] == 17
    finally:
        alerts.close()
        store.close()


@pytest.mark.parametrize('legacy_settings', [True, False])
def test_disabled_monitoring_preserves_no_worker_no_incident_behavior(tmp_path, monkeypatch, legacy_settings):
    state = install_pipeline(monkeypatch)
    configured = settings(unattended_objects=False)
    if legacy_settings:
        del configured.unattended_objects
        del configured.unattended_seconds
    store = Store(tmp_path)
    try:
        engine = Engine(store)
        engine.run(configured)
        store.jobs.join()
        assert engine.state['status'] == 'finished' and state.pose_calls == 66
        assert state.constructors == []
        assert store.incidents() == []
    finally:
        store.close()


@pytest.mark.parametrize('damage', ['sequence', 'source', 'shape', 'wall_stale', 'missing_context',
                                    'error', 'missing', 'sample_camera', 'current_camera'])
def test_lost_provenance_or_camera_motion_cannot_finish_previously_armed_absence(tmp_path, monkeypatch, damage):
    state = install_pipeline(monkeypatch, damage=damage)
    store = Store(tmp_path)
    try:
        engine = Engine(store)
        engine.run(settings())
        store.jobs.join()
        assert engine.state['status'] == 'finished', engine.state
        assert state.pose_calls == 66
        assert store.incidents() == []
        assert not engine.state['unattended_meta']['tracks'][0]['prior_attendant_seen']
        if damage == 'sample_camera':
            assert (5, .3, SHAPE) in state.attendance_updates
        if damage == 'current_camera':
            assert not any(camera for _, camera, _ in state.attendance_updates)
    finally:
        store.close()


def test_recording_worker_timeout_interrupts_attendance_but_keeps_pose_processing(tmp_path, monkeypatch):
    state = install_pipeline(monkeypatch, recording=True, damage='timeout')
    store = Store(tmp_path)
    try:
        engine = Engine(store)
        engine.run(settings())
        store.jobs.join()
        assert engine.state['status'] == 'finished', engine.state
        assert engine.state['processed_frames'] == state.pose_calls == 66
        assert store.incidents() == []
        assert engine.state['object_status'] == 'error'
        assert engine.state['unattended_meta']['tracks'] == []
        assert state.file_requests == [(1+4*n, float(n)) for n in range(6)]
        assert state.worker_stopped >= 1
    finally:
        store.close()


def test_bag_worker_startup_failure_is_optional_and_preserves_pose_processing(tmp_path, monkeypatch):
    state = install_pipeline(monkeypatch)
    def missing_model(*args, **kwargs):
        raise RuntimeError('fixture general-object weights missing')
    monkeypatch.setattr(module, 'ObjectWorker', missing_model)
    store = Store(tmp_path)
    try:
        engine = Engine(store)
        engine.run(settings())
        store.jobs.join()
        assert engine.state['status'] == 'finished' and state.pose_calls == 66
        assert engine.state['object_status'] == 'error'
        assert engine.state['unattended_meta']['tracks'] == []
        assert store.incidents() == []
    finally:
        store.close()


def test_bags_only_worker_uses_general_same_frame_context_without_loading_weapon_model(monkeypatch):
    import torch
    import vmd.objects as objects

    stop, calls = threading.Event(), []
    frame = np.zeros((*SHAPE, 3), np.uint8)
    class General:
        def __init__(self, model_dir, device, *, class_ids):
            calls.append(('general', class_ids))
        def infer(self, image):
            assert image is frame
            stop.set()
            return [bag(), person()]
    class Weapon:
        def __init__(self, *args):
            calls.append(('weapon',))
            raise AssertionError('Bags-only configuration must not load weapons')
    monkeypatch.setattr(objects, 'ObjectModel', General)
    monkeypatch.setattr(objects, 'WeaponModel', Weapon)
    monkeypatch.setattr(torch, 'set_num_threads', lambda _: None)
    monkeypatch.setattr(cv2, 'setNumThreads', lambda _: None)
    requests, responses = queue.Queue(), queue.Queue()
    requests.put((9, 2.5, time.time(), frame))
    objects.run_objects(requests, responses, stop, 'bags', 'cpu', 'unused-models')
    updates = []
    while not responses.empty():
        updates.append(responses.get_nowait())
    result = updates[-1]
    assert calls == [('general', [0, 24, 26, 28])]
    assert result['sequence'] == 9 and result['source_time'] == 2.5
    assert result['frame_shape'] == list(SHAPE)
    assert result['context_objects'] == [bag(), person()]
    assert result['detections'] == [] and result['knife_status'] == 'off'
    assert result['knife_error'] is None and result['jpeg']


def test_recording_object_worker_keeps_its_frame_until_exact_result_through_warmup(monkeypatch):
    from vmd.objects import ObjectWorker

    worker = ObjectWorker.__new__(ObjectWorker)
    worker.closed, worker.status, worker.error = False, 'loading', None
    worker.requests = queue.Queue(maxsize=1)
    frame = np.full((*SHAPE, 3), 19, np.uint8)
    stop, polls = threading.Event(), []
    exact = object_sample(21, 5)
    def poll():
        polls.append(True)
        if len(polls) == 1:
            return 'loading', None, None
        if len(polls) == 2:
            frame[:] = 99
            # A different original frame cannot satisfy this blocking request.
            return 'ready', object_sample(20, 4.75), None
        return 'ready', exact, None
    monkeypatch.setattr(worker, 'poll', poll)
    result = worker.analyze_file_frame(21, 5, frame, stop, timeout=1)
    assert result is exact and len(polls) == 3
    sequence, timestamp, submitted, queued_frame = worker.requests.get_nowait()
    assert (sequence, timestamp) == (21, 5) and submitted > 0
    assert np.all(queued_frame == 19) and np.all(frame == 99)
    assert worker.requests.empty()
