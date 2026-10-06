import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from vmd.objects import WeaponIncidents
from vmd.engine import Engine
from vmd.storage import Store
from vmd.alerts import AlertAgent


def sample(sequence, timestamp, labels=('knife', 'gun')):
    return {'sequence': sequence, 'source_time': timestamp, 'submitted_at': time.time(),
            'knife_status': 'ready', 'jpeg': cv2.imencode('.jpg', np.full((64,96,3), (200,0,0), np.uint8))[1].tobytes(),
            'latency_ms': 10, 'detections': [
                {'label': label, 'confidence': .95, 'box': [10,10,30,30]} for label in labels]}


def test_weapon_confidence_must_exceed_ninety_percent_before_an_episode_starts():
    gate = WeaponIncidents()
    for sequence, confidence in enumerate((.89, .90, .9001), start=1):
        observed = sample(sequence, sequence)
        for item in observed['detections']:
            item['confidence'] = confidence
        visible, events = gate.update(observed, sequence)
        assert len(visible) == len(events) == (2 if confidence > .90 else 0)


def test_weapon_episodes_deduplicate_samples_and_rearm_after_clear_interval():
    gate = WeaponIncidents()
    first = sample(1, 1)
    assert len(gate.update(first, 1)[1]) == 2
    assert gate.update(first, 2)[1] == []
    for sequence in range(2, 30):
        assert gate.update(sample(sequence, sequence), sequence)[1] == []
    assert gate.update(sample(30, 30, ()), 30) == ([], [])
    assert len(gate.update(sample(35, 35), 35)[1]) == 2
    assert gate.update(sample(36, 36), 50) == ([], [])  # Late result.
    assert gate.update(sample(36, 36), 35) == ([], [])  # Future/mismatched result.
    broken = sample(36, 36)
    broken['knife_status'] = 'error'
    assert gate.update(broken, 36) == ([], [])
    expired = sample(36, 36)
    expired['submitted_at'] -= 20
    assert gate.update(expired, 36) == ([], [])
    assert gate.update(sample(36, 36), 36, status='error') == ([], [])


@pytest.mark.parametrize('sample_sequence', [1, 8])
def test_engine_saves_reviewable_weapon_clips_using_the_original_object_frame(tmp_path, monkeypatch, sample_sequence):
    import threading
    import vmd.engine as module
    step = [0]
    class Capture:
        finished, error = False, None
        file = False
        def __init__(self, *args): self.stop_event = threading.Event()
        def start(self): pass
        def stop(self): pass
        def latest(self, after):
            step[0] += 1
            if step[0] > 30:
                self.finished = True
                return None
            return step[0], step[0] / 10, np.full((64,96,3), 10, np.uint8)
    class Pose:
        def __init__(self, *args): pass
        def infer(self, frame): return []
    original = sample(sample_sequence, sample_sequence / 10)
    class Objects:
        def __init__(self, *args, **kwargs): pass
        def start(self): pass
        def stop(self): pass
        def submit(self, *args): return False
        def poll(self): return ('ready', original, None) if step[0] >= 10 else ('loading', None, None)
    monkeypatch.setattr(module, 'Capture', Capture)
    monkeypatch.setattr(module, 'PoseModel', Pose)
    monkeypatch.setattr(module, 'ObjectWorker', Objects)
    monkeypatch.setattr(module, 'choose_device', lambda _: 'cpu')
    store = Store(tmp_path)
    try:
        engine = Engine(store, camera_id='camera', name='Phone')
        engine.run(SimpleNamespace(mode='live', source='fixture', device='cpu', depth='off',
            threshold=.6, hold_seconds=.7, target_fps=1000, object_detection=True, object_fps=1))
        store.jobs.join()
        incidents = store.incidents()
        assert {i['event_type'] for i in incidents} == {'knife_detected', 'gun_detected'}
        assert len(incidents) == 2
        for event in incidents:
            assert event['camera_name'] == 'Phone' and event['signals']['sequence'] == sample_sequence
            assert event['signals']['live_camera'] is True
            assert event['clip'] and event['review'] == 'unreviewed'
            cap = cv2.VideoCapture(str(store.clips/event['clip']))
            frames = []
            while True:
                ok, image = cap.read()
                if not ok: break
                frames.append(image)
            cap.release()
            assert frames and any(frame[:,:,0].mean() > 150 for frame in frames)
            assert frames[-1][:,:,0].mean() < 30  # Later pose frames keep their own pixels.
            assert store.review(event['id'], 'confirmed')
            assert store.incident(event['id'])['review'] == 'confirmed'
    finally: store.close()


@pytest.mark.parametrize('action', ['dispatch', 'false_positive'])
@pytest.mark.parametrize('label', ['knife', 'gun'])
def test_handled_camera_keeps_running_and_saves_a_new_weapon_episode(tmp_path, monkeypatch, action, label):
    import threading
    import vmd.engine as module
    step = [0]
    handled = []
    monkeypatch.setattr(module.time, 'time', lambda: 1000 + step[0])

    class Capture:
        finished, error, file = False, None, False
        def __init__(self, *args): self.stop_event = threading.Event()
        def start(self): pass
        def stop(self): pass
        def latest(self, after):
            step[0] += 1
            if step[0] == 5:
                store.jobs.join()
                event, = store.incidents()
                if action == 'dispatch':
                    alerts.act(event['id'], 'dispatch')
                else:
                    assert alerts.review(event['id'], 'false_positive')
                handled.append(store.incident(event['id']))
            if step[0] > 38:
                self.finished = True
                return None
            return step[0], float(step[0]), np.full((64,96,3), 10, np.uint8)

    class Pose:
        def __init__(self, *args): pass
        def infer(self, frame): return []

    class Objects:
        def __init__(self, *args, **kwargs): pass
        def start(self): pass
        def stop(self): pass
        def submit(self, *args): return False
        def poll(self):
            labels = (label,) if 2 <= step[0] <= 4 or step[0] >= 36 else ()
            return 'ready', sample(step[0], float(step[0]), labels), None

    monkeypatch.setattr(module, 'Capture', Capture)
    monkeypatch.setattr(module, 'PoseModel', Pose)
    monkeypatch.setattr(module, 'ObjectWorker', Objects)
    monkeypatch.setattr(module, 'choose_device', lambda _: 'cpu')
    store = Store(tmp_path)
    alerts = AlertAgent(store, background=False)
    store.on_incident = alerts.register
    try:
        engine = Engine(store, camera_id='camera', name='Phone')
        engine.run(SimpleNamespace(mode='live', source='fixture', device='cpu', depth='off',
            threshold=.6, hold_seconds=.7, target_fps=1000, object_detection=True, object_fps=1))
        store.jobs.join()
        assert engine.state['status'] == 'finished', engine.state['message']
        assert engine.state['processed_frames'] == 38 and not engine.stop_event.is_set()
        newer, original = store.incidents()
        assert original == handled[0]
        assert newer['id'] != original['id'] and newer['clip'] != original['clip']
        assert newer['created'] - original['handled_at'] == 31
        assert newer['handled_at'] is None and newer['review'] == 'unreviewed'
        assert original['review'] == ('false_positive' if action == 'false_positive' else 'unreviewed')
        for event in (original, newer):
            assert event['camera_id'] == 'camera' and event['event_type'] == f'{label}_detected'
            assert event['clip'] and event['clip_error'] is None
            cap = cv2.VideoCapture(str(store.clips/event['clip']))
            try:
                assert cap.read()[0]
            finally:
                cap.release()
        # The second saved incident is independently actionable, even after a
        # false-positive review or dispatch of the first one from this camera.
        second = alerts.act(newer['id'], 'dispatch')
        assert second['state'] == 'requested' and second['id'] == newer['id']
        assert store.incident(original['id']) == original
    finally:
        alerts.close()
        store.close()
