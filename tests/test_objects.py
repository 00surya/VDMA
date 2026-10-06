from conftest import authenticate
import time
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from vmd.api import create_app
from vmd.engine import Engine
from vmd.objects import ObjectModel, WeaponModel, WEIGHTS, KNIFE_WEIGHTS, annotate_objects, weapon_detections


def test_missing_corrupt_weights_are_rejected_before_model_loading(tmp_path):
    with pytest.raises(RuntimeError, match='weights missing'):
        ObjectModel(tmp_path)
    (tmp_path/WEIGHTS).write_bytes(b'bad')
    with pytest.raises(RuntimeError, match='checksum'):
        ObjectModel(tmp_path)
    with pytest.raises(RuntimeError, match='weights missing'):
        WeaponModel(tmp_path)
    (tmp_path/KNIFE_WEIGHTS).write_bytes(b'bad specialist')
    with pytest.raises(RuntimeError, match='checksum'):
        WeaponModel(tmp_path)


def test_assalim_enables_both_weapon_classes_at_the_same_threshold():
    class Array:
        def __init__(self, value): self.value = value
        def cpu(self): return self
        def tolist(self): return self.value
    model = WeaponModel.__new__(WeaponModel)
    model.device = 'cpu'
    def predict(frame, **options):
        assert options['classes'] == [0, 1] and options['conf'] == .90
        return [SimpleNamespace(names=dict(enumerate(WeaponModel.labels)), boxes=SimpleNamespace(
            xyxy=Array([[10, 20, 30, 40], [50, 60, 70, 80]]),
            conf=Array([.9001, .98]), cls=Array([1, 0])))]
    model.model = SimpleNamespace(predict=predict)
    found = model.infer(np.zeros((100,100,3), np.uint8))
    assert [item['label'] for item in found] == ['knife', 'guns']
    assert found[0]['confidence'] == .9001 and found[0]['context_only']
    assert len(weapon_detections(found)) == 2  # Do not round before applying the threshold.


def test_only_confident_weapon_labels_are_exposed_without_person_count_gate():
    knife = {'label':'knife', 'confidence':.95, 'box':[11,21,31,61], 'context_only':False}
    gun = {**knife, 'label':'guns'}
    noise = [{**knife, 'label': label} for label in ('person', 'toilet', 'scissors')]
    noise += [{**knife, 'label': label, 'confidence': confidence}
              for label in ('knife', 'gun', 'guns')
              for confidence in (.49, .8999, .90, float('nan'), 1.1)]
    assert weapon_detections([knife, gun, *noise]) == [knife, {**gun, 'label':'gun'}]
    assert weapon_detections(noise) == []
    frame = np.zeros((100, 100, 3), np.uint8)
    np.testing.assert_array_equal(annotate_objects(frame, noise), frame)


def test_unexpected_specialist_class_order_is_rejected(tmp_path, monkeypatch):
    import hashlib
    payload = b'fixture only'
    (tmp_path/KNIFE_WEIGHTS).write_bytes(payload)
    monkeypatch.setattr(WeaponModel, 'checksum', hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(WeaponModel, 'load_model', lambda *args: SimpleNamespace(names={0:'knife',1:'guns'}))
    with pytest.raises(RuntimeError, match='class mapping'):
        WeaponModel(tmp_path)


@pytest.mark.parametrize('model_type', [ObjectModel, WeaponModel])
def test_cpu_budget_is_restored_after_yolo_lazy_device_setup(tmp_path, monkeypatch, model_type):
    import hashlib
    import torch
    from ultralytics.utils.torch_utils import select_device
    payload = b'local fixture, no model inference'
    (tmp_path/model_type.weights).write_bytes(payload)
    monkeypatch.setattr(model_type, 'checksum', hashlib.sha256(payload).hexdigest())
    observed = []

    class LazyYOLO:
        names = dict(enumerate(model_type.labels))
        def __init__(self): self.callbacks = []
        def add_callback(self, event, callback):
            assert event == 'on_predict_start'
            self.callbacks.append(callback)
        def predict(self, frame, **options):
            # Exercise the installed library's actual thread reset. Both models
            # initialize separately, and repeated predictions must stay bounded.
            select_device('cpu', verbose=False)
            for callback in self.callbacks:
                callback(self)
            observed.append(torch.get_num_threads())
            return [SimpleNamespace(boxes=None)]

    monkeypatch.setattr(model_type, 'load_model', lambda *args: LazyYOLO())
    original_threads = torch.get_num_threads()
    try:
        model = model_type(tmp_path)
        for _ in range(2):
            assert model.infer(np.zeros((64,96,3), np.uint8)) == []
        assert observed == [1, 1]
    finally:
        torch.set_num_threads(original_threads)


def test_object_preview_blurs_detected_head_and_preserves_input():
    frame = np.random.default_rng(19).integers(0, 256, (200, 200, 3), dtype=np.uint8)
    original = frame.copy()
    people = [{'label':'person','confidence':.9,'box':[30,20,130,180]}]
    preview = annotate_objects(frame, [{'label':'toilet','confidence':.99,'box':[100,100,150,150]}], people)
    assert preview[30:55,50:110].var() < frame[30:55,50:110].var()*.1
    np.testing.assert_array_equal(frame, original)
    np.testing.assert_array_equal(preview[75:], frame[75:])


def test_object_frame_identity_and_stale_or_failed_results_are_hidden(tmp_path):
    app = create_app(tmp_path)
    with TestClient(app) as client:
        authenticate(client)
        engine = Engine(app.state.store, camera_id='fixture')
        app.state.cameras['fixture'] = engine
        engine.frames = {'pose': b'new-pose', 'objects': b'old-object-sample'}
        engine.state.update(sequence=50, source_time=5, object_sequence=40, object_source_time=4,
            object_status='ready', object_submitted_at=time.time(), object_fps=1,
            scene_objects=[{'label':'knife', 'confidence':.95, 'context_only':True}])
        data = client.get('/api/cameras/fixture/frames').json()
        assert data['sequence'] == 50 and data['object_meta']['sequence'] == 40
        assert not data['object_meta']['context_only'] and data['objects']
        assert data['scene_objects'][0]['label'] == 'knife'
        assert client.get('/api/incidents').json() == []
        engine.state['object_submitted_at'] = time.time()-8
        assert client.get('/api/cameras/fixture/frames').json()['objects'] is None
        assert client.get('/api/cameras/fixture/frames').json()['scene_objects'] == []
        assert engine.snapshot()['scene_objects'] == []
        engine.state.update(object_submitted_at=time.time(), object_status='error')
        assert client.get('/api/cameras/fixture/frames').json()['objects'] is None


def test_object_failure_does_not_stop_pose_or_create_incidents(monkeypatch):
    import threading
    import vmd.engine as module
    class Capture:
        finished, error = False, None
        def __init__(self, *args): self.stop_event = threading.Event()
        def start(self): pass
        def stop(self): pass
        def latest(self, after):
            if after >= 20:
                self.finished = True
                return None
            return after+1, (after+1)*.1, np.zeros((64, 96, 3), np.uint8)
    class Pose:
        def __init__(self, *args): pass
        def infer(self, frame): return []
    class Objects:
        stopped = False
        def __init__(self, *args, **kwargs): pass
        def start(self): pass
        def stop(self): Objects.stopped = True
        def poll(self): return 'error', None, 'fixture object failure'
        def submit(self, *args): return False
    records = []
    monkeypatch.setattr(module, 'Capture', Capture)
    monkeypatch.setattr(module, 'PoseModel', Pose)
    monkeypatch.setattr(module, 'ObjectWorker', Objects)
    monkeypatch.setattr(module, 'choose_device', lambda _: 'cpu')
    engine = Engine(SimpleNamespace(error=None, dropped=0, enqueue=lambda *args: records.append(args)))
    engine.run(SimpleNamespace(mode='live', source='fixture', device='cpu', depth='off', threshold=.6,
        hold_seconds=.7, target_fps=1000, object_detection=True, object_fps=1))
    state = engine.snapshot()
    assert state['status'] == 'finished' and state['sequence'] == 20
    assert state['object_meta']['error'] == 'fixture object failure'
    assert not [record for record in records if record[0] == 'incident']
    assert Objects.stopped


@pytest.mark.parametrize('knife_failure', [None, 'load', 'infer'])
def test_worker_exposes_only_specialist_weapons_and_survives_failure(monkeypatch, knife_failure):
    import queue
    import threading
    import vmd.objects as module
    stop = threading.Event()
    frame = np.zeros((64, 96, 3), np.uint8)
    class General:
        def __init__(self, *args): pass
        def infer(self, image):
            assert image is frame
            stop.set()
            return [{'label':'person', 'confidence':.9, 'box':[0,0,50,60]},
                    {'label':'knife', 'confidence':.95, 'box':[60,20,80,50]}]
    class Knife:
        def __init__(self, *args):
            if knife_failure == 'load': raise RuntimeError('fixture missing knife weights')
        def infer(self, image):
            assert image is frame
            if knife_failure == 'infer': raise ValueError('fixture inference failure')
            return [{'label':'knife', 'confidence':.95, 'box':[60,20,80,50], 'context_only':True},
                    {'label':'guns', 'confidence':.9001, 'box':[30,20,50,50], 'context_only':True},
                    {'label':'guns', 'confidence':.90, 'box':[10,10,20,20]},
                    {'label':'knife', 'confidence':.89, 'box':[10,10,20,20]}]
    monkeypatch.setattr(module, 'ObjectModel', General)
    monkeypatch.setattr(module, 'WeaponModel', Knife)
    requests, responses = queue.Queue(), queue.Queue()
    requests.put((7, 2.5, time.time(), frame))
    module.run_objects(requests, responses, stop, 'objects', 'cpu', 'unused')
    responses.get()  # Initial ready message.
    sample = responses.get()
    assert sample['status'] == 'ready' and sample['sequence'] == 7 and sample['source_time'] == 2.5
    assert sample['jpeg']
    assert [item['label'] for item in sample['detections']] == ([] if knife_failure else ['knife', 'gun'])
    assert sample['knife_status'] == ('error' if knife_failure else 'ready')
    assert bool(sample['knife_error']) == bool(knife_failure)
    assert any(item['label'] == 'knife' for item in sample['detections']) == (knife_failure is None)
