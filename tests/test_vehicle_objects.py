"""Rider context shares the general worker without broadening public weapons."""
import queue
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

import vmd.objects as objects


@pytest.mark.parametrize('unattended,weapons,riders,mode', [
    (False, True, False, 'objects'), (True, True, False, 'objects_and_bags'),
    (True, False, False, 'bags'), (False, False, False, 'objects'),
    (False, True, True, 'objects_riders'), (False, False, True, 'riders'),
    (True, True, True, 'objects_and_bags_riders'), (True, False, True, 'bags_riders'),
])
def test_worker_modes_preserve_legacy_arguments_and_add_optional_riders(monkeypatch, unattended, weapons, riders, mode):
    calls = []
    def initialize(self, *args, **kwargs):
        calls.append((args, kwargs))
        self.process = SimpleNamespace(name='unused')
    monkeypatch.setattr(objects.DepthWorker, '__init__', initialize)
    worker = objects.ObjectWorker('fixture-models', fps=.8, unattended=unattended, weapons=weapons, riders=riders)
    assert calls == [((mode, 'cpu', 'fixture-models'), {'fps': .8, 'runner': objects.run_objects})]
    assert worker.process.name == 'vmd-objects'


def run_worker(monkeypatch, mode, failure=None):
    stop = threading.Event()
    frame = np.zeros((64, 96, 3), np.uint8)
    general_calls, weapon_calls, infer_options = [], [], []
    context = [{'label': 'person', 'confidence': .25, 'box': [0, 0, 30, 60], 'context_only': True},
               {'label': 'motorcycle', 'confidence': .8, 'box': [20, 30, 90, 60], 'context_only': True},
               {'label': 'backpack', 'confidence': .7, 'box': [70, 25, 90, 55], 'context_only': True},
               {'label': 'knife', 'confidence': .99, 'box': [40, 20, 60, 45], 'context_only': True}]
    class General:
        confidence = .4
        def __init__(self, *args, **kwargs):
            general_calls.append((args, kwargs))
            if failure == 'general_load':
                raise RuntimeError('Fixture general model missing')
        def infer(self, image):
            assert image is frame
            infer_options.append(self.confidence)
            stop.set()
            if failure == 'general_infer':
                raise ValueError('Fixture general inference failed')
            return context
    class Weapon:
        def __init__(self, *args, **kwargs):
            weapon_calls.append((args, kwargs))
            if failure == 'weapon_load':
                raise RuntimeError('Fixture specialist missing')
        def infer(self, image):
            assert image is frame
            if failure == 'weapon_infer':
                raise ValueError('Fixture specialist inference failed')
            return [{'label': 'guns', 'confidence': .95, 'box': [35, 20, 55, 40], 'context_only': True},
                    {'label': 'knife', 'confidence': .90, 'box': [35, 20, 55, 40], 'context_only': True}]
    monkeypatch.setattr(objects, 'ObjectModel', General)
    monkeypatch.setattr(objects, 'WeaponModel', Weapon)
    if failure == 'encode':
        monkeypatch.setattr('cv2.imencode', lambda *args: (False, None))
    requests, responses = queue.Queue(), queue.Queue()
    requests.put((17, 4.25, time.time(), frame))
    objects.run_objects(requests, responses, stop, mode, 'cpu', 'fixture-models')
    messages = []
    while not responses.empty():
        messages.append(responses.get_nowait())
    return SimpleNamespace(messages=messages, general=general_calls, weapons=weapon_calls,
                           confidence=infer_options, context=context, frame=frame)


@pytest.mark.parametrize('mode,classes,weapons', [
    ('riders', [0, 3], False), ('objects_riders', [0, 3], True),
    ('bags_riders', [0, 3, 24, 26, 28], False),
    ('objects_and_bags_riders', [0, 3, 24, 26, 28], True),
])
def test_rider_class_selection_context_output_and_specialist_separation(monkeypatch, mode, classes, weapons):
    result = run_worker(monkeypatch, mode)
    assert result.general == [(('fixture-models', 'cpu'), {'class_ids': classes})]
    assert result.confidence == [.2]
    assert bool(result.weapons) is weapons
    assert result.messages[0] == {'status': 'ready'}
    sample = result.messages[1]
    assert sample['status'] == 'ready' and sample['sequence'] == 17 and sample['source_time'] == 4.25
    assert sample['frame_shape'] == [64, 96] and sample['jpeg']
    assert sample['context_objects'] == result.context
    assert [item['label'] for item in sample['detections']] == (['gun'] if weapons else [])
    assert all(not item['context_only'] for item in sample['detections'])
    assert sample['knife_status'] == ('ready' if weapons else 'off')
    assert sample['knife_error'] is None


@pytest.mark.parametrize('mode', ['riders', 'bags_riders'])
def test_context_only_rider_mode_does_not_attempt_loading_weapon_weights(monkeypatch, mode):
    result = run_worker(monkeypatch, mode, 'weapon_load')
    assert result.weapons == []
    sample = result.messages[-1]
    assert sample['status'] == 'ready' and sample['knife_status'] == 'off'
    assert sample['knife_error'] is None and sample['detections'] == []
    assert sample['context_objects']


@pytest.mark.parametrize('failure', ['weapon_load', 'weapon_infer'])
def test_specialist_failure_preserves_rider_context_without_public_general_knives(monkeypatch, failure):
    result = run_worker(monkeypatch, 'objects_riders', failure)
    sample = result.messages[-1]
    assert sample['status'] == 'ready' and sample['knife_status'] == 'error'
    assert sample['knife_error'] and sample['context_objects'] == result.context
    assert sample['detections'] == []


@pytest.mark.parametrize('failure', ['general_load', 'general_infer', 'encode'])
def test_rider_worker_failures_are_returned_as_errors_without_predictions(monkeypatch, failure):
    result = run_worker(monkeypatch, 'riders', failure)
    assert result.weapons == []
    assert result.messages[-1]['status'] == 'error'
    assert result.messages[-1]['error']
    assert 'context_objects' not in result.messages[-1] and 'detections' not in result.messages[-1]


def test_general_model_enforces_rider_class_filter_before_emitting_context():
    class Array:
        def __init__(self, value): self.value = value
        def cpu(self): return self
        def tolist(self): return self.value
    model = objects.ObjectModel.__new__(objects.ObjectModel)
    model.device, model.class_ids, model.confidence = 'cpu', [0, 3], .2
    def predict(frame, **options):
        assert options['classes'] == [0, 3] and options['conf'] == .2
        return [SimpleNamespace(names=dict(enumerate(objects.COCO_CLASSES)),
            boxes=SimpleNamespace(xyxy=Array([[0, 0, 30, 60]]*4),
                                  conf=Array([.25, .7, .9, .99]), cls=Array([0, 3, 24, 43])))]
    model.model = SimpleNamespace(predict=predict)
    context = model.infer(np.zeros((64, 96, 3), np.uint8))
    assert [item['label'] for item in context] == ['person', 'motorcycle']
    assert all(item['context_only'] and item['detector'] == 'YOLO26s' for item in context)
    assert objects.weapon_detections(context) == []


def test_vehicle_monitoring_preview_preserves_frame_and_never_labels_public_weapon():
    frame = np.zeros((64, 96, 3), np.uint8)
    vehicle = {'label': 'motorcycle', 'confidence': .8, 'box': [20, 30, 80, 55]}
    preview = objects.annotate_objects(frame, [], [], [vehicle])
    assert np.any(preview != frame)
    assert not np.any(frame)
    assert objects.weapon_detections([vehicle]) == []
