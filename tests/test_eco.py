from conftest import authenticate
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from vmd.api import StartRequest, create_app
from vmd.eco import EcoGate
from vmd.engine import Engine


def quiet_gate():
    gate = EcoGate()
    frame = np.zeros((90, 160, 3), np.uint8)
    for tick in range(111):
        gate.observe(frame, tick/10)
    assert gate.state == 'quiet'
    return gate, frame


def test_quiet_periodic_checks_motion_wakeup_and_interaction_hold():
    gate, frame = quiet_gate()
    for worker, period in [('pose', .5), ('objects', 2), ('depth', 10)]:
        assert gate.should_run(worker, 11)
        assert not gate.should_run(worker, 11+period/2)
        assert gate.should_run(worker, 11+period)
    moving = frame.copy()
    moving[25:55, 50:90] = 255
    assert gate.observe(moving, 11.1)
    assert gate.should_run('pose', 11.1)
    assert gate.should_run('depth', 11.1)
    gate, frame = quiet_gate()
    gate.keep_active(11)
    for tick in range(111, 211):
        assert gate.observe(frame, tick/10)
    assert not gate.observe(frame, 21.1)


def test_background_detects_slow_change_and_resolution_or_time_changes_reset():
    gate, frame = quiet_gate()
    for tick, value in enumerate(range(2, 20, 2), 111):
        if gate.observe(np.full_like(frame, value), tick/10):
            break
    assert gate.state == 'active'
    gate, frame = quiet_gate()
    assert gate.observe(np.zeros((180, 90, 3), np.uint8), 11.1)
    gate, frame = quiet_gate()
    assert gate.observe(frame, 1)
    assert gate.should_run('pose', 1)


@pytest.mark.parametrize('toggle', [False, True])
def test_engine_saves_pose_and_depth_work_then_wakes_without_restarting(monkeypatch, toggle):
    import vmd.engine as module
    seen, submits = [], []

    class Capture:
        finished, error = False, None
        def __init__(self, source): self.stop_event = threading.Event()
        def start(self): pass
        def stop(self): pass
        def latest(self, after):
            if after >= 170:
                self.finished = True
                return None
            seq = after+1
            if seq == 150 and toggle:
                engine.set_eco_mode(False)
            image = np.zeros((90, 160, 3), np.uint8)
            if seq >= 150 and not toggle:
                image[20:60, 40:100] = 255
            return seq, seq/10, image

    class Pose:
        def __init__(self, *args): pass
        def infer(self, image):
            seen.append(1)
            return []

    class Depth:
        def __init__(self, *args, **kwargs): pass
        def start(self): pass
        def stop(self): pass
        def poll(self): return 'ready', None, None
        def submit(self, seq, stamp, image):
            submits.append(stamp)
            return True

    monkeypatch.setattr(module, 'Capture', Capture)
    monkeypatch.setattr(module, 'PoseModel', Pose)
    monkeypatch.setattr(module, 'DepthWorker', Depth)
    monkeypatch.setattr(module, 'choose_device', lambda _: 'cpu')
    store = SimpleNamespace(error=None, dropped=0, enqueue=lambda *args: None)
    engine = Engine(store)
    settings = SimpleNamespace(mode='live', source='fixture', device='cpu', depth='ZipDepth',
        depth_fps=1, threshold=.6, hold_seconds=.7, target_fps=1000, eco_mode=True)
    engine.settings = settings
    engine.run(settings)
    state = engine.snapshot()
    assert state['status'] == 'finished', state
    assert state['eco_skipped_frames'] >= 25
    assert state['processed_frames'] == len(seen) < 145
    assert not [stamp for stamp in submits if 11 < stamp < 14.9]
    assert any(stamp >= 15 for stamp in submits)
    assert state['eco_state'] == ('off' if toggle else 'active')
    assert not state['sampling_limited']


def test_eco_api_is_validated_and_changes_only_the_selected_camera(tmp_path):
    app = create_app(tmp_path)
    headers = {'x-vmd-client': 'dashboard'}
    with TestClient(app) as client:
        authenticate(client)
        first = Engine(app.state.store, camera_id='first')
        second = Engine(app.state.store, camera_id='second')
        for engine in (first, second):
            engine.settings = StartRequest(mode='live', source='0')
        app.state.cameras.update(first=first, second=second)
        path = '/api/cameras/first/eco'
        assert client.post(path, json={'enabled':True}).status_code == 403
        assert client.post(path, json={'enabled':'false'}, headers=headers).status_code == 422
        assert client.post(path, json={'enabled':True}, headers=headers).json()['eco_mode']
        assert first.settings.eco_mode and not second.settings.eco_mode
        assert not client.post(path, json={'enabled':False}, headers=headers).json()['eco_mode']
        assert client.post('/api/cameras', json={'mode':'demo','eco_mode':True}, headers=headers).status_code == 422
