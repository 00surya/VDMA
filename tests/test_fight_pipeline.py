"""Exercise the real engine, confirmation, event storage and asynchronous join."""
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

import vmd.engine as module
from test_fight_confirmation import interacting_people, depth_map
from test_bilateral_punches import slow_people
from vmd.stabilization import PoseStabilizer


@pytest.mark.parametrize('detection_mode', ['depth_confirmed', 'responsive'])
@pytest.mark.parametrize('depth_case', ['matched', 'wrong_time', 'wrong_shape', 'separated'])
@pytest.mark.parametrize('cadence', ['continuous', 'brief_strikes', 'slow_bilateral'])
def test_live_engine_only_saves_fight_after_matching_depth_and_three_seconds(monkeypatch, depth_case, detection_mode, cadence):
    capture = None
    progress = []

    class Capture:
        session_id = 'camera-session'
        file = True
        def __init__(self, source):
            nonlocal capture
            capture = self
            self.sequence = 0
            self.finished = False
            self.error = None
            self.stop_event = threading.Event()
        def start(self): pass
        def stop(self): pass
        def identity(self, sequence):
            return {'session_id': self.session_id, 'frame_id': sequence}
        def latest(self, after):
            if after == -1:
                return self.packet
            if self.sequence >= 160:
                self.finished = True
                return None
            progress.append(dict(engine.state))
            self.sequence += 1
            self.packet = (self.sequence, self.sequence * (.125 if cadence != 'continuous' else .1), np.zeros((360, 440, 3), np.uint8))
            return self.packet

    class Pose:
        def __init__(self, *args): pass
        def infer(self, frame):
            if cadence == 'slow_bilateral':
                return slow_people(capture.sequence/8)
            people = interacting_people()
            if cadence == 'brief_strikes':
                for person in people:
                    person.limb_speeds = {}
                    person.limb_motion = {}
                if capture.sequence % 8 == 1:
                    people[0].limb_speeds = {9: 1.4}
                    people[0].limb_motion = {9: .8}
            return people

    class Motion:
        def infer(self, *args): return (.002 if cadence == 'slow_bilateral' else .1), 0
        def person_flow(self, person): return .002 if cadence == 'slow_bilateral' else .1
        def joint_motion(self, person, index):
            return (.25 if index == 9 else 0) if cadence == 'slow_bilateral' else .8

    class Stabilizer:
        def update(self, people, *args): return people

    class Depth:
        def __init__(self, *args, **kwargs):
            self.status = 'ready'
            self.samples = []
            self.result = None
        def start(self): pass
        def stop(self): pass
        def submit(self, sequence, source_time, frame, priority=False):
            if not priority and not capture.finished and sequence % (8 if cadence != 'continuous' else 5):
                return False
            if capture.finished:
                self.samples.clear()
            sample = {'sequence': sequence, 'source_time': source_time + (.01 if depth_case == 'wrong_time' else 0),
                      'depth': depth_map(depth_case == 'separated') if depth_case != 'wrong_shape' else np.zeros((2, 2)),
                      'identity': {'session_id': 'old-session' if depth_case == 'wrong_session' else capture.session_id},
                      'submitted_at': time.time(), 'latency_ms': 1, 'jpeg': b'depth'}
            # Results arrive after two newer pose frames; final draining is immediate.
            self.samples.append((sequence if capture.finished else sequence + 2, sample))
            return True
        def poll(self):
            while self.samples and self.samples[0][0] <= capture.sequence:
                self.result = self.samples.pop(0)[1]
            return 'ready', self.result, None

    class Store:
        error = None
        dropped = 0
        learner = SimpleNamespace(predict=lambda *args: None)
        def __init__(self): self.records = []
        def enqueue(self, kind, payload):
            self.records.append((kind, payload))
            return True

    monkeypatch.setattr(module, 'Capture', Capture)
    monkeypatch.setattr(module, 'PoseModel', Pose)
    monkeypatch.setattr(module, 'Motion', Motion)
    monkeypatch.setattr(module, 'PoseStabilizer', PoseStabilizer if cadence == 'slow_bilateral' else Stabilizer)
    monkeypatch.setattr(module, 'annotate', lambda frame, people: frame.copy())
    monkeypatch.setattr(module, 'DepthWorker', Depth)
    monkeypatch.setattr(module, 'choose_device', lambda _: 'cpu')
    store = Store()
    engine = module.Engine(store)
    # Feed deterministic source-time frames quickly; no real models/cameras used.
    engine.run(SimpleNamespace(mode='live', source='fixture', threshold=.6,
                               hold_seconds=.7, target_fps=1000, eco_mode=depth_case == 'matched', device='cpu', depth='ZipDepth', depth_fps=1, detection_mode=detection_mode))
    assert engine.state['status'] == 'finished', engine.state.get('message')
    incidents = [payload[0] for kind, payload in store.records if kind == 'incident']
    fights = [event for event in incidents if event['event_type'] == 'fight']
    assert any(state.get('assessment') in {'checking_interaction', 'possible_fight'} for state in progress)
    assert all((state.get('alert') or {}).get('event_type') != 'fight' for state in progress if state.get('source_time', 0) < 3)
    if depth_case == 'matched' and detection_mode == 'depth_confirmed':
        assert len(fights) == 1
        assert any(state.get('alert', {}).get('label') == 'FIGHT DETECTED' for state in progress if state.get('alert'))
        assert fights[0]['signals']['fight_timer_seconds'] >= 3
        assert fights[0]['signals']['source_seconds'] > fights[0]['signals']['depth_source_time']
        assert fights[0]['signals']['depth_samples'] >= 2
        if cadence == 'brief_strikes':
            assert fights[0]['signals']['strike_bouts'] >= 3
            assert any(event['event_type'] == 'possible_fight' for event in incidents)
        elif cadence == 'slow_bilateral':
            strikes = fights[0]['signals']['slow_punch_strikes']
            assert {strike['actor'] for strike in strikes} == {1, 2}
            assert not any(state.get('signals', {}).get('candidate') for state in progress), 'The fast rule must not explain this positive'
    else:
        assert fights == []
        assert not any(state.get('assessment') == 'fight_detected' for state in progress)
    # Pending interactions must not be offered to training as normal scenes.
    assert not [payload for kind, payload in store.records if kind == 'normal_sample']
