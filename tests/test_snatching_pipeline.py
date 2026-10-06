"""Real engine, frame/depth join, clip writer and response state; no providers."""
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

import vmd.engine as module
from vmd.alerts import AlertAgent
from vmd.storage import Store
from test_snatching import scene


@pytest.mark.parametrize('depth_case', ['matched', 'missing', 'wrong_time', 'separated'])
def test_grab_priority_depth_escalates_one_playable_incident(monkeypatch, tmp_path, depth_case):
    source = SimpleNamespace(t=0, sequence=0)
    priority_frames = []

    class Capture:
        file = False
        def __init__(self, *args):
            self.finished = False
            self.error = None
            self.stop_event = threading.Event()
        def start(self): pass
        def stop(self): pass
        def latest(self, after):
            if source.sequence >= 32:
                self.finished = True
                return None
            source.t = source.sequence/8
            source.sequence += 1
            return source.sequence, source.t, np.zeros((400, 960, 3), np.uint8)

    class Pose:
        def __init__(self, *args): pass
        def infer(self, frame): return scene(source.t)

    class Motion:
        def infer(self, *args): return .1, 0
        def person_flow(self, person): return .1
        def joint_motion(self, *args): return .8

    class Stabilizer:
        def update(self, people, *args): return people

    class Depth:
        def __init__(self, *args, **kwargs):
            self.samples, self.result = [], None
        def start(self): pass
        def stop(self): pass
        def submit(self, sequence, timestamp, frame, priority=False):
            if priority:
                priority_frames.append(sequence)
            if depth_case == 'missing' or (not priority and sequence % 8):
                return False
            depth = np.full((400, 960), .05, np.float32)
            for p in scene(timestamp):
                # Depth covers the torso surface, including points contracted
                # inward from its landmarks; isolated joint dots are not a body.
                torso = [p.keypoints[i] for i in (5, 6, 11, 12)]
                left, right = max(0, int(min(p[0] for p in torso))-12), min(960, int(max(p[0] for p in torso))+13)
                top, bottom = max(0, int(min(p[1] for p in torso))-12), min(400, int(max(p[1] for p in torso))+13)
                if right > left and bottom > top:
                    depth[top:bottom, left:right] = .3 if depth_case=='separated' and p.track_id==1 else .8
            self.samples.append((sequence+2, {'sequence': sequence,
                'source_time': timestamp+(.01 if depth_case=='wrong_time' else 0), 'depth': depth,
                'submitted_at': time.time(), 'latency_ms': 1, 'jpeg': b'depth'}))
            return True
        def poll(self):
            while self.samples and self.samples[0][0] <= source.sequence:
                self.result = self.samples.pop(0)[1]
            return 'ready', self.result, None

    for key, value in [('Capture',Capture),('PoseModel',Pose),('Motion',Motion),
                       ('PoseStabilizer',Stabilizer),('DepthWorker',Depth)]:
        monkeypatch.setattr(module, key, value)
    monkeypatch.setattr(module, 'choose_device', lambda _: 'cpu')
    monkeypatch.setattr(module, 'annotate', lambda frame, people: frame.copy())
    store = Store(tmp_path)
    agent = AlertAgent(store, background=False)
    store.on_incident = agent.register
    try:
        engine = module.Engine(store)
        engine.run(SimpleNamespace(mode='live', source='fixture', threshold=.6, hold_seconds=.7,
            target_fps=1000, eco_mode=False, device='cpu', depth='ZipDepth', depth_fps=1, detection_mode='depth_confirmed'))
        store.jobs.join()
        assert engine.state['status'] == 'finished', engine.state
        assert priority_frames == [2], 'capture depth at the observed approach, between regular samples'
        incidents = store.incidents()
        assert len(incidents) == 1 and incidents[0]['clip']
        event = incidents[0]
        assert (store.clips/event['clip']).stat().st_size > 0
        assert set(engine.state['stage_ms']) == {'pose', 'motion_and_stabilization', 'rules', 'render_encode'}
        if depth_case == 'matched':
            assert event['event_type'] == 'snatching_detected'
            assert event['signals']['depth_source_time'] == .125
            assert agent.get(event['id'])['state'] == 'pending'
            assert event['signals']['stages'][0]['event_type'] == 'possible_snatching'
        else:
            assert event['event_type'] == 'possible_snatching'
            assert agent.get(event['id']) is None
        assert store.error is None
    finally:
        agent.close()
        store.close()
