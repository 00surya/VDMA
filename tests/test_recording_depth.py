"""Recorded analysis waits for exact depth without contacting any provider."""
import queue
import threading
import time
from functools import partial
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import vmd.engine as engine_module
from vmd.alerts import AlertAgent
from vmd.depth_worker import DepthWorker, replace_latest
from vmd.storage import Store
from test_fight_confirmation import depth_map, interacting_people


def recording_depth_runner(requests, responses, stop, *args, warmup=.15, delay=.04, mismatch=False):
    if stop.wait(warmup):
        return
    replace_latest(responses, {'status': 'ready'})
    while not stop.is_set():
        try:
            sequence, source_time, submitted_at, frame = requests.get(timeout=.1)
        except queue.Empty:
            continue
        if stop.wait(delay):
            return
        depth = depth_map() if frame.shape[:2] == (360, 440) else np.full(frame.shape[:2], sequence, np.float32)
        replace_latest(responses, {'status': 'ready', 'sequence': sequence,
            'source_time': source_time+(.01 if mismatch else 0), 'submitted_at': submitted_at,
            'depth': depth, 'jpeg': b'depth-fixture', 'latency_ms': round(delay*1000),
            'frame_value': float(frame[0, 0, 0])})


def stalled_recording_depth(requests, responses, stop, *args):
    stop.wait(5)


def test_recording_waits_through_warmup_and_ignores_live_wall_clock_throttle():
    worker = DepthWorker('fixture', 'cpu', 'unused', fps=.1, runner=recording_depth_runner)
    worker.start()
    stop = threading.Event()
    try:
        # These timestamps suppress submit(), but must not drop a serialized
        # recording request selected by its original video timestamp.
        worker.last_submit = worker.last_priority = time.monotonic()
        assert not worker.submit(999, 99, np.zeros((4, 4, 3), np.uint8), priority=True)
        samples = []
        for sequence, source_time in ((1, 0), (2, .125), (3, .25)):
            frame = np.full((4, 4, 3), sequence*10, np.uint8)
            sample = worker.analyze_file_frame(sequence, source_time, frame, stop, timeout=5)
            samples.append(sample)
        assert [(sample['sequence'], sample['source_time']) for sample in samples] == [(1, 0), (2, .125), (3, .25)]
        assert [sample['frame_value'] for sample in samples] == [10, 20, 30]
        assert all(np.all(sample['depth'] == sample['sequence']) for sample in samples)
    finally:
        worker.stop()
    assert not worker.process.is_alive()


def test_recording_does_not_accept_result_for_same_sequence_at_wrong_source_time():
    worker = DepthWorker('fixture', 'cpu', 'unused', runner=partial(recording_depth_runner, warmup=0, delay=0, mismatch=True))
    worker.start()
    try:
        deadline = time.monotonic()+5
        while worker.poll()[0] == 'loading' and time.monotonic() < deadline:
            time.sleep(.01)
        assert worker.status == 'ready'
        with pytest.raises(RuntimeError, match='timed out'):
            worker.analyze_file_frame(7, 1.25, np.zeros((4, 4, 3), np.uint8), threading.Event(), timeout=.3)
        assert worker.result and worker.result['source_time'] == 1.26
    finally:
        worker.stop()


@pytest.mark.parametrize('action', ['stop', 'timeout'])
def test_recording_depth_wait_is_interruptible_and_bounded(action):
    worker = DepthWorker('fixture', 'cpu', 'unused', runner=stalled_recording_depth)
    worker.start()
    stop = threading.Event()
    timer = threading.Timer(.08, stop.set) if action == 'stop' else None
    if timer:
        timer.start()
    start = time.monotonic()
    try:
        if action == 'timeout':
            with pytest.raises(RuntimeError, match='timed out'):
                worker.analyze_file_frame(1, 0, np.zeros((4, 4, 3), np.uint8), stop, timeout=.08)
        else:
            assert worker.analyze_file_frame(1, 0, np.zeros((4, 4, 3), np.uint8), stop, timeout=5) is None
        assert time.monotonic()-start < .8
    finally:
        if timer:
            timer.cancel()
        worker.stop()


def test_real_recording_capture_and_slow_depth_keep_selected_frames_and_never_register_calls(monkeypatch, tmp_path):
    path = tmp_path/'recording.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 24, (440, 360))
    assert writer.isOpened()
    for index in range(48):
        writer.write(np.full((360, 440, 3), index*3, np.uint8))
    writer.release()
    poses, depth_samples, order = [], [], []

    class Pose:
        def __init__(self, *args): pass
        def infer(self, frame):
            poses.append(float(frame[0, 0, 0]))
            order.append(('pose', len(poses)))
            return interacting_people()

    class Motion:
        def infer(self, *args): return .1, 0
        def person_flow(self, *args): return .1
        def joint_motion(self, *args): return .8

    class Stabilizer:
        def update(self, people, *args): return people

    class RecordingDepth(DepthWorker):
        def __init__(self, model_type, device, model_dir, fps):
            super().__init__(model_type, device, model_dir, fps,
                runner=partial(recording_depth_runner, warmup=.25, delay=.18))
        def submit(self, *args, **kwargs):
            raise AssertionError('Recording analysis must not use dropping live submission')
        def analyze_file_frame(self, sequence, source_time, frame, stop_event, timeout=5):
            sample = super().analyze_file_frame(sequence, source_time, frame, stop_event, timeout)
            depth_samples.append((sequence, source_time, sample['frame_value']))
            order.append(('depth', sequence))
            return sample

    for name, value in [('PoseModel', Pose), ('Motion', Motion), ('PoseStabilizer', Stabilizer), ('DepthWorker', RecordingDepth)]:
        monkeypatch.setattr(engine_module, name, value)
    monkeypatch.setattr(engine_module, 'choose_device', lambda device: 'cpu')
    store = Store(tmp_path/'data')
    agent = AlertAgent(store, background=False)
    store.on_incident = agent.register
    engine = engine_module.Engine(store)
    try:
        engine.run(SimpleNamespace(mode='live', source=str(path), threshold=.6, hold_seconds=.3,
            fight_confirmation_seconds=.5, target_fps=8, eco_mode=True, device='cpu',
            depth='ZipDepth', depth_fps=1, detection_mode='depth_confirmed', object_detection=False))
        store.jobs.join()
        assert engine.state['status'] == 'finished', engine.state
        assert engine.state['processed_frames'] == 16
        assert engine.state['eco_skipped_frames'] == 0
        assert poses == pytest.approx([index*3 for index in range(0, 48, 3)], abs=1)
        assert [(sequence, timestamp) for sequence, timestamp, _ in depth_samples] == [(1, 0), (2, .125), (10, 1.125)]
        assert [value for _, _, value in depth_samples] == pytest.approx([0, 9, 81], abs=1)
        assert order.index(('depth', 2)) < order.index(('pose', 3))
        assert engine.capture.completed and engine.capture.metadata()['progress'] == 1
        incidents = store.incidents()
        assert len(incidents) == 1 and incidents[0]['event_type'] == 'fight'
        incident = incidents[0]
        assert incident['signals']['live_camera'] is False
        assert incident['signals']['depth_samples'] >= 2
        assert incident['signals']['required_seconds'] == .5
        assert incident['clip'] and (store.clips/incident['clip']).stat().st_size > 0
        assert agent.get(incident['id']) is None
        with store.connect() as connection:
            assert connection.execute('SELECT COUNT(*) FROM response_alerts').fetchone()[0] == 0
            assert connection.execute('SELECT COUNT(*) FROM telemetry WHERE live_camera != 0').fetchone()[0] == 0
        assert store.error is None
    finally:
        engine.stop()
        agent.close()
        store.close()
