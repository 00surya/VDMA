from conftest import authenticate
from functools import partial
import multiprocessing as mp
import os
import queue
import signal
import time
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from vmd.api import create_app
from vmd.depth_worker import DepthWorker, replace_latest
from vmd.engine import Engine
from vmd.storage import Store


def echo_depth(requests, responses, stop, *args):
    sequence, source_time, submitted_at, frame = requests.get(timeout=5)
    responses.put({"status": "ready", "sequence": sequence, "source_time": source_time,
                   "submitted_at": submitted_at, "latency_ms": 10, "jpeg": b"fixture"})
    stop.wait(10)


def stuck_depth(requests, responses, stop, *args):
    time.sleep(15)


def report_then_block(lifetime, *args):
    lifetime.send(os.getpid())
    time.sleep(30)


def fixture_parent(lifetime):
    worker = DepthWorker('fixture', 'cpu', 'unused', runner=partial(report_then_block, lifetime))
    worker.start()
    time.sleep(30)


def test_worker_exits_if_owning_server_crashes_during_inference():
    context = mp.get_context('spawn')
    reader, writer = context.Pipe(duplex=False)
    parent = context.Process(target=fixture_parent, args=(writer,))
    child_pid, child_exited = None, False
    parent.start()
    writer.close()
    try:
        assert reader.poll(10), 'Fixture inference worker did not start'
        child_pid = reader.recv()
        parent.kill()  # Bypass normal daemon-child cleanup, as a native crash does.
        parent.join(timeout=3)
        assert not parent.is_alive()
        assert reader.poll(5), 'Inference worker survived the death of its owner'
        with pytest.raises(EOFError):
            reader.recv()
        child_exited = True
    finally:
        if parent.is_alive():
            parent.kill()
            parent.join(timeout=3)
        if child_pid and not child_exited:
            try:
                os.kill(child_pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        reader.close()


def test_worker_rate_limit_provenance_and_shutdown():
    worker=DepthWorker('fixture','cpu','unused',fps=.1,runner=echo_depth)
    worker.start()
    try:
        frame=np.zeros((4,4,3),np.uint8)
        assert worker.submit(23,4.2,frame)
        assert not worker.submit(24,4.3,frame)
        deadline=time.monotonic()+5
        result=None
        while time.monotonic()<deadline:
            _,result,error=worker.poll()
            if result:break
            time.sleep(.02)
        assert result and result['sequence']==23 and result['source_time']==4.2
        assert not error
    finally:
        worker.stop()
    assert not worker.process.is_alive()


def test_hung_depth_can_be_stopped_without_hanging_session():
    worker=DepthWorker('fixture','cpu','unused',runner=stuck_depth)
    worker.start()
    started=time.monotonic()
    worker.stop()
    assert time.monotonic()-started<3
    assert not worker.process.is_alive()


def test_latest_queue_stays_bounded():
    channel=queue.Queue(maxsize=1)
    for n in range(100):replace_latest(channel,n)
    assert channel.qsize()==1 and channel.get_nowait()==99


def test_depth_provenance_and_stale_images_are_filtered(tmp_path):
    app=create_app(tmp_path)
    with TestClient(app) as client:
        authenticate(client)
        engine=app.state.engine
        engine.frames={'pose':b'pose','depth':b'depth'}
        engine.state.update(sequence=20,source_time=2,depth_status='ready',depth_sequence=10,
                            depth_source_time=1,depth_submitted_at=time.time(),last_frame_at=time.time())
        bundle=client.get('/api/frames').json()
        assert bundle['sequence']==20 and bundle['depth_meta']['sequence']==10
        assert bundle['depth_meta']['source_time']==1
        assert bundle['depth'] and not bundle['depth_meta']['stale']
        engine.state['depth_submitted_at']=time.time()-10
        assert client.get('/api/frames').json()['depth'] is None
        assert client.get('/api/frame/depth').status_code==204
        assert client.get('/api/frame/pose').content==b'pose'


def test_pose_keeps_advancing_during_depth_loading_and_failure(monkeypatch,tmp_path):
    import vmd.engine as module
    class FakeCapture:
        def __init__(self,source):
            self.finished=False
            self.error=None
            self.stop_event=__import__('threading').Event()
        def start(self):pass
        def stop(self):pass
        def latest(self,after):return after+1,(after+1)*.1,np.zeros((64,96,3),np.uint8)
    class FakePose:
        def __init__(self,*args):pass
        def infer(self,frame):return []
    class FakeDepth:
        instance=None
        def __init__(self,*args,**kwargs):
            FakeDepth.instance=self
            self.failed=False
            self.stopped=False
        def start(self):pass
        def submit(self,*args):pass
        def poll(self):return ('error',None,'Fixture depth failure') if self.failed else ('loading',None,None)
        def stop(self):self.stopped=True
    monkeypatch.setattr(module,'Capture',FakeCapture)
    monkeypatch.setattr(module,'PoseModel',FakePose)
    monkeypatch.setattr(module,'choose_device',lambda device:'cpu')
    monkeypatch.setattr(module,'DepthWorker',FakeDepth)
    store=Store(tmp_path)
    engine=Engine(store)
    settings=SimpleNamespace(mode='live',source='fixture',device='cpu',depth='MiDaS_small',depth_fps=.5,threshold=.68,hold_seconds=1.2,target_fps=30)
    try:
        engine.start(settings)
        deadline=time.monotonic()+3
        while engine.snapshot()['sequence']<6 and time.monotonic()<deadline:time.sleep(.01)
        state=engine.snapshot()
        assert state['status']=='running' and state['sequence']>=6
        assert state['depth_meta']['status']=='loading'
        FakeDepth.instance.failed=True
        after=state['sequence']
        while engine.snapshot()['sequence']<after+4 and time.monotonic()<deadline:time.sleep(.01)
        state=engine.snapshot()
        assert state['sequence']>=after+4 and state['status']=='running'
        assert state['depth_meta']['error']=='Fixture depth failure'
    finally:
        engine.stop()
        store.close()
    assert FakeDepth.instance.stopped


def test_priority_grab_sample_bypasses_normal_rate_but_stays_bounded(monkeypatch):
    worker = DepthWorker('fixture', 'cpu', 'unused', fps=1)
    worker.requests = queue.Queue(maxsize=1)
    now = [100.0]
    monkeypatch.setattr('vmd.depth_worker.time.monotonic', lambda: now[0])
    frame = np.zeros((4,4,3), np.uint8)
    assert worker.submit(1, 0, frame)
    now[0] += .125
    assert not worker.submit(2, .125, frame)
    assert worker.submit(2, .125, frame, priority=True)
    assert not worker.submit(3, .25, frame, priority=True)
    assert worker.requests.qsize() == 1
    sequence, source_time, _, _ = worker.requests.get_nowait()
    assert (sequence, source_time) == (2, .125)
    worker.responses.close()
