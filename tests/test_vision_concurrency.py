import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from vmd.vision import PoseModel


@pytest.mark.parametrize("fail_first", [False, True])
def test_mps_cameras_share_one_gpu_section_through_cpu_transfer_and_sync(monkeypatch, fail_first):
    # Reproduce overlapping independent camera instances without requiring a GPU.
    # The real two-camera workload segfaulted in PyTorch/Metal before this guard.
    state = {"active": 0, "peak": 0, "calls": 0, "copies": 0, "syncs": 0}
    state_lock = threading.Lock()
    ready = threading.Barrier(2)

    class Result:
        boxes = keypoints = None

        def cpu(self):
            assert state["active"] == 1
            time.sleep(.01)  # Device-to-host work must also be inside the guard.
            state["copies"] += 1
            return self

    class Model:
        def predict(self, *args, **kwargs):
            with state_lock:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
                state["calls"] += 1
                first = state["calls"] == 1
            time.sleep(.01)
            if fail_first and first:
                raise RuntimeError("Fixture inference error")
            return [Result()]

    def synchronize():
        time.sleep(.01)
        assert state["active"] == 1
        state["syncs"] += 1
        state["active"] -= 1

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(mps=SimpleNamespace(synchronize=synchronize)))

    def camera():
        pose = PoseModel.__new__(PoseModel)
        pose.model, pose.device = Model(), "mps"
        ready.wait(timeout=5)
        try:
            return pose.infer(None)
        except RuntimeError as exc:
            assert str(exc) == "Fixture inference error"
            return "failed"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: camera(), range(2)))
    assert state == {"active": 0, "peak": 1, "calls": 2, "copies": 1 if fail_first else 2, "syncs": 2}
    assert results.count("failed") == int(fail_first)
