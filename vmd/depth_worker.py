"""Rate-limited, isolated depth inference; never on the pose/motion critical path."""
import multiprocessing as mp
import os
import queue
import threading
import time


def run_with_parent(runner, *args):
    """A native crash skips server cleanup; its inference children must exit too."""
    parent = mp.parent_process()

    def watch_parent():
        parent.join()
        # Do not wait for native inference or multiprocessing queue finalizers.
        os._exit(0)

    threading.Thread(target=watch_parent, daemon=True, name="parent-watch").start()
    runner(*args)


def replace_latest(channel, value):
    """Bound memory to one pending item. Dropping on a feeder race is acceptable."""
    try:
        channel.put_nowait(value)
    except queue.Full:
        try:
            channel.get_nowait()
        except queue.Empty:
            return False
        try:
            channel.put_nowait(value)
        except queue.Full:
            return False
    return True


def run_depth(requests, responses, stop, model_type, device, model_dir):
    try:
        # Separate CPU pools avoid changing the pose model's global torch settings.
        if hasattr(os, "nice"):
            try:
                os.nice(5)
            except OSError:
                pass
        import cv2
        import torch
        from .vision import DepthModel, depth_view
        cv2.setNumThreads(1)
        torch.set_num_threads(1)
        if model_type == "ZipDepth":
            from .depth import DepthModel as ZipDepthModel
            model = ZipDepthModel(device, model_dir)
        else:
            model = DepthModel(model_type, device, model_dir)
        replace_latest(responses, {"status": "ready"})
        while not stop.is_set():
            try:
                sequence, source_time, submitted_at, frame = requests.get(timeout=.1)
            except queue.Empty:
                continue
            if stop.is_set():
                break
            started = time.monotonic()
            depth = model.infer(frame)
            ok, encoded = cv2.imencode(".jpg", depth_view(depth))
            if not ok:
                raise RuntimeError("Could not encode depth image")
            replace_latest(responses, {
                "status": "ready", "sequence": sequence, "source_time": source_time,
                "submitted_at": submitted_at, "jpeg": encoded.tobytes(),
                "depth": depth.astype("float16"),
                "latency_ms": round((time.monotonic()-started)*1000),
            })
    except Exception as exc:
        message = str(exc) if isinstance(exc, RuntimeError) else f"Depth failed ({type(exc).__name__}); check depth model dependencies"
        replace_latest(responses, {"status": "error", "error": message})


class DepthWorker:
    def __init__(self, model_type, device, model_dir, fps=.5, runner=run_depth):
        context = mp.get_context("spawn")
        self.requests = context.Queue(maxsize=1)
        self.responses = context.Queue(maxsize=1)
        self.stop_event = context.Event()
        self.interval = 1/fps
        self.last_submit = float("-inf")
        self.last_priority = float("-inf")
        self.status = "loading"
        self.error = None
        self.result = None
        self.closed = False
        self.process = context.Process(target=run_with_parent, args=(runner, self.requests, self.responses, self.stop_event, model_type, device, str(model_dir)), daemon=True, name="vmd-depth")

    def start(self):
        self.process.start()

    def submit(self, sequence, source_time, frame, priority=False):
        now = time.monotonic()
        if self.closed or self.status == "error":
            return False
        # A brief grab can fall between normal depth samples. Snapshot its exact
        # frame once; the queue still holds only the latest request.
        if priority and now-self.last_priority < .5:
            return False
        if not priority and now-self.last_submit < self.interval:
            return False
        # Own the queued frame; upstream code may reuse its capture buffer.
        if not replace_latest(self.requests, (sequence, source_time, time.time(), frame.copy())):
            return False
        self.last_submit = now
        if priority:
            self.last_priority = now
        return True

    def poll(self):
        if self.closed:
            return self.status, self.result, self.error
        while True:
            try:
                update = self.responses.get_nowait()
            except queue.Empty:
                break
            self.status = update["status"]
            self.error = update.get("error")
            if "jpeg" in update:
                self.result = update
        if self.process.exitcode is not None and self.status != "error":
            self.status, self.error = "error", "Depth worker exited; pose and motion remain active"
        return self.status, self.result, self.error

    def analyze_file_frame(self, sequence, source_time, frame, stop_event, timeout=60):
        """Backpressure for a recording: retain this exact frame through warmup.

        The caller serializes requests and chooses cadence in video time. Live
        cameras continue using the bounded asynchronous submit/poll path.
        """
        if self.closed or self.status == 'error':
            raise RuntimeError(self.error or 'Depth worker is unavailable')
        request = (sequence, source_time, time.time(), frame.copy())
        deadline = time.monotonic() + timeout
        submitted = False
        while not stop_event.is_set():
            if not submitted:
                submitted = replace_latest(self.requests, request)
            status, result, error = self.poll()
            if status == 'error':
                raise RuntimeError(error or 'Recording depth analysis failed')
            if result and result['sequence'] == sequence and result['source_time'] == source_time:
                return result
            if time.monotonic() >= deadline:
                raise RuntimeError('Depth analysis timed out on this recording. Retry or check the depth worker.')
            stop_event.wait(.01)
        return None

    def stop(self):
        if self.closed:
            return
        self.stop_event.set()
        if self.process.pid is not None:
            self.process.join(timeout=.25)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=2)
        self.closed = True
        self.status = "stopped"
        for channel in (self.requests, self.responses):
            channel.cancel_join_thread()
            channel.close()
