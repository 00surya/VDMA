"""Live latest-frame capture and bounded, deterministic recording analysis."""
import threading
import time
from math import ceil, isfinite
from pathlib import Path
from urllib.parse import urlsplit
import cv2


def validate_source(source):
    if source.isdecimal():
        index = int(source)
        if index > 9:
            raise ValueError("Camera index must be between 0 and 9")
        return index, False
    parsed = urlsplit(source)
    if parsed.scheme in {"http", "https", "rtsp", "rtsps"}:
        if not parsed.hostname:
            raise ValueError("Stream URL needs a hostname")
        return source, False
    path = Path(source).expanduser()
    if path.is_file() and path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}:
        return str(path.resolve()), True
    raise ValueError("Use a camera index, HTTP/RTSP stream URL, or an existing local video path")


class Capture:
    def __init__(self, source, target_fps=8):
        self.source, self.file = validate_source(source)
        if not isfinite(target_fps) or target_fps <= 0:
            raise ValueError("Analysis frame rate must be positive")
        self.target_fps = target_fps
        self.lock = threading.Lock()
        self.ready = threading.Condition(self.lock)
        self.stop_event = threading.Event()
        self.packet = None
        self.error = None
        self.finished = False
        self.completed = False
        self.source_fps = None
        self.total_frames = None
        self.duration_seconds = None
        self.position_seconds = 0.0
        self.thread = threading.Thread(target=self.run, daemon=True, name="camera-capture")

    def start(self):
        self.thread.start()

    def latest(self, after):
        with self.ready:
            # A recording advances only after the caller confirms it finished
            # the preceding packet. Merely fetching it does not release it.
            if self.file and self.packet is not None and self.packet[0] <= after:
                self.position_seconds = self.packet[1]
                self.packet = None
                self.ready.notify_all()
            return self.packet if self.packet is not None and self.packet[0] > after else None

    def metadata(self):
        with self.lock:
            progress = (min(1.0, self.position_seconds/self.duration_seconds)
                        if self.duration_seconds else None)
            if self.file and self.completed:
                progress = 1.0
            return {'source_fps': self.source_fps, 'duration_seconds': self.duration_seconds,
                    'position_seconds': self.position_seconds, 'progress': progress}

    def run(self):
        cap = None
        try:
            if isinstance(self.source, str) and not self.file:
                cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000])
            else:
                cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                raise RuntimeError("Cannot open input. Check the camera URL, network, permissions, or file.")
            fps = cap.get(cv2.CAP_PROP_FPS)
            fps = fps if 1 <= fps <= 240 else 25
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) if self.file else 0
            with self.lock:
                self.source_fps = fps
                self.total_frames = int(frame_count) if isfinite(frame_count) and frame_count > 0 else None
                self.duration_seconds = self.total_frames/fps if self.total_frames else None
            sample_fps = min(fps, self.target_fps)
            sequence, source_index = 0, 0
            while not self.stop_event.is_set():
                if self.file:
                    sample_index = ceil(sequence*fps/sample_fps-1e-9)
                    while source_index < sample_index:
                        if self.stop_event.is_set():
                            return
                        if not cap.grab():
                            self.completed = True
                            return
                        source_index += 1
                ok, frame = cap.read()
                if not ok:
                    if self.file:
                        self.completed = True
                        break
                    raise RuntimeError("Camera disconnected or no frame arrived within the read timeout. Reconnect the source.")
                sequence += 1
                timestamp = source_index/fps if self.file else time.monotonic()
                source_index += 1
                # Keep aspect ratio and even dimensions for evidence encoding.
                scale = min(1, 960/frame.shape[1], 480/frame.shape[0] if self.file else 1)
                w,h = max(2, int(frame.shape[1]*scale)//2*2), max(2, int(frame.shape[0]*scale)//2*2)
                frame = cv2.resize(frame, (w,h))
                with self.ready:
                    self.packet = (sequence, timestamp, frame)
                    if self.file:
                        while self.packet is not None and not self.stop_event.is_set():
                            self.ready.wait(timeout=.1)
        except Exception as exc:
            # Do not echo URLs/credentials from lower-level exception strings.
            self.error = str(exc) if isinstance(exc, RuntimeError) else f"Capture failed ({type(exc).__name__})"
        finally:
            if cap is not None:
                cap.release()
            self.finished = True

    def stop(self):
        self.stop_event.set()
        with self.ready:
            self.ready.notify_all()
        self.thread.join(timeout=6)
