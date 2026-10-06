"""Keep ByteTrack's process-global ID counter local to each camera."""
import threading


class CameraTracker:
    _lock = threading.Lock()

    def __init__(self):
        from ultralytics.trackers.basetrack import BaseTrack
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.utils import YAML, IterableSimpleNamespace, ROOT
        self.base = BaseTrack
        args = IterableSimpleNamespace(**YAML.load(ROOT / 'cfg/trackers/bytetrack.yaml'))
        with self._lock:
            saved = self.base._count
            try:
                self.tracker = BYTETracker(args=args)
                self.counter = 0
            finally:
                self.base._count = saved

    def update(self, boxes, image):
        # Serialize only the small CPU tracking step, not neural inference.
        with self._lock:
            saved = self.base._count
            self.base._count = self.counter
            try:
                return self.tracker.update(boxes, image)
            finally:
                self.counter = self.base._count
                self.base._count = saved
