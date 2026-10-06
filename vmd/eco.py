"""Cheap scene checks schedule expensive inference; capture stays connected."""
import cv2
import numpy as np


class EcoGate:
    ACTIVE_HOLD_SECONDS = 10.0
    # Stay below the behavior filters' 0.65s maximum frame gap.
    QUIET_POSE_INTERVAL = .5
    QUIET_DEPTH_INTERVAL = 10.0
    QUIET_OBJECT_INTERVAL = 2.0

    def __init__(self, motion_ratio=.008, pixel_delta=12):
        self.motion_ratio, self.pixel_delta = motion_ratio, pixel_delta
        self.previous = self.background = None
        self.timestamp = None
        self.active_until = float('-inf')
        self.last = dict.fromkeys(('pose', 'depth', 'objects'), float('-inf'))
        self.state, self.motion_score = 'starting', 0.0

    def observe(self, frame, timestamp):
        h, w = frame.shape[:2]
        scale = min(1, 160/w, 90/h)
        gray = cv2.cvtColor(cv2.resize(frame, (max(1, round(w*scale)), max(1, round(h*scale))),
                                      interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        reset = (self.previous is None or self.previous.shape != gray.shape
                 or self.timestamp is None or not 0 < timestamp-self.timestamp <= 1)
        self.timestamp = timestamp
        if reset:
            self.previous, self.background = gray, gray.astype(np.float32)
            self.last = dict.fromkeys(self.last, float('-inf'))
            self.motion_score = 1.0
            self.keep_active(timestamp)
            return True
        delta = np.maximum(cv2.absdiff(gray, self.previous),
                           cv2.absdiff(gray, cv2.convertScaleAbs(self.background)))
        ratio = float(np.count_nonzero(delta >= self.pixel_delta) / delta.size)
        average = float(delta.mean() / 255)
        self.motion_score = min(1, max(ratio/self.motion_ratio, average/.025))
        if ratio >= self.motion_ratio or average >= .025:
            self.keep_active(timestamp)
        cv2.accumulateWeighted(gray, self.background, .025)
        self.previous = gray
        self.state = 'active' if timestamp <= self.active_until else 'quiet'
        return self.state == 'active'

    def should_run(self, worker, timestamp):
        interval = {'pose': self.QUIET_POSE_INTERVAL, 'depth': self.QUIET_DEPTH_INTERVAL,
                    'objects': self.QUIET_OBJECT_INTERVAL}[worker]
        if self.state != 'quiet' or timestamp-self.last[worker] >= interval-1e-6:
            self.last[worker] = timestamp
            return True
        return False

    def keep_active(self, timestamp):
        self.active_until = timestamp + self.ACTIVE_HOLD_SECONDS
        self.state = 'active'

    def snapshot(self):
        return {'eco_state': self.state, 'eco_motion_score': round(self.motion_score, 3),
                'eco_quiet_pose_fps': 1/self.QUIET_POSE_INTERVAL,
                'eco_quiet_depth_fps': 1/self.QUIET_DEPTH_INTERVAL}
