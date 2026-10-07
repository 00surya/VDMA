"""Conservative, sampled-frame bag attendance checks; never infer contents.

Call only with a fresh general-object result and its source timestamp. People
and bags must be from that SAME frame, not current pose or old detections.
An object never observed alongside an unambiguous nearby person is unarmed.
The result is a reviewable unattended-bag hypothesis, not ownership, a bomb,
or proof of deliberate abandonment. No model or wall-clock timer is used here.
"""
from dataclasses import dataclass, field
from math import hypot, isfinite
import uuid

from .heuristics import Assessment


BAG_LABELS = frozenset({'backpack', 'handbag', 'suitcase'})


def _number(value):
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)
    except OverflowError:
        return False


def _center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _diagonal(box):
    return hypot(box[2] - box[0], box[3] - box[1])


def _iou(a, b):
    area = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - area
    return area / union if union else 0


def _compatible(a, b):
    size_a, size_b = _diagonal(a), _diagonal(b)
    return (max(size_a, size_b) / min(size_a, size_b) <= 1.6
            and _iou(a, b) >= .35
            and hypot(*(_center(a)[i] - _center(b)[i] for i in (0, 1))) <= max(6, min(size_a, size_b) * .45))


def _near(bag, person):
    # Rectangle separation tolerates a bag on the floor beside a person's feet.
    dx = max(bag[0] - person[2], person[0] - bag[2], 0)
    dy = max(bag[1] - person[3], person[1] - bag[3], 0)
    return hypot(dx, dy) <= max(_diagonal(bag) * 1.25, (person[3] - person[1]) * .35)


def _carried(bag, person):
    x, y = _center(bag)
    return (person[0] <= x <= person[2]
            and person[1] <= y < person[3] - (person[3] - person[1]) * .18)


def _compatible_person(a, b):
    # No identity is inferred; abrupt replacement cannot arm the bag by adding
    # up unrelated one-frame nearby people.
    return (_iou(a, b) >= .2 and hypot(*(_center(a)[i] - _center(b)[i] for i in (0, 1)))
            <= min(_diagonal(a), _diagonal(b)) * .35)


@dataclass
class _Bag:
    track: int
    label: str
    box: tuple
    last_seen: float
    anchor: tuple
    stationary_since: float
    episode_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    attendance_since: float | None = None
    attendant_last_seen: float | None = None
    attendant_samples: int = 0
    attendant_box: tuple | None = None
    armed: bool = False
    absence_since: float | None = None
    reported: bool = False

    def break_attendance(self):
        self.attendance_since = self.attendant_last_seen = None
        self.attendant_samples = 0
        self.attendant_box = None
        self.armed = False
        self.absence_since = None


class UnattendedObjects:
    """Track at most ``max_tracks`` bags using source-time evidence only.

    ``update(objects, timestamp, camera_motion=0, frame_shape=None,
    status='ready')`` returns only newly triggered Assessments. ``objects`` is
    the raw YOLO26s list (person + backpack/handbag/suitcase) from one frame;
    each item has label/confidence/box. ``None`` or non-ready status interrupts
    evidence. Equal timestamps are repeated polling and do not advance time;
    reversed time, a gap > max_gap, shape change or camera motion resets it.

    Arming needs >= 2 source seconds and >= 3 samples of a stationary bag beside
    exactly one nearby confident person. This establishes nearby attendance,
    not an owner identity. Any nearby person interrupts absence. Missing bags,
    carried/moving boxes or ambiguous associations cannot complete an episode.
    """
    attendance_seconds = 2.0
    camera_motion_limit = .09
    max_detections = 100

    def __init__(self, threshold_seconds=60, max_gap=3, max_tracks=24):
        if not _number(threshold_seconds) or not 10 <= threshold_seconds <= 3600:
            raise ValueError('Unattended-object threshold must be 10–3600 seconds')
        if not _number(max_gap) or not 0 < max_gap <= 10:
            raise ValueError('Unattended-object sample gap must be positive and at most 10 seconds')
        if isinstance(max_tracks, bool) or not isinstance(max_tracks, int) or not 1 <= max_tracks <= 128:
            raise ValueError('Unattended-object tracks must be 1–128')
        self.threshold_seconds = float(threshold_seconds)
        self.max_gap, self.max_tracks = float(max_gap), max_tracks
        self.tracks = {}
        self.last_time = self.frame_shape = None
        self._next_track = 0

    def reset(self):
        self.tracks.clear()
        self.last_time = self.frame_shape = None

    @staticmethod
    def _parse(objects, shape):
        bags, people = [], []
        if not isinstance(objects, (list, tuple)) or len(objects) > UnattendedObjects.max_detections:
            return None
        for item in objects:
            if not isinstance(item, dict):
                return None
            label = item.get('label')
            if not isinstance(label, str):
                return None
            if label not in BAG_LABELS and label != 'person':
                continue
            box, confidence = item.get('box'), item.get('confidence')
            if (not isinstance(box, (list, tuple)) or len(box) != 4
                    or any(not _number(value) for value in box)
                    or not _number(confidence) or not 0 <= confidence <= 1
                    or box[0] < 0 or box[1] < 0 or box[2] <= box[0] or box[3] <= box[1]
                    or shape and (box[2] > shape[1] or box[3] > shape[0])):
                return None
            box = tuple(float(value) for value in box)
            if label == 'person' and confidence >= .2:
                # Weak visible people block an absence claim but cannot arm it.
                people.append((box, confidence >= .4))
            elif label in BAG_LABELS and confidence >= .5:
                bags.append((label, box))
        return bags, people

    def update(self, objects, timestamp, camera_motion=0, frame_shape=None, *, status='ready'):
        shape = None
        if frame_shape is not None:
            if (not isinstance(frame_shape, (list, tuple)) or len(frame_shape) not in (2, 3)
                    or any(not _number(value) or value <= 0 or int(value) != value for value in frame_shape[:2])):
                self.reset()
                return []
            shape = tuple(int(value) for value in frame_shape[:2])
        parsed = self._parse(objects, shape)
        if (status != 'ready' or parsed is None or not _number(timestamp) or timestamp < 0
                or not _number(camera_motion) or not 0 <= camera_motion <= self.camera_motion_limit):
            self.reset()
            return []
        timestamp = float(timestamp)
        if self.last_time is not None and timestamp == self.last_time and shape == self.frame_shape:
            return []
        if (self.last_time is not None and not 0 < timestamp - self.last_time <= self.max_gap
                or self.last_time is not None and shape != self.frame_shape):
            self.reset()
        self.last_time, self.frame_shape = timestamp, shape
        bags, people = parsed
        self.tracks = {key: bag for key, bag in self.tracks.items() if timestamp - bag.last_seen <= self.max_gap}

        # A multi-match is not solved by nearest-neighbour guessing. Drop the
        # pending evidence for all involved tracks and leave this sample unarmed.
        matches = {index: [key for key, bag in self.tracks.items()
                           if label == bag.label and _compatible(box, bag.box)]
                   for index, (label, box) in enumerate(bags)}
        reverse = {key: [index for index, keys in matches.items() if key in keys] for key in self.tracks}
        overlap = {index for index, (_, box) in enumerate(bags)
                   if any(other != index and _iou(box, other_box) >= .25
                          for other, (_, other_box) in enumerate(bags))}
        used, events = set(), []
        for index, (label, box) in enumerate(bags):
            keys = matches[index]
            if index in overlap or len(keys) > 1 or keys and len(reverse[keys[0]]) > 1:
                for key in keys:
                    self.tracks[key].break_attendance()
                continue
            if keys:
                bag = self.tracks[keys[0]]
                used.add(bag.track)
                # Compare with a fixed anchor, so small per-sample drift cannot
                # accumulate into a falsely stationary travelling bag.
                stationary = (_iou(box, bag.anchor) >= .7
                              and hypot(*(_center(box)[i] - _center(bag.anchor)[i] for i in (0, 1)))
                              <= max(3, _diagonal(bag.anchor) * .10))
                if not stationary:
                    bag.break_attendance()
                    bag.anchor, bag.stationary_since = box, timestamp
                    bag.episode_id, bag.reported = uuid.uuid4().hex, False
                bag.box, bag.last_seen = box, timestamp
            else:
                if len(self.tracks) >= self.max_tracks:
                    continue  # Do not evict an observed episode to admit clutter.
                self._next_track += 1
                bag = _Bag(self._next_track, label, box, timestamp, box, timestamp)
                self.tracks[bag.track] = bag
                used.add(bag.track)

            nearby = [(person, confident) for person, confident in people if _near(box, person)]
            if nearby:
                if bag.absence_since is not None:
                    # A returning/passing person ends this absence and must
                    # establish a new continuous attendance interval.
                    bag.break_attendance()
                bag.absence_since = None
                if len(nearby) != 1 or not nearby[0][1] or _carried(box, nearby[0][0]):
                    bag.break_attendance()
                    continue
                if bag.attendant_box is not None and not _compatible_person(bag.attendant_box, nearby[0][0]):
                    bag.break_attendance()
                if bag.attendance_since is None:
                    bag.attendance_since = timestamp
                bag.attendant_last_seen = timestamp
                bag.attendant_samples += 1
                bag.attendant_box = nearby[0][0]
                if timestamp - bag.attendance_since >= self.attendance_seconds and bag.attendant_samples >= 3:
                    bag.armed = True
                continue
            if not bag.armed:
                bag.break_attendance()
                continue
            if bag.reported:
                continue
            if bag.absence_since is None:
                bag.absence_since = timestamp
            absent = timestamp - bag.absence_since
            if absent < self.threshold_seconds:
                continue
            bag.reported = True
            events.append(Assessment(.75, 'unattended_object', [
                f'{label.capitalize()} remained stationary with no detected nearby person for {absent:.1f} seconds',
                'A nearby attendant was observed earlier; association and object contents need human review',
            ], {
                'episode_id': bag.episode_id, 'object_track': bag.track,
                'object_label': label, 'object_box': list(box), 'source_time': timestamp,
                'stationary_seconds': round(timestamp - bag.stationary_since, 3),
                'absent_seconds': round(absent, 3), 'threshold_seconds': self.threshold_seconds,
                'absence_since': bag.absence_since, 'prior_attendant_seen': True,
                'attendant_first_seen': bag.attendance_since,
                'attendant_last_seen': bag.attendant_last_seen,
                'attendant_observed_seconds': round(bag.attendant_last_seen - bag.attendance_since, 3),
                'attendant_samples': bag.attendant_samples, 'attendant_reference_box': list(bag.attendant_box),
                'frame_matched': True, 'ownership_verified': False, 'contents_verified': False,
            }, trigger=True, event_type='unattended_object'))
        for key, bag in self.tracks.items():
            if key not in used:
                bag.break_attendance()
                bag.stationary_since = timestamp
                bag.anchor = bag.box
        return events

    def snapshot(self):
        """Read-only countdowns at the last accepted source sample, not now."""
        values = []
        for bag in self.tracks.values():
            visible = bag.last_seen == self.last_time
            absent = max(0, self.last_time - bag.absence_since) if visible and bag.absence_since is not None else 0
            state = ('not_visible' if not visible else 'alerted' if bag.reported else
                     'counting' if bag.armed and bag.absence_since is not None else
                     'attended' if bag.armed else 'observing_attendance' if bag.attendance_since is not None else
                     'waiting_for_attendant')
            values.append({'object_track': bag.track, 'episode_id': bag.episode_id,
                           'label': bag.label, 'box': list(bag.box), 'status': state,
                           'absent_seconds': round(absent, 3),
                           'stationary_seconds': round(max(0, self.last_time - bag.stationary_since), 3),
                           'threshold_seconds': self.threshold_seconds,
                           'prior_attendant_seen': bag.armed, 'last_seen': bag.last_seen})
        return {'source_time': self.last_time, 'threshold_seconds': self.threshold_seconds, 'tracks': values}
