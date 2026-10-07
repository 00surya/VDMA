"""Opt-in rider/near-neck/drag hypotheses for manual evidence review only.

This does not detect a necklace, ownership, theft or physical contact. Its
broader raw-hand allowance is separate from standing-snatching confirmation.
The caller joins object results with poses from the EXACT original frame;
current stabilized poses supply continuous motion, never an old object frame.
"""
from collections import Counter, deque
from dataclasses import replace
from math import hypot, isfinite
import uuid

from .heuristics import Assessment, Rules, segment_intersects_box
from .snatching import arm_position, body_anchors, neck_region, standing, torso


def _distance(first, second):
    return hypot(*(first[axis]-second[axis] for axis in (0, 1)))


def _center(box):
    return ((box[0]+box[2])/2, (box[1]+box[3])/2)


def _area(box):
    return (box[2]-box[0])*(box[3]-box[1])


def _intersection(first, second):
    return max(0, min(first[2], second[2])-max(first[0], second[0]))*max(
        0, min(first[3], second[3])-max(first[1], second[1]))


def _iou(first, second):
    shared = _intersection(first, second)
    return shared/(_area(first)+_area(second)-shared)


def _diagonal(box):
    return hypot(box[2]-box[0], box[3]-box[1])


def _shape(value):
    if (not isinstance(value, (tuple, list)) or len(value) < 2
            or any(not isinstance(v, (int, float)) or not isfinite(v) or v <= 0 for v in value[:2])):
        return None
    return tuple(value[:2])


def _box(value, shape):
    if (not isinstance(value, (tuple, list)) or len(value) != 4
            or any(not isinstance(v, (int, float)) or not isfinite(v) for v in value)
            or value[2] <= value[0] or value[3] <= value[1]
            or min(value[:2]) < 0 or value[2] > shape[1] or value[3] > shape[0]):
        return None
    return tuple(value)


def _visible(person, joint, shape):
    point = person.keypoints[joint]
    return (len(point) == 3 and all(isfinite(v) for v in point) and point[2] >= .55
            and 0 <= point[0] < shape[1] and 0 <= point[1] < shape[0])


def _reliable(person, shape):
    return (person.pose_reliable and isfinite(person.scale) and person.scale > 0
            and len(person.keypoints) == 17 and _box(person.box, shape) is not None
            and all(_visible(person, joint, shape) for joint in (5, 6))
            and any(_visible(person, joint, shape) for joint in (11, 12)))


def _people(people, shape):
    counts = Counter(p.track_id for p in people)
    return {p.track_id: p for p in people[:32] if p.track_id >= 0
            and counts[p.track_id] == 1 and _reliable(p, shape)}


def _continuous(current, previous):
    return (previous is not None and .5 < current.scale/previous.scale < 2
            and _distance(torso(current), torso(previous)) <= max(current.scale, previous.scale)*.9)


def _rider(person, motorcycle):
    x1, y1, x2, y2 = person.box
    lower = (x1, y1+(y2-y1)*.55, x2, y2)
    bottom = ((x1+x2)/2, y2)
    return (_intersection(lower, motorcycle)/max(1, _area(lower)) >= .4
            and motorcycle[0] <= bottom[0] <= motorcycle[2]
            and motorcycle[1] <= bottom[1] <= motorcycle[3])


class VehicleSnatchingHeuristic:
    """A visible near-neck pull, rider movement and supported target reaction.

    ``observe_vehicles`` consumes a caller-verified object/pose source frame.
    ``update(raw_people, stable_people, timestamp, frame_shape, camera_motion)``
    returns a list of Assessments, always ``possible_snatching`` and review-only.
    ``snapshot`` reports blocked/checking phases without raising an incident.
    """
    grab_seconds = 1.0
    reaction_seconds = 3.0
    context_seconds = 2.5
    target_grace_seconds = 1.0
    near_margin = .18
    max_vehicles, max_pairs = 8, 12

    def __init__(self, rules=None):
        self.rules = rules or Rules()
        self.reset()

    def reset(self):
        self.vehicles, self.bindings, self.pending, self.cooldowns = {}, {}, {}, {}
        self.previous_raw, self.previous_stable = {}, {}
        self.last_time = self.frame_shape = None
        self.last_vehicle_time, self.last_vehicle_sequence = float('-inf'), None
        self.next_vehicle = 0
        self.blockers = ['Waiting for a supported rider reach near another person’s neck']

    def observe_vehicles(self, objects, people, source_time, current_time, sequence,
                         camera_motion=0, shape=None):
        frame_shape = _shape(shape)
        if (frame_shape is None or not all(isfinite(v) for v in (source_time, current_time, camera_motion))
                or not 0 <= current_time-source_time <= self.context_seconds
                or camera_motion > self.rules.camera_speed or camera_motion < 0
                or not isinstance(objects, (list, tuple)) or len(objects) > 100):
            self.bindings.clear()
            self.blockers = ['Fresh matched motorcycle observations are unavailable']
            return
        if source_time <= self.last_vehicle_time or sequence == self.last_vehicle_sequence:
            return
        if self.frame_shape is not None and frame_shape != self.frame_shape:
            self.reset()
        self.last_vehicle_time, self.last_vehicle_sequence = source_time, sequence
        self.frame_shape = frame_shape
        motorcycles, general_people = [], []
        for item in objects:
            if not isinstance(item, dict):
                self.bindings.clear()
                return
            label = item.get('label', item.get('class_name'))
            if label not in {'motorcycle', 'person'}:
                continue
            box, confidence = _box(item.get('box'), frame_shape), item.get('confidence', 0)
            if box is None or not isinstance(confidence, (int, float)) or not isfinite(confidence):
                self.bindings.clear()
                return
            if not .4 <= confidence <= 1:
                continue
            (motorcycles if label == 'motorcycle' else general_people).append((confidence, box))
        # Nested lower-score motorcycle predictions must not manufacture two
        # riders/vehicles. Distinct overlapping bikes remain ambiguous below.
        accepted = []
        for confidence, box in sorted(motorcycles, reverse=True):
            if any(_iou(box, other) >= .65 or (_intersection(box, other)/min(_area(box), _area(other)) >= .9
                    and _distance(_center(box), _center(other)) <= max(_diagonal(box), _diagonal(other))*.25)
                   for _, other in accepted):
                continue
            accepted.append((confidence, box))
        self.vehicles = {key: value for key, value in self.vehicles.items()
                         if source_time-value['time'] <= self.context_seconds}
        matched, used = [], set()
        for confidence, box in accepted[:self.max_vehicles]:
            choices = [(key, value) for key, value in self.vehicles.items() if key not in used
                       and .45 <= _area(box)/_area(value['box']) <= 2.2
                       and _distance(_center(box), _center(value['box'])) <= max(_diagonal(box), _diagonal(value['box']))*.75
                       and _iou(box, value['box']) >= .1]
            choices.sort(key=lambda item: _iou(box, item[1]['box']), reverse=True)
            if len(choices) == 1 or len(choices) > 1 and _iou(box, choices[0][1]['box'])-_iou(box, choices[1][1]['box']) >= .15:
                key, value = choices[0]
            else:
                if len(self.vehicles) >= self.max_vehicles:
                    continue
                self.next_vehicle += 1
                key, value = self.next_vehicle, {'samples': deque(maxlen=16)}
            value.update(box=box, time=source_time, sequence=sequence, confidence=confidence)
            value['samples'].append((source_time, _center(box)))
            self.vehicles[key] = value
            matched.append((key, box))
            used.add(key)
        candidates = _people(people, frame_shape)
        associations = {}
        for track, person in candidates.items():
            if not any(_iou(person.box, box) >= .35 for _, box in general_people):
                continue
            found = [key for key, box in matched if _rider(person, box)]
            if len(found) == 1:
                associations[track] = found[0]
        counts = Counter(associations.values())
        # One unambiguous observed rider per motorcycle. A different rider or
        # vehicle cannot finish a previously armed pair episode.
        self.bindings = {track: {'vehicle': key, 'time': source_time, 'sequence': sequence}
                         for track, key in associations.items() if counts[key] == 1}

    def _binding(self, track, timestamp):
        binding = self.bindings.get(track)
        return binding if binding and 0 <= timestamp-binding['time'] <= self.context_seconds else None

    def _high_five(self, actor, target, old_target, wrist, shape, scale):
        hand = actor.keypoints[wrist]
        neck = neck_region(target)[1]
        return (hand[1] < neck[1]-scale*.05 and
                any(_visible(target, joint, shape) and _visible(old_target, joint, shape)
                   and target.keypoints[joint][1] < neck[1]-scale*.05
                   and old_target.keypoints[joint][1] < neck_region(old_target)[1][1]-scale*.05
                   and _distance(hand[:2], target.keypoints[joint][:2]) < scale*.12
                   for joint in (9, 10)))

    def update(self, raw_people, stable_people, timestamp, frame_shape=None, camera_motion=0):
        shape = _shape(frame_shape)
        if (shape is None or not all(isfinite(v) for v in (timestamp, camera_motion))
                or camera_motion < 0 or camera_motion > self.rules.camera_speed):
            self.reset()
            self.blockers = ['Camera movement or invalid frame interrupts rider checks']
            return []
        raw, stable = _people(raw_people, shape), _people(stable_people, shape)
        dt = timestamp-self.last_time if self.last_time is not None else 0
        if self.last_time is None:
            self.last_time, self.frame_shape = timestamp, shape
            self.previous_raw, self.previous_stable = raw, stable
            return []
        if not 0 < dt <= self.rules.max_gap_seconds or shape != self.frame_shape:
            self.reset()
            self.last_time, self.frame_shape = timestamp, shape
            self.previous_raw, self.previous_stable = raw, stable
            self.blockers = ['Frame timing or size changed; rider evidence reset']
            return []
        self.last_time = timestamp
        continuous = {track: person for track, person in stable.items()
                      if track in raw and _continuous(person, self.previous_stable.get(track))}
        self.cooldowns = {key: time for key, time in self.cooldowns.items()
                          if timestamp-time < self.rules.cooldown_seconds}
        choices = []
        self.blockers = []
        for pair, state in list(self.pending.items()):
            actor, target = continuous.get(state['actor']), continuous.get(state['target'])
            raw_actor, raw_target = raw.get(state['actor']), raw.get(state['target'])
            binding = self._binding(state['actor'], timestamp)
            vehicle = self.vehicles.get(state['vehicle'])
            # A visible changed/unreliable/duplicate target is contradictory
            # evidence, rather than a brief missing observation. Never let an
            # earlier reaction finish a new same-ID person's episode.
            target_discontinuous = (target is None and
                                    any(p.track_id == state['target'] for p in (*raw_people, *stable_people)))
            if (actor is None or binding is None or binding['vehicle'] != state['vehicle'] or vehicle is None
                    or target_discontinuous
                    or timestamp-state['contact'] > self.grab_seconds and state.get('pull') is None
                    or state.get('pull') is not None and timestamp-state['pull'] > self.reaction_seconds):
                if state.get('reported'):
                    self.cooldowns[pair] = timestamp
                self.pending.pop(pair)
                continue
            if target is not None and raw_target is not None:
                state.update(target_center=torso(target), target_time=timestamp)
            elif timestamp-state['target_time'] > self.target_grace_seconds:
                self.pending.pop(pair)
                continue
            if state.get('pull') is None:
                if target is None or raw_target is None or not _visible(raw_actor, state['wrist'], shape):
                    state['blockers'] = ['Waiting for a clear pulling wrist and target torso']
                    continue
                hand = raw_actor.keypoints[state['wrist']]
                neck = neck_region(replace(raw_target, body_scale=state['target_scale']))[1]
                radius = _distance(hand[:2], neck)/state['target_scale']
                arm = arm_position(actor, state['wrist'])
                if ((actor.limb_speeds or {}).get(state['wrist'], 0) >= self.rules.wrist_speed*.7
                        and (actor.limb_motion or {}).get(state['wrist'], 0) >= .175
                        and _distance(arm, state['arm'])/state['actor_scale'] >= .08
                        and radius-state['radius'] >= .10):
                    state.update(pull=timestamp, actor_origin=torso(actor), target_origin=torso(target),
                                 vehicle_origin=_center(vehicle['box']), reaction=None,
                                 rider_path=deque(maxlen=64), target_path=deque(maxlen=64))
                else:
                    state['blockers'] = ['Waiting for an image-supported wrist withdrawal']
                    continue
            center = torso(actor)
            state['rider_path'].append((timestamp, center, actor.local_flow or 0))
            while state['rider_path'] and (timestamp-state['rider_path'][0][0] > 2 or
                    timestamp-state['rider_path'][0][0] > 1 and len(state['rider_path']) > 4):
                state['rider_path'].popleft()
            path = state['rider_path']
            span = timestamp-path[0][0]
            rider_speed = _distance(center, path[0][1])/state['actor_scale']/max(span, .001)
            rider_travel = _distance(center, state['actor_origin'])/state['actor_scale']
            vehicle_travel = _distance(_center(vehicle['box']), state['vehicle_origin'])/state['actor_scale']
            rider_supported = (len(path) >= 4 and span >= .4 and rider_speed >= .20
                               and sum(flow >= .12 for _, _, flow in path) >= 3)
            vehicle_reobserved = (vehicle['time'] > state['pull'] and binding['time'] == vehicle['time'])
            if target is not None:
                target_center = torso(target)
                state['target_path'].append((timestamp, target_center, target.local_flow or 0))
                while state['target_path'] and (timestamp-state['target_path'][0][0] > 2 or
                        timestamp-state['target_path'][0][0] > 1 and len(state['target_path']) > 4):
                    state['target_path'].popleft()
                reaction_path = state['target_path']
                reaction_span = timestamp-reaction_path[0][0]
                reaction_speed = _distance(target_center, reaction_path[0][1])/state['target_scale']/max(reaction_span, .001)
                reaction_travel = _distance(target_center, state['target_origin'])/state['target_scale']
                toward = tuple(state['actor_origin'][axis]-state['target_origin'][axis] for axis in (0, 1))
                toward_length = hypot(*toward)
                toward_travel = (sum((target_center[axis]-state['target_origin'][axis])*toward[axis]
                                     for axis in (0, 1))/toward_length/state['target_scale']) if toward_length else 0
                shoulder, hip = body_anchors(target)
                fallen = (abs(hip[0]-shoulder[0]) > abs(hip[1]-shoulder[1])*1.5
                          and abs(hip[0]-shoulder[0]) > state['target_scale']*.12
                          and target.box[2]-target.box[0] > (target.box[3]-target.box[1])*1.2)
                if (len(reaction_path) >= 4 and reaction_span >= .4 and reaction_travel >= .25
                        and reaction_speed >= .4 and sum(flow >= .12 for _, _, flow in reaction_path) >= 3
                        and (toward_travel >= .2 or fallen)):
                    state['reaction'] = {'source_time': timestamp, 'net_body_heights': reaction_travel,
                        'speed_body_heights_s': reaction_speed, 'toward_body_heights': toward_travel,
                        'falling_posture': fallen}
            blockers = []
            if state['reaction'] is None:
                blockers.append('Waiting for abrupt supported movement of the other person')
            if not vehicle_reobserved or vehicle_travel < .2:
                blockers.append('Waiting for the same original motorcycle to move')
            if not rider_supported or rider_travel < .2:
                blockers.append('Waiting for continuous supported rider movement')
            if target is None:
                blockers.append('Other person temporarily hidden; last reliable position retained briefly')
            state.update(blockers=blockers, rider_travel=rider_travel, vehicle_travel=vehicle_travel,
                         rider_speed=rider_speed, target_age=timestamp-state['target_time'])
            ready = state['reaction'] is not None and vehicle_reobserved and vehicle_travel >= .2 and rider_travel >= .2 and rider_supported
            if ready or state.get('reported'):
                trigger = not state.get('reported')
                state['reported'] = True
                reasons = ['Near-neck pull followed by rider movement and abrupt movement of the other person',
                           'The same originally associated motorcycle moved; vehicle/person association needs review',
                           'Contact and the object involved are uncertain; this does not establish theft',
                           'Manual review only; no confirmed-snatching or automatic-call classification']
                choices.append(Assessment(.65, 'possible_snatching', reasons, {
                    'episode_id': state['id'], 'vehicle_context': True, 'review_only': True,
                    'contact_uncertain': True, 'property_removal_verified': False,
                    'actor_track': state['actor'], 'target_track': state['target'],
                    'vehicle_track': state['vehicle'], 'vehicle_sequence': vehicle['sequence'],
                    'vehicle_source_time': vehicle['time'], 'grab_seconds': state['contact'],
                    'pull_seconds': state['pull'], 'near_neck_margin_body_heights': self.near_margin,
                    'raw_neck_gap_body_heights': state['neck_gap'],
                    'rider_movement_body_heights': rider_travel, 'vehicle_movement_body_heights': vehicle_travel,
                    'rider_speed_body_heights_s': rider_speed, 'target_reaction': state['reaction'],
                    'target_reference_age_seconds': state['target_age'], 'heuristic': True,
                }, pair, trigger, 'possible_snatching'))
        for actor_track, actor in continuous.items():
            binding = self._binding(actor_track, timestamp)
            if binding is None:
                continue
            for target_track, target in continuous.items():
                pair = tuple(sorted((actor_track, target_track)))
                if (actor_track == target_track or pair in self.pending or pair in self.cooldowns
                        or len(self.pending) >= self.max_pairs or not standing(target)):
                    continue
                raw_actor, raw_target = raw[actor_track], raw[target_track]
                old_actor, old_target = self.previous_raw.get(actor_track), self.previous_raw.get(target_track)
                old_stable = self.previous_stable.get(actor_track)
                if old_actor is None or old_target is None or old_stable is None:
                    continue
                scaled_target = replace(raw_target, body_scale=target.scale)
                region, neck = neck_region(scaled_target)
                old_neck = neck_region(replace(old_target, body_scale=target.scale))[1]
                margin = target.scale*self.near_margin
                near = (region[0]-margin, region[1], region[2]+margin, region[3])
                for wrist in (9, 10):
                    if (not _visible(raw_actor, wrist, shape) or not _visible(old_actor, wrist, shape)
                            or (actor.limb_speeds or {}).get(wrist, 0) < .35
                            or (actor.limb_motion or {}).get(wrist, 0) < .12):
                        continue
                    point, old_point = raw_actor.keypoints[wrist], old_actor.keypoints[wrist]
                    start = tuple(old_point[axis]+neck[axis]-old_neck[axis] for axis in (0, 1))
                    radius = _distance(point[:2], neck)/target.scale
                    arm, old_arm = arm_position(actor, wrist), arm_position(old_stable, wrist)
                    # Raw geometry locates the visible hand, while the filtered
                    # poses measure supported motion. Requiring their one-frame
                    # deltas to coincide rejects an approach as the filter settles.
                    stable_neck, old_stable_neck = neck_region(target)[1], neck_region(self.previous_stable[target_track])[1]
                    stable_start = tuple(old_stable.keypoints[wrist][axis]+stable_neck[axis]-old_stable_neck[axis]
                                         for axis in (0, 1))
                    approach = (_distance(stable_start, stable_neck)-
                                _distance(actor.keypoints[wrist][:2], stable_neck))/target.scale
                    if (approach < .03
                            or _distance(arm, old_arm) < actor.scale*.03
                            or _distance(start, point[:2]) > actor.scale
                            or not segment_intersects_box(start, point, near)
                            or not near[0] <= point[0] <= near[2] or not near[1] <= point[1] <= near[3]
                            or _distance(point[:2], neck_region(raw_actor)[1]) < actor.scale*.15
                            or self._high_five(raw_actor, raw_target, old_target, wrist, shape, target.scale)):
                        continue
                    gap = max(region[0]-point[0], 0, point[0]-region[2])/target.scale
                    self.pending[pair] = {'id': uuid.uuid4().hex, 'actor': actor_track, 'target': target_track,
                        'vehicle': binding['vehicle'], 'wrist': wrist, 'contact': timestamp,
                        'radius': radius, 'arm': arm, 'actor_scale': actor.scale,
                        'target_scale': target.scale, 'target_center': torso(target), 'target_time': timestamp,
                        'pull': None, 'neck_gap': gap, 'blockers': ['Waiting for an image-supported wrist withdrawal']}
                    break
        self.previous_raw, self.previous_stable = raw, stable
        if not self.pending and not self.blockers:
            self.blockers = ['Waiting for a supported rider reach near another person’s neck']
        return choices

    def snapshot(self):
        candidates = [{'actor_track': state['actor'], 'target_track': state['target'],
                       'vehicle_track': state['vehicle'], 'episode_id': state['id'],
                       'phase': 'possible_snatching' if state.get('reported') else
                                'checking_reaction_and_rider' if state.get('pull') is not None else 'checking_pull',
                       'grab_seconds': state['contact'], 'pull_seconds': state.get('pull'),
                       'contact_uncertain': True, 'review_only': True,
                       'blockers': list(state.get('blockers', [])),
                       'rider_movement_body_heights': state.get('rider_travel', 0),
                       'vehicle_movement_body_heights': state.get('vehicle_travel', 0),
                       'target_reaction': dict(state['reaction']) if state.get('reaction') else None}
                      for state in self.pending.values()]
        riders = sum(self._binding(track, self.last_time) is not None
                     for track in self.previous_stable) if self.last_time is not None else 0
        phase = candidates[0]['phase'] if candidates else 'monitoring' if riders else 'waiting_rider_context'
        blockers = list(self.blockers) if candidates or riders else [
            'Waiting for an unambiguous rider and fresh motorcycle sample']
        return {'source_time': self.last_time, 'vehicle_context': True, 'review_only': True,
                'phase': phase, 'blockers': blockers, 'candidates': candidates, 'riders': riders,
                'vehicles': len(self.vehicles)}
