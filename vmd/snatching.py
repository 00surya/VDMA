"""Standing-person neck reach -> supported pull -> same-person rapid departure.

These are reviewable movement hypotheses, not proof that property was taken.
No extra model is loaded; pose, image motion and matched depth are shared.
"""
from collections import Counter, deque
from itertools import permutations
from math import hypot, isfinite
import uuid

from .heuristics import Assessment, segment_intersects_box


def torso(person):
    return tuple(sum(person.keypoints[i][axis] for i in (5, 6, 11, 12))/4 for axis in (0, 1))


def neck_region(person):
    left, right = person.keypoints[5], person.keypoints[6]
    x, y = (left[0]+right[0])/2, (left[1]+right[1])/2
    shoulder_width = hypot(left[0]-right[0], left[1]-right[1])
    hips = [person.keypoints[i] for i in (11, 12) if person.keypoints[i][2] >= .55]
    hip = tuple(sum(p[i] for p in hips)/len(hips) for i in (0, 1)) if hips else (x, y)
    # A collar/upper-chest reach can land over either shoulder. Use the same
    # bounded landmark allowance as body contact, not the full detection box.
    margin = min(person.scale*.12, max(shoulder_width*.30, hypot(x-hip[0], y-hip[1])*.18))
    width = shoulder_width/2 + margin
    return (x-width, y-person.scale*.09, x+width, y+person.scale*.12), (x, y)


def standing(person):
    if not person.pose_reliable or any(person.keypoints[i][2] < .55 for i in (5, 6, 11, 12)):
        return False
    shoulder = tuple(sum(person.keypoints[i][axis] for i in (5, 6))/2 for axis in (0, 1))
    hip = tuple(sum(person.keypoints[i][axis] for i in (11, 12))/2 for axis in (0, 1))
    return (hip[1]-shoulder[1] > max(person.scale*.12, abs(hip[0]-shoulder[0])*1.4)
            and person.height > (person.box[2]-person.box[0])*1.3)


def torso_extent(person):
    shoulder = tuple(sum(person.keypoints[i][axis] for i in (5, 6))/2 for axis in (0, 1))
    hip = tuple(sum(person.keypoints[i][axis] for i in (11, 12))/2 for axis in (0, 1))
    return max(hypot(shoulder[0]-hip[0], shoulder[1]-hip[1]), person.height*.25, 1)


class SnatchingHeuristic:
    grab_seconds = 1.0
    escape_seconds = 5.0

    def __init__(self, rules):
        self.rules = rules
        self.previous, self.pending, self.cooldowns = {}, {}, {}
        self.depth = deque()
        self.last_time = None
        self.last_depth_time = float('-inf')

    def depth_unavailable(self):
        self.depth.clear()
        for state in self.pending.values():
            state['depth_time'] = None
            if state.get('depth_status') not in {'uncertain', 'separated'}:
                state['depth_status'] = 'unavailable'

    def observe_depth(self, source_time, observations, current_time):
        if (not isfinite(source_time) or source_time <= self.last_depth_time
                or not 0 <= current_time-source_time <= 2.5):
            return
        self.last_depth_time = source_time
        self.depth.append((source_time, {tuple(sorted(item['tracks'])): item['status'] for item in observations}))
        while self.depth and current_time-self.depth[0][0] > self.escape_seconds+1:
            self.depth.popleft()

    def update(self, people, timestamp, camera_motion=0):
        dt = timestamp-self.last_time if self.last_time is not None else 0
        valid = 0 < dt <= self.rules.max_gap_seconds and camera_motion <= self.rules.camera_speed
        self.last_time = timestamp
        counts = Counter(p.track_id for p in people)
        tracks = {p.track_id: p for p in people if p.track_id >= 0 and counts[p.track_id] == 1 and standing(p)}
        if not valid:
            self.pending.clear()
            self.depth.clear()
            self.previous = tracks
            return []
        self.cooldowns = {pair: time for pair, time in self.cooldowns.items() if timestamp-time < self.rules.cooldown_seconds}
        choices = []
        # First follow existing episodes, including people who have moved apart.
        for pair, state in list(self.pending.items()):
            actor, target = tracks.get(state['actor']), tracks.get(state['target'])
            if actor is None or target is None or timestamp-state['contact'] > self.escape_seconds:
                if state.get('possible_reported') or state.get('reported'):
                    self.cooldowns[pair] = timestamp
                self.pending.pop(pair)
                continue
            point = actor.keypoints[state['joint']]
            _, neck = neck_region(target)
            radius = hypot(point[0]-neck[0], point[1]-neck[1])/target.scale
            speed = (actor.limb_speeds or {}).get(state['joint'], 0)
            motion = (actor.limb_motion or {}).get(state['joint'], 0)
            trigger = False
            if not state.get('possible'):
                if timestamp-state['contact'] > self.grab_seconds:
                    self.pending.pop(pair)
                    continue
                if (point[2] >= .55 and speed >= self.rules.wrist_speed*.7 and motion >= .175
                        and radius-state['radius'] >= .10):
                    state.update(possible=True, pull=timestamp, origin=torso(actor),
                                 distance=hypot(*(torso(actor)[i]-torso(target)[i] for i in (0, 1))),
                                 scale=actor.scale, torso_extent=torso_extent(actor), path=deque(), depth_time=None,
                                 depth_status='waiting')
            if not state.get('possible'):
                continue
            # Only depth around the observed reach supports the original grab.
            # Later withdrawal/departure frames cannot repair a mismatched grab.
            readings = [(time, values.get(pair, 'uncertain')) for time, values in self.depth
                        if state['contact']-.15 <= time <= state['contact']+.15]
            if readings:
                state['depth_status'] = readings[-1][1]
                state['depth_time'] = readings[-1][0] if readings[-1][1] == 'compatible' else None
            center = torso(actor)
            distance = hypot(*(center[i]-torso(target)[i] for i in (0, 1)))
            legs = any(actor.keypoints[i][2] >= .55 and (actor.limb_speeds or {}).get(i, 0) >= .6 and
                       (actor.limb_motion or {}).get(i, 0) >= .12 for i in (15, 16))
            legs_visible = any(actor.keypoints[i][2] >= .55 for i in (15, 16))
            state['path'].append((timestamp, center, distance, legs, actor.local_flow or 0, legs_visible))
            while state['path'] and timestamp-state['path'][0][0] > 1.0:
                state['path'].popleft()
            path = state['path']
            span = timestamp-path[0][0]
            travel = hypot(*(center[i]-path[0][1][i] for i in (0, 1)))/state['scale']
            separation = (distance-path[0][2])/state['scale']
            moving_away = sum(b[2] > a[2] for a, b in zip(path, list(path)[1:]))
            outward_path = len(path) >= 4 and span >= .4 and moving_away >= (len(path)-1)*.7
            rapid_departure = outward_path and travel/max(span, .001) >= .9 and separation >= .3
            leg_support = sum(p[3] and p[4] >= self.rules.flow_speed*.5 for p in path) >= 2
            # A waist-cropped source cannot provide ankle swing. Require the
            # same still-visible upright torso to move away across multiple
            # image-supported observations; disappearance alone never qualifies.
            torso_support = (len(path) >= 5 and not any(p[5] for p in path)
                             and sum(p[4] >= self.rules.flow_speed for p in path) >= len(path)*.7)
            # The full-height estimate is unsuitable for waist-cropped video.
            # Keep a fixed observed torso reference from the pull and require
            # 1.5 torso lengths/second plus .75 torso lengths of separation.
            torso_departure = (outward_path and torso_support
                               and travel*state['scale']/state['torso_extent']/max(span, .001) >= 1.5
                               and separation*state['scale']/state['torso_extent'] >= .75)
            running = (rapid_departure and leg_support) or torso_departure
            if running:
                state['running_at'] = timestamp
                state['escape_support'] = 'legs' if leg_support else 'visible_torso'
            # Unknown depth can leave a motion-only review warning. Known
            # incompatible grab depth vetoes both warning and confirmation;
            # retain the observed path so a late original-frame result can help.
            if state.get('depth_status') in {'uncertain', 'separated'}:
                continue
            # An unrelated runner, a victim moving away, or walking without leg
            # motion cannot upgrade the actor's episode. Wait briefly for late depth.
            confirmed = (state.get('running_at') is not None and state.get('depth_time') is not None)
            if confirmed and not state.get('reported'):
                state['reported'] = True
                trigger = True
            elif not state.get('possible_reported') and not state.get('reported'):
                state['possible_reported'] = True
                trigger = True
            event_type = 'snatching_detected' if state.get('reported') else 'possible_snatching'
            reasons = ['The same wrist reached another standing person’s neck/upper chest and withdrew sharply',
                       'Image motion supports the pulling wrist; property removal is not verified']
            if state.get('reported'):
                reasons += [('The same actor then moved rapidly away with supported leg motion'
                             if state.get('escape_support') == 'legs'
                             else 'The same visible upper body moved rapidly away; legs were outside the usable view'),
                            'Matched relative depth supports the original grab']
            else:
                reasons += ['Awaiting same-person escape and reliable depth at the grab']
            choices.append(Assessment(.85 if state.get('reported') else .65, event_type, reasons,
                {'episode_id': state['id'], 'actor_track': state['actor'], 'target_track': state['target'],
                 'grab_seconds': state['contact'], 'pull_seconds': state['pull'],
                 'escape_detected': state.get('running_at') is not None,
                 'escape_support': state.get('escape_support'),
                 'depth_status': state.get('depth_status', 'waiting'),
                 'depth_source_time': state.get('depth_time'), 'heuristic': True}, pair, trigger, event_type))
        # Arm only an observed approach into the neck region, not a hand already
        # resting there. Existing poses and per-limb optical flow do all the work.
        for actor, target in permutations(tracks.values(), 2):
            pair = tuple(sorted((actor.track_id, target.track_id)))
            if pair in self.pending or pair in self.cooldowns:
                continue
            old_actor, old_target = self.previous.get(actor.track_id), self.previous.get(target.track_id)
            if old_actor is None or old_target is None:
                continue
            if hypot(*(torso(actor)[i]-torso(target)[i] for i in (0, 1))) > max(actor.height, target.height)*.9:
                continue
            region, neck = neck_region(target)
            old_neck = neck_region(old_target)[1]
            for joint in (9, 10):
                point, old = actor.keypoints[joint], old_actor.keypoints[joint]
                if (min(point[2], old[2]) < .55 or (actor.limb_speeds or {}).get(joint, 0) < .35
                        or (actor.limb_motion or {}).get(joint, 0) < .12):
                    continue
                start = tuple(old[i]+neck[i]-old_neck[i] for i in (0, 1))
                previous_radius = hypot(start[0]-neck[0], start[1]-neck[1])/target.scale
                radius = hypot(point[0]-neck[0], point[1]-neck[1])/target.scale
                if (previous_radius-radius < .03 or hypot(point[0]-start[0], point[1]-start[1]) > actor.scale
                        or not segment_intersects_box(start, point, region)):
                    continue
                # Own-neck adjustments must not masquerade as reaching another person.
                own_neck = neck_region(actor)[1]
                if hypot(point[0]-own_neck[0], point[1]-own_neck[1]) < actor.scale*.15:
                    continue
                self.pending[pair] = {'id': uuid.uuid4().hex, 'actor': actor.track_id, 'target': target.track_id,
                                      'joint': joint, 'contact': timestamp, 'radius': radius}
                break
        self.previous = tracks
        return choices
