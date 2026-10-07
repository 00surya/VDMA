"""Standing-person neck reach -> supported pull -> same-person rapid departure.

These are reviewable movement hypotheses, not proof that property was taken.
No extra model is loaded; pose, image motion and matched depth are shared.
"""
from collections import Counter, deque
from itertools import permutations
from math import hypot, isfinite
import uuid

from .heuristics import Assessment, segment_intersects_box


def body_anchors(person):
    shoulder = tuple(sum(person.keypoints[i][axis] for i in (5, 6))/2 for axis in (0, 1))
    hips = [person.keypoints[i] for i in (11, 12) if person.keypoints[i][2] >= .55]
    hip = tuple(sum(point[axis] for point in hips)/len(hips) for axis in (0, 1)) if hips else shoulder
    return shoulder, hip


def torso(person):
    shoulder, hip = body_anchors(person)
    return tuple((shoulder[axis]+hip[axis])/2 for axis in (0, 1))


def arm_position(person, joint):
    center = torso(person)
    return tuple(person.keypoints[joint][axis]-center[axis] for axis in (0, 1))


def continuous(person, previous):
    """An unchanged numeric track ID cannot bridge a body/scale jump."""
    return (previous is not None and .5 < person.scale/previous.scale < 2
            and hypot(*(torso(person)[axis]-torso(previous)[axis] for axis in (0, 1)))
            <= max(person.scale, previous.scale)*.9)


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
    if (not person.pose_reliable or any(person.keypoints[i][2] < .55 for i in (5, 6))
            or not any(person.keypoints[i][2] >= .55 for i in (11, 12))):
        return False
    shoulder, hip = body_anchors(person)
    return (hip[1]-shoulder[1] > max(person.scale*.12, abs(hip[0]-shoulder[0])*1.4)
            and person.height > (person.box[2]-person.box[0])*1.3)


def torso_extent(person):
    shoulder, hip = body_anchors(person)
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
        self._diagnostic_counts = {'visible_people': 0, 'reliable_poses': 0,
            'tracked_people': 0, 'eligible_standing_people': 0, 'continuous_people': 0,
            'body_pose_missing_people': 0, 'nonstanding_people': 0,
            'unassigned_or_duplicate_people': 0, 'wrist_motion_ready_people': 0,
            'warming_people': 0}
        self._diagnostic_interruption = 'waiting_input'
        self._diagnostic_gap = None
        self._diagnostic_camera_moving = False
        self._observations = {}

    @property
    def diagnostics(self):
        return self.snapshot()

    def snapshot(self):
        """Return bounded status copy; inspection never advances an event rule."""
        pairs = []
        for pair, state in sorted(self.pending.items())[:8]:
            pulled = bool(state.get('possible'))
            escaped = state.get('running_at') is not None
            depth_status = state.get('depth_status', 'waiting')
            blockers = []
            if depth_status in {'uncertain', 'separated'}:
                phase = 'depth_blocked'
                blockers.append('Original-grab depth is separated' if depth_status == 'separated'
                                else 'Original-grab depth is uncertain; this cannot confirm a snatch')
            elif state.get('reported'):
                phase = 'confirmed'
            elif not pulled:
                phase = 'reach_seen'
                blockers.append('Waiting for the same wrist to pull back with image-supported motion within 1 second')
            elif escaped and state.get('depth_time') is None:
                phase = 'waiting_depth'
                blockers.append('Escape observed; waiting for compatible depth from the original grab frame')
            elif state.get('depth_time') is None:
                phase = 'waiting_escape_and_depth'
                blockers += ['Waiting for the same person to move rapidly away',
                             'Waiting for compatible depth from the original grab frame']
            else:
                phase = 'waiting_escape'
                blockers.append('Pull observed; waiting for the same person to move rapidly away with image motion')
            pairs.append({'pair': list(pair), 'phase': phase,
                'actor_track': state['actor'], 'target_track': state['target'],
                'grab_seconds': state['contact'], 'pull_seconds': state.get('pull'),
                'reach_seen': True, 'pull_seen': pulled, 'escape_detected': escaped,
                'escape_support': state.get('escape_support'),
                'depth_status': depth_status, 'depth_source_time': state.get('depth_time'),
                'depth_compatible': depth_status == 'compatible' and state.get('depth_time') is not None,
                'reported': bool(state.get('reported')),
                'departure_observations': len(state.get('path', ())), 'blockers': blockers})
        counts = self._diagnostic_counts
        interruption = self._diagnostic_interruption
        if interruption == 'camera_moving':
            phase, blockers = 'camera_moving', ['Camera movement reset the standing-person snatch check']
        elif interruption == 'source_gap':
            phase, blockers = 'source_gap', ['A source-time gap or rewind reset the standing-person snatch check']
        elif interruption == 'waiting_input':
            phase, blockers = 'waiting_input', ['Waiting for camera observations']
        elif counts['eligible_standing_people'] < 2:
            phase, blockers = 'needs_people', ['Standing-person snatch check needs two reliably tracked standing people']
            if counts['body_pose_missing_people']:
                blockers.append('Both shoulders and at least one hip must be visible with a reliable pose')
            if counts['nonstanding_people']:
                blockers.append('Seated or rider poses do not enter the standing-person rule')
            if counts['unassigned_or_duplicate_people']:
                blockers.append('Unassigned or duplicate track IDs cannot support this check')
        elif interruption == 'warming_up' or counts['continuous_people'] < 2:
            phase, blockers = 'warming_up', ['Waiting for continuous, stable track identities']
        elif pairs:
            rank = {'reach_seen': 1, 'waiting_escape_and_depth': 2, 'waiting_escape': 3,
                    'waiting_depth': 4, 'depth_blocked': 5, 'confirmed': 6}
            selected = max(pairs, key=lambda item: rank[item['phase']])
            phase, blockers = selected['phase'], list(selected['blockers'])
        elif any(set(pair) <= self.previous.keys() for pair in self.cooldowns):
            phase, blockers = 'cooldown', ['This tracked pair has a recent snatch episode in cooldown']
        elif counts['warming_people']:
            phase, blockers = 'warming_wrist_motion', ['Waiting for stable visible wrists and image-supported motion']
        else:
            phase, blockers = 'waiting_reach', ['Waiting for an observed wrist approach into another person’s neck or upper chest']
            if not counts['wrist_motion_ready_people']:
                blockers.append('No supported wrist movement in the eligible poses')
        return {**counts, 'phase': phase, 'timestamp': self.last_time,
                'source_gap': interruption == 'source_gap',
                'source_gap_seconds': self._diagnostic_gap,
                'camera_moving': self._diagnostic_camera_moving,
                'blockers': blockers, 'pending_pairs': len(self.pending),
                'pairs_truncated': len(self.pending) > len(pairs), 'pairs': pairs}

    def _record_diagnostics(self, people, observed, continuous_tracks, dt, camera_motion, interruption=None):
        """Summarize existing gates without admitting or rejecting any track."""
        counts = Counter(person.track_id for person in people)
        assigned = {person.track_id: person for person in people
                    if person.track_id >= 0 and counts[person.track_id] == 1}
        body_ready = [person for person in assigned.values() if person.pose_reliable
                      and all(person.keypoints[i][2] >= .55 for i in (5, 6))
                      and any(person.keypoints[i][2] >= .55 for i in (11, 12))]
        self._observations = {track: min(3, self._observations.get(track, 0)+1)
                              if interruption is None and track in continuous_tracks else 1
                              for track in observed}
        self._diagnostic_counts = {'visible_people': len(people),
            'reliable_poses': sum(bool(person.pose_reliable) for person in people),
            'tracked_people': len(assigned), 'eligible_standing_people': len(observed),
            'continuous_people': len(continuous_tracks),
            'body_pose_missing_people': len(assigned)-len(body_ready),
            'nonstanding_people': len(body_ready)-len(observed),
            'unassigned_or_duplicate_people': len(people)-len(assigned),
            'wrist_motion_ready_people': sum(any(person.keypoints[i][2] >= .55
                and (person.limb_speeds or {}).get(i, 0) >= .35
                and (person.limb_motion or {}).get(i, 0) >= .12 for i in (9, 10))
                for person in continuous_tracks.values()),
            'warming_people': sum(count < 3 for count in self._observations.values())}
        self._diagnostic_interruption = interruption
        self._diagnostic_gap = dt if interruption == 'source_gap' else None
        self._diagnostic_camera_moving = camera_motion > self.rules.camera_speed

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
        first_observation = self.last_time is None
        dt = timestamp-self.last_time if self.last_time is not None else 0
        valid = 0 < dt <= self.rules.max_gap_seconds and camera_motion <= self.rules.camera_speed
        self.last_time = timestamp
        counts = Counter(p.track_id for p in people)
        tracks = {p.track_id: p for p in people if p.track_id >= 0 and counts[p.track_id] == 1 and standing(p)}
        if not valid:
            self.pending.clear()
            self.depth.clear()
            self.previous = tracks
            interruption = ('camera_moving' if camera_motion > self.rules.camera_speed
                            else 'warming_up' if first_observation else 'source_gap')
            self._record_diagnostics(people, tracks, {}, dt, camera_motion, interruption)
            return []
        # The stabilizer already rejects many pose jumps. Keep this gate here
        # as well so neither a reassigned ID nor a scale discontinuity can finish
        # a previous person's grab, even with old compatible depth in memory.
        observed = tracks
        tracks = {track: person for track, person in tracks.items()
                  if continuous(person, self.previous.get(track))}
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
            arm = arm_position(actor, state['joint'])
            arm_travel = hypot(*(arm[axis]-state['arm'][axis] for axis in (0, 1)))/state['actor_scale']
            trigger = False
            if not state.get('possible'):
                if timestamp-state['contact'] > self.grab_seconds:
                    self.pending.pop(pair)
                    continue
                if (point[2] >= .55 and speed >= self.rules.wrist_speed*.7 and motion >= .175
                        and arm_travel >= .08
                        and radius-state['radius'] >= .10):
                    state.update(possible=True, pull=timestamp, pull_distance=arm_travel, origin=torso(actor),
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
            target_center = torso(target)
            state['path'].append((timestamp, center, distance, legs, actor.local_flow or 0, legs_visible, target_center))
            # At two sampled frames/second, a fixed one-second history cannot
            # contain the four/five observations required below. Extend the
            # bounded history instead of accepting fewer escape observations.
            departure_window = min(2.0, max(1.0, dt*4))
            while state['path'] and timestamp-state['path'][0][0] > departure_window:
                state['path'].popleft()
            path = state['path']
            span = timestamp-path[0][0]
            travel = hypot(*(center[i]-path[0][1][i] for i in (0, 1)))/state['scale']
            separation = (distance-path[0][2])/state['scale']
            steps = list(zip(path, list(path)[1:]))
            # Credit only distance gained through the actor's own movement,
            # holding the target fixed at each previous observation. Signed
            # changes preserve sideways escape without counting victim motion
            # or accumulating outward steps while ignoring inward reversals.
            actor_outward = sum(
                hypot(*(b[1][axis]-a[6][axis] for axis in (0, 1)))
                - hypot(*(a[1][axis]-a[6][axis] for axis in (0, 1)))
                for a, b in steps)/state['scale']
            moving_away = sum(b[2] > a[2] for a, b in steps)
            outward_path = (len(path) >= 4 and span >= .4 and actor_outward >= .2
                            and moving_away >= (len(path)-1)*.7)
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
                 'pull_body_heights': state['pull_distance'],
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
                arm, old_arm = arm_position(actor, joint), arm_position(old_actor, joint)
                # Target movement, a translated frozen arm or a camera-local
                # tracking drift is not an observed reach by this actor.
                if hypot(*(arm[axis]-old_arm[axis] for axis in (0, 1))) < actor.scale*.03:
                    continue
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
                                      'joint': joint, 'contact': timestamp, 'radius': radius,
                                      'arm': arm, 'actor_scale': actor.scale}
                break
        self.previous = observed
        self._record_diagnostics(people, observed, tracks, dt, camera_motion)
        return choices
