"""Dimensionless heuristic signals. A score is not a probability of violence."""
from dataclasses import dataclass, field, replace
from collections import deque
from itertools import combinations
from math import acos, degrees, hypot


@dataclass
class Person:
    track_id: int
    box: tuple[float, float, float, float]
    keypoints: list[tuple[float, float, float]]  # COCO x, y, confidence
    depth: float | None = None  # frame-normalized inverse depth, never metres
    limb_speeds: dict | None = None
    body_scale: float | None = None
    pose_reliable: bool = True
    local_flow: float | None = None
    limb_motion: dict | None = None  # measured pixel flow, body heights / second

    @property
    def height(self):
        return max(1.0, self.box[3] - self.box[1])

    @property
    def center(self):
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def scale(self):
        return self.body_scale or self.height


@dataclass
class Rules:
    threshold: float = 0.60
    hold_seconds: float = 0.7
    fight_confirmation_seconds: float = 3.0
    cooldown_seconds: float = 12.0
    max_gap_seconds: float = 0.65
    signal_grace_seconds: float = 0.25
    proximity_heights: float = 0.85
    wrist_speed: float = 0.85  # body heights per second, relative to torso
    flow_speed: float = 0.04  # frame diagonals per second, person-local regions
    camera_speed: float = 0.09
    keypoint_confidence: float = 0.45
    strike_window_seconds: float = 2.4
    strike_evidence_fraction: float = 0.6
    contact_margin: float = 0.12
    contact_motion_speed: float = 0.35  # body heights / second around the moving limb


@dataclass
class Assessment:
    score: float = 0.0
    state: str = "observing"
    reasons: list[str] = field(default_factory=list)
    signals: dict = field(default_factory=dict)
    pair: tuple[int, int] | None = None
    trigger: bool = False
    event_type: str = "fight"
    events: list = field(default_factory=list)
    candidates: list = field(default_factory=list)


def segment_intersects_box(start, end, box):
    """Test the observed hand path against a rectangle without extrapolating it."""
    lower, upper = 0.0, 1.0
    for axis in (0, 1):
        delta = end[axis] - start[axis]
        if abs(delta) < 1e-9:
            if not box[axis] <= start[axis] <= box[axis + 2]:
                return False
            continue
        entry = (box[axis] - start[axis]) / delta
        leave = (box[axis + 2] - start[axis]) / delta
        lower = max(lower, min(entry, leave))
        upper = min(upper, max(entry, leave))
        if lower > upper:
            return False
    return True


def body_contact_regions(person, rules):
    """Use visible body landmarks; extended arms must not enlarge the target."""
    shoulders = [person.keypoints[index] for index in (5, 6)]
    hips = [person.keypoints[index] for index in (11, 12)
            if person.keypoints[index][2] >= rules.keypoint_confidence]
    if any(point[2] < rules.keypoint_confidence for point in shoulders) or not hips:
        return []
    shoulder_width = hypot(shoulders[0][0]-shoulders[1][0],
                           shoulders[0][1]-shoulders[1][1])
    shoulder_center = tuple(sum(point[axis] for point in shoulders)/2 for axis in (0, 1))
    hip_center = tuple(sum(point[axis] for point in hips)/len(hips) for axis in (0, 1))
    torso_length = hypot(shoulder_center[0]-hip_center[0], shoulder_center[1]-hip_center[1])
    if max(shoulder_width, torso_length) <= 1:
        return []
    # Padding tolerates landmark error, but never uses the full detection box:
    # boxes often contain empty space between an outstretched hand and torso.
    # Projected shoulders can coincide in a side view. A small torso-length
    # allowance preserves visible body thickness without inventing a hidden pose.
    extent_limit = min(person.scale, person.height) * rules.contact_margin
    margin = min(extent_limit, max(shoulder_width * .30, torso_length * .18))
    torso = shoulders + hips
    regions = [(min(p[0] for p in torso)-margin, min(p[1] for p in torso)-margin,
                max(p[0] for p in torso)+margin, max(p[1] for p in torso)+margin)]
    face = [person.keypoints[index] for index in range(5)
            if person.keypoints[index][2] >= rules.keypoint_confidence]
    if face:
        # A punch to the head remains contact even above the shoulder rectangle.
        head_margin = min(extent_limit, max(shoulder_width * .35, torso_length * .18))
        regions.append((min(p[0] for p in face)-head_margin, min(p[1] for p in face)-head_margin,
                        max(p[0] for p in face)+head_margin, max(p[1] for p in face)+head_margin))
    return regions


def guard_contact(actor, target, index, rules, old_actor=None, old_target=None):
    """A punch may stop at a bent, raised guard instead of reaching the torso."""
    if index not in (9, 10):
        return None
    point = actor.keypoints[index]
    anchors = [target.keypoints[i] for i in (0, 5, 6, 11, 12)]
    if min(point[2], *(anchor[2] for anchor in anchors)) < rules.keypoint_confidence:
        return None
    shoulder_y = (target.keypoints[5][1]+target.keypoints[6][1])/2
    hip_y = (target.keypoints[11][1]+target.keypoints[12][1])/2
    if hip_y <= shoulder_y:
        return None
    regions = body_contact_regions(target, rules)
    for joint in (9, 10):
        guard, elbow, shoulder = (target.keypoints[i] for i in (joint, joint-2, joint-4))
        if min(guard[2], elbow[2], shoulder[2]) < rules.keypoint_confidence:
            continue
        u = tuple(guard[i]-elbow[i] for i in (0, 1))
        v = tuple(shoulder[i]-elbow[i] for i in (0, 1))
        denominator = hypot(*u)*hypot(*v)
        if not denominator or degrees(acos(max(-1, min(1, sum(a*b for a, b in zip(u, v))/denominator)))) > 125:
            continue
        if not target.keypoints[0][1] <= guard[1] <= shoulder_y+.25*(hip_y-shoulder_y):
            continue
        distance = min((hypot(max(r[0]-guard[0], 0, guard[0]-r[2]),
                              max(r[1]-guard[1], 0, guard[1]-r[3])) for r in regions), default=float('inf'))
        if distance > .15*target.scale or hypot(point[0]-guard[0], point[1]-guard[1]) > .08*min(actor.scale, target.scale):
            continue
        if old_actor is not None:
            old = old_actor.keypoints[index]
            if old[2] < rules.keypoint_confidence:
                continue
            own = tuple(sum(actor.keypoints[i][axis] for i in (5, 6))/2 for axis in (0, 1))
            previous = tuple(sum(old_actor.keypoints[i][axis] for i in (5, 6))/2 for axis in (0, 1))
            motion = tuple((point[i]-own[i])-(old[i]-previous[i]) for i in (0, 1))
            toward = tuple(guard[i]-old[i] for i in (0, 1))
            denominator = hypot(*motion)*hypot(*toward)
            if not denominator or sum(a*b for a, b in zip(motion, toward))/denominator < .5:
                continue
        return {'contact_kind': 'guard', 'target_track': target.track_id, 'target_joint': joint}
    return None


class FightHeuristic:
    def __init__(self, rules=None):
        self.rules = rules or Rules()
        self.previous = {}
        self.last_time = None
        self.pending = {}
        self.last_positive = {}
        self.cooldowns = {}
        self.episodes = set()
        self.strike_history = {}
        self.limb_phases = {}
        self.strike_cycles = {}
        self.slow_punch_phases = {}
        self.slow_punch_cycles = {}
        self.slow_punch_strikes = {}

    def reset_slow_punch_pair(self, pair):
        """Discard arm trajectories after reliable separation or track changes."""
        pair = tuple(sorted(pair))
        self.slow_punch_cycles.pop(pair, None)
        self.slow_punch_strikes.pop(pair, None)
        self.slow_punch_phases = {key: value for key, value in self.slow_punch_phases.items()
                                  if tuple(sorted(key[:2])) != pair}

    def _slow_punch_evidence(self, a, b, previous, timestamp, reaches, proximity, valid, dt):
        """Separate evidence for repeated bilateral punches below the fast rule.

        Each wrist must approach the other torso, extend from its own shoulder,
        enter a body/head region, then withdraw with measured motion on that wrist.
        The score scales speed and wrist image motion to .25 body scales/second;
        it is an evidence score, not a probability. The confirmation layer applies
        the user's threshold and mandatory matching-depth window independently.
        """
        pair = tuple(sorted((a.track_id, b.track_id)))
        signals = {'slow_punch_activity': False, 'slow_punch_contact': False,
                   'slow_punch_striking': False, 'slow_punch_strikes': [],
                   'slow_punch_striking_score': 0.0,
                   'slow_punch_striking_contacts': [],
                   'slow_punch_score': 0.0, 'bilateral_punching': False,
                   'bilateral_punch_cycles': []}
        if not valid or not (a.pose_reliable and b.pose_reliable):
            self.reset_slow_punch_pair(pair)
            return signals
        torso_centers, shoulder_centers = {}, {}
        for person in (a, b):
            old = previous.get(person.track_id)
            visible = tuple(i for i in (5, 6, 11, 12)
                            if person.keypoints[i][2] >= self.rules.keypoint_confidence)
            old_visible = tuple(i for i in (5, 6, 11, 12)
                                if old and old.keypoints[i][2] >= self.rules.keypoint_confidence)
            if (not old or not old.pose_reliable or visible != old_visible
                    or 5 not in visible or 6 not in visible or len(visible) < 3):
                self.reset_slow_punch_pair(pair)
                return signals
            torso_centers[person.track_id] = tuple(
                tuple(sum(pose.keypoints[i][axis] for i in visible)/len(visible) for axis in (0, 1))
                for pose in (person, old))
            shoulder_centers[person.track_id] = tuple(
                tuple(sum(pose.keypoints[i][axis] for i in (5, 6))/2 for axis in (0, 1))
                for pose in (person, old))
        cycles = self.slow_punch_cycles.setdefault(pair, deque())
        strikes = self.slow_punch_strikes.setdefault(pair, deque())
        while strikes and timestamp-strikes[0]['contact'] > 8:
            strikes.popleft()
        while cycles and timestamp-cycles[0]['time'] > 8:
            cycles.popleft()
        active_motion_score = 0.0
        for actor, target in ((a, b), (b, a)):
            old_actor = previous[actor.track_id]
            torso, old_torso = torso_centers[target.track_id]
            shoulder, old_shoulder = shoulder_centers[actor.track_id]
            for index in (9, 10):
                key = (actor.track_id, target.track_id, index)
                point, old_point = actor.keypoints[index], old_actor.keypoints[index]
                if min(point[2], old_point[2]) < self.rules.keypoint_confidence:
                    self.slow_punch_phases.pop(key, None)
                    continue
                radius = hypot(point[0]-torso[0], point[1]-torso[1])/target.scale
                old_radius = hypot(old_point[0]-old_torso[0], old_point[1]-old_torso[1])/target.scale
                extension = hypot(point[0]-shoulder[0], point[1]-shoulder[1])/actor.scale
                old_extension = hypot(old_point[0]-old_shoulder[0], old_point[1]-old_shoulder[1])/actor.scale
                # The stabilizer subtracts its deadband from fast-rule velocity.
                # Use the actual filtered, torso-relative displacement here, still
                # requiring measured image motion on the very same wrist.
                speed = hypot((point[0]-shoulder[0])-(old_point[0]-old_shoulder[0]),
                              (point[1]-shoulder[1])-(old_point[1]-old_shoulder[1])) / (actor.scale*dt)
                image_motion = (actor.limb_motion or {}).get(index, 0)
                supported = speed >= .10 and image_motion >= .12
                inward = old_radius-radius > .003 and extension-old_extension > .003
                outward = radius-old_radius > .003 and old_extension-extension > .003
                phase = self.slow_punch_phases.get(key)
                if phase and (timestamp-phase['start'] > 4.5 or timestamp-phase['last_support'] > 1.5):
                    self.slow_punch_phases.pop(key, None)
                    phase = None
                if phase is None and supported and inward:
                    phase = {'start': timestamp, 'last_support': timestamp,
                             'origin': old_radius, 'origin_extension': old_extension,
                             'closest': radius, 'extended': extension, 'contact': None, 'approach_samples': 0}
                    self.slow_punch_phases[key] = phase
                if phase and phase['contact'] is None and supported and outward:
                    # Do not stitch separate waves or pose jumps into one punch.
                    self.slow_punch_phases.pop(key, None)
                    continue
                if not phase or not supported or not (inward or (outward and phase['contact'] is not None)):
                    continue
                if inward:
                    phase['approach_samples'] += 1
                phase['last_support'] = timestamp
                phase['closest'] = min(phase['closest'], radius)
                phase['extended'] = max(phase['extended'], extension)
                contact = reaches(actor, target, (index,))
                contact_details = {'contact_kind': 'body', 'target_track': target.track_id, 'target_joint': None}
                # A staged punch can stop just short of the body, and filtered
                # wrists lag by a few pixels. Bound this slow-cycle allowance by
                # visible anatomy; the fast single-strike rule stays unchanged.
                if not contact:
                    left, right = target.keypoints[5], target.keypoints[6]
                    shoulder_width = hypot(left[0]-right[0], left[1]-right[1])
                    torso_length = hypot(torso[0]-((left[0]+right[0])/2),
                                         torso[1]-((left[1]+right[1])/2))*2
                    allowance = min(.08*min(actor.scale, target.scale),
                                    max(.5*shoulder_width, .2*torso_length))
                    contact = any(region[0]-allowance <= point[0] <= region[2]+allowance
                                  and region[1]-allowance <= point[1] <= region[3]+allowance
                                  for region in body_contact_regions(target, self.rules))
                # Torso padding cannot turn a hand hanging below someone's
                # hips into a body-directed punch.
                hip_y = max(target.keypoints[i][1] for i in (11, 12)
                            if target.keypoints[i][2] >= self.rules.keypoint_confidence)
                contact = contact and point[1] <= hip_y
                guarded = guard_contact(actor, target, index, self.rules, old_actor)
                if guarded:
                    # A raised hand inside padded torso bounds is still the
                    # reached surface; depth must compare that specific guard.
                    contact, contact_details = True, guarded
                signals['slow_punch_activity'] = True
                signals['slow_punch_contact'] |= contact
                active_motion_score = max(active_motion_score, .45*min(1, speed/.25)
                                          + .30*min(1, image_motion/.25))
                if (phase['contact'] is None and contact
                        and (phase['approach_samples'] >= 2 or contact_details['contact_kind'] == 'guard')
                        and phase['origin']-radius >= .08
                        and extension-phase['origin_extension'] >= .05):
                    phase['contact'] = timestamp
                    if not any(strike['actor'] == actor.track_id
                               and abs(strike['contact']-timestamp) < .5 for strike in strikes):
                        strikes.append({'actor': actor.track_id, 'wrist': index,
                                        'start': phase['start'], 'contact': timestamp,
                                        **contact_details,
                                        'score': round(.25*proximity + .45*min(1, speed/.25)
                                                       + .30*min(1, image_motion/.25), 3)})
                # Start the review clock at a supported strike, not after three
                # complete punch-and-withdrawal cycles. Bare approaches do not count.
                signals['slow_punch_striking'] |= phase['contact'] is not None
                if phase['contact'] is not None:
                    signals['slow_punch_striking_contacts'].append({'actor': actor.track_id,
                        'wrist': index, 'contact': phase['contact'],
                        'score': round(.25*proximity + .45*min(1, speed/.25) + .30*min(1, image_motion/.25), 3)})
                    signals['slow_punch_striking_score'] = max(signals['slow_punch_striking_score'],
                        round(.25*proximity + .45*min(1, speed/.25) + .30*min(1, image_motion/.25), 3))
                if (phase['contact'] is not None and outward and radius-phase['closest'] >= .08
                        and phase['extended']-extension >= .05 and timestamp-phase['start'] >= .5):
                    # Two hands completing one embrace/push are one actor action,
                    # not two independent punches toward the bilateral minimum.
                    if not any(cycle['actor'] == actor.track_id
                               and abs(cycle['contact']-phase['contact']) < .5 for cycle in cycles):
                        cycles.append({'actor': actor.track_id, 'wrist': index, 'start': phase['start'],
                                       'contact': phase['contact'], 'time': timestamp})
                    self.slow_punch_phases.pop(key, None)
        if signals['slow_punch_activity']:
            signals['slow_punch_score'] = round(.25*proximity + active_motion_score, 3)
        # Both arms moving into one embrace are not independent punch actions.
        # This slow-strike route is separate from the existing grapple detector.
        for actor in (a, b):
            contacts = [phase['contact'] for key, phase in self.slow_punch_phases.items()
                        if key[0] == actor.track_id and tuple(sorted(key[:2])) == pair
                        and phase['contact'] is not None]
            if len(contacts) == 2 and abs(contacts[0]-contacts[1]) <= .25:
                for strike in strikes:
                    if strike['actor'] == actor.track_id and min(abs(strike['contact']-t) for t in contacts) <= .25:
                        strike['two_handed'] = True
        signals['bilateral_punch_cycles'] = list(cycles)
        signals['slow_punch_strikes'] = list(strikes)
        signals['bilateral_punching'] = (len(cycles) >= 3 and all(
            any(cycle['actor'] == actor.track_id for cycle in cycles) for actor in (a, b)))
        return signals

    def update(self, people, timestamp, local_flow=0.0, camera_motion=0.0):
        r = self.rules
        dt = timestamp - self.last_time if self.last_time is not None else 0
        valid_time = 0 < dt <= r.max_gap_seconds
        if not valid_time:
            self.previous.clear()
            self.pending.clear()
            self.last_positive.clear()
            self.episodes.clear()
            self.strike_history.clear()
            self.limb_phases.clear()
            self.strike_cycles.clear()
            self.slow_punch_phases.clear()
            self.slow_punch_cycles.clear()
            self.slow_punch_strikes.clear()
        previous_people = self.previous
        wrist = {}
        for person in people:
            old = self.previous.get(person.track_id)
            speeds = []
            if valid_time and old:
                for index in (9, 10):
                    a, b = person.keypoints[index], old.keypoints[index]
                    if min(a[2], b[2]) >= r.keypoint_confidence:
                        # Subtract torso translation to avoid treating walking as punching.
                        c, d = person.center, old.center
                        speeds.append(hypot((a[0]-c[0])-(b[0]-d[0]), (a[1]-c[1])-(b[1]-d[1])) / (dt * person.height))
            wrist[person.track_id] = max(speeds, default=0)
            if person.limb_speeds is not None:
                wrist[person.track_id] = max((person.limb_speeds.get(i,0) for i in (9,10)),default=0)
        self.previous = {p.track_id: p for p in people}
        self.last_time = timestamp
        self.cooldowns = {k: v for k, v in self.cooldowns.items() if timestamp - v < r.cooldown_seconds}
        best = Assessment(signals={"people": len(people), "local_flow": round(local_flow, 3), "camera_motion": round(camera_motion, 3),
            "reliable_poses": sum(p.pose_reliable for p in people),
            "threshold": r.threshold, "required_seconds": r.hold_seconds,
            "blockers": ["Fight detection needs two visible tracked people"] if len(people)<2 else []})
        if camera_motion > r.camera_speed:
            self.pending.clear()
            self.last_positive.clear()
            self.episodes.clear()
            self.strike_history.clear()
            self.limb_phases.clear()
            self.strike_cycles.clear()
            self.slow_punch_phases.clear()
            self.slow_punch_cycles.clear()
            self.slow_punch_strikes.clear()
            best.state = "camera_moving"
            best.reasons = ["Camera motion is too high; interaction alerts are paused"]
            best.signals["blockers"] = list(best.reasons)
            return best
        active_pairs = set()
        events = []
        candidates = []
        active_fights = []
        visible_pairs = set()
        contact_regions = {person.track_id: body_contact_regions(person, r) for person in people}
        for a, b in combinations(people, 2):
            if a.track_id < 0 or b.track_id < 0 or a.track_id == b.track_id:
                continue
            pair = tuple(sorted((a.track_id, b.track_id)))
            visible_pairs.add(pair)
            # Torso-derived scales can be only half the visible standing height.
            # Use the full visible body extent for pair distance, while retaining
            # torso-normalized limb speeds and the same narrow contact region.
            scale = (max(a.scale, a.height) + max(b.scale, b.height)) / 2
            distance = hypot(a.center[0]-b.center[0], a.center[1]-b.center[1]) / scale
            near = distance < r.proximity_heights
            # Detailed limb/path checks are only useful for nearby pairs.
            if not near:
                self.strike_history.pop(pair, None)
                self.strike_cycles.pop(pair, None)
                self.reset_slow_punch_pair(pair)
                continue
            speed = max(wrist[a.track_id], wrist[b.track_id])
            # Strong depth disagreement vetoes a pair; missing depth is explicitly neutral.
            depth_ok = a.depth is None or b.depth is None or abs(a.depth-b.depth) < 0.3
            proximity = max(0, 1 - distance / (r.proximity_heights * 1.5))
            limb = min(1, speed / r.wrist_speed)
            pair_flow = max(a.local_flow or 0,b.local_flow or 0) if a.local_flow is not None and b.local_flow is not None else local_flow
            motion = min(1, max(0, pair_flow) / r.flow_speed)
            # Include the observed limb path: a punch can land and retract between
            # processed frames. Require the same limb and consecutive track history.
            def reaches(actor, target, indices):
                regions = contact_regions[target.track_id]
                old_actor = previous_people.get(actor.track_id) if valid_time else None
                old_target = previous_people.get(target.track_id) if valid_time else None
                for index in indices:
                    point = actor.keypoints[index]
                    if point[2] < r.keypoint_confidence:
                        continue
                    if any(region[0] <= point[0] <= region[2] and region[1] <= point[1] <= region[3]
                           for region in regions):
                        return True
                    if not old_actor or not old_target or not old_actor.pose_reliable or not old_target.pose_reliable:
                        continue
                    old_point = old_actor.keypoints[index]
                    if old_point[2] < r.keypoint_confidence:
                        continue
                    # Compare in the target's current frame of reference so simple
                    # shared translation cannot manufacture a crossing path.
                    start = tuple(old_point[axis] + target.center[axis] - old_target.center[axis] for axis in (0, 1))
                    if (hypot(point[0]-start[0], point[1]-start[1]) <= actor.scale
                            and any(segment_intersects_box(start, point, region) for region in regions)):
                        return True
                return False
            hand_contact = reaches(a,b,(9,10)) or reaches(b,a,(9,10))
            foot_contact = reaches(a,b,(15,16)) or reaches(b,a,(15,16))
            kick_speed = max([p.limb_speeds.get(i,0) for p in (a,b) if p.limb_speeds is not None for i in (15,16)] or [0])
            if a.limb_speeds is not None and b.limb_speeds is not None:
                speed = max([actor.limb_speeds.get(i,0) for actor,target in ((a,b),(b,a)) for i in (9,10) if reaches(actor,target,(i,))] or [0])
                kick_speed = max([actor.limb_speeds.get(i,0) for actor,target in ((a,b),(b,a)) for i in (15,16) if reaches(actor,target,(i,))] or [0])
                limb = min(1,speed/r.wrist_speed)
            # Frame-normalized box flow can be tiny when only a fist moves. The
            # stabilizer already measured and validated flow around that exact limb.
            contact_motion = max([
                actor.limb_motion.get(index, 0)
                for actor, target in ((a, b), (b, a))
                if actor.limb_motion is not None and actor.limb_speeds is not None
                for index in (9, 10, 15, 16)
                if actor.limb_speeds.get(index, 0) >= r.wrist_speed * .7
                and reaches(actor, target, (index,))
            ] or [0])
            motion = min(1, max(motion, contact_motion / r.contact_motion_speed))
            pattern = "Striking arm movement" if hand_contact else "Close interaction"
            if foot_contact and kick_speed>speed:
                speed, limb, pattern = kick_speed,min(1,kick_speed/r.wrist_speed),"Fast kick toward another person's body"
            # Contact plus image-supported body displacement can also indicate a shove.
            # Opposing torso velocities distinguish this from people walking together.
            def torso_velocity(person):
                old = previous_people.get(person.track_id)
                if not valid_time or old is None or person.local_flow is None or person.local_flow < r.flow_speed*.5:
                    return (0,0)
                if any(min(person.keypoints[i][2],old.keypoints[i][2])<.55 for i in (5,6,11,12)):
                    return (0,0)
                return tuple(sum(person.keypoints[i][axis]-old.keypoints[i][axis] for i in (5,6,11,12))/(4*dt*scale) for axis in (0,1))
            va,vb=torso_velocity(a),torso_velocity(b)
            relative_body_speed=hypot(va[0]-vb[0],va[1]-vb[1])
            shove = hand_contact and speed>=r.wrist_speed*.35 and relative_body_speed>=.7 and motion>=1
            clinch = hand_contact and distance < .5 and all(
                (actor.local_flow or 0) >= r.flow_speed and any(
                    reaches(actor, target, (index,)) and
                    (actor.limb_speeds or {}).get(index, 0) >= r.wrist_speed*.4
                    for index in (9, 10)) for actor, target in ((a, b), (b, a)))
            if shove or clinch:
                limb=max(limb,.85)
                pattern="Contact with rapid body displacement" if shove else "Sustained close-contact struggle motion"
            score = 0.25 * proximity + 0.45 * limb + 0.30 * motion if near and depth_ok else 0.0
            reasons = []
            if near:
                reasons.append("Two tracked people are close in the image")
            if speed >= r.wrist_speed or shove or clinch:
                reasons.append(pattern)
            if pair_flow >= r.flow_speed:
                reasons.append("High local motion around people")
            if contact_motion >= r.contact_motion_speed * .5:
                reasons.append("Image motion supports the same striking limb")
            if not depth_ok:
                reasons.append("Relative depth disagrees with proximity")
            # Require distinct evidence, sustained on the same tracked pair.
            candidate = valid_time and score >= r.threshold and (speed >= r.wrist_speed * .7 or shove or clinch) and motion >= .5 and (hand_contact or foot_contact) and a.pose_reliable and b.pose_reliable
            slow_signals = self._slow_punch_evidence(a, b, previous_people, timestamp, reaches, proximity,
                                                     valid_time and depth_ok, dt)
            cycles = self.strike_cycles.setdefault(pair, deque())
            while cycles and timestamp-cycles[0] > 4:
                cycles.popleft()
            # Count observed extension/withdrawal cycles, not every fast frame.
            # Distances are relative to the other torso, removing shared translation.
            for actor, target in ((a, b), (b, a)):
                torso = tuple(sum(target.keypoints[i][axis] for i in (5, 6, 11, 12))/4 for axis in (0, 1))
                for index in (9, 10, 15, 16):
                    key = (actor.track_id, target.track_id, index)
                    if not candidate or not reaches(actor, target, (index,)) or (actor.limb_speeds or {}).get(index, 0) < r.wrist_speed*.7:
                        continue
                    point = actor.keypoints[index]
                    radius = hypot(point[0]-torso[0], point[1]-torso[1])/target.scale
                    phase = self.limb_phases.setdefault(key, {'low': radius, 'high': radius, 'out': False, 'time': timestamp})
                    phase['time'] = timestamp
                    if not phase['out']:
                        phase['low'] = min(phase['low'], radius)
                        if radius-phase['low'] >= .09:
                            cycles.append(timestamp)
                            phase.update(out=True, high=radius)
                    else:
                        phase['high'] = max(phase['high'], radius)
                        if phase['high']-radius >= .09:
                            phase.update(out=False, low=radius)
            history = self.strike_history.setdefault(pair, deque())
            while history and timestamp-history[0][0]>max(r.strike_window_seconds,r.hold_seconds*2):
                history.popleft()
            if not near or not depth_ok:
                history.clear()
            if candidate and valid_time:
                history.append((timestamp, min(dt, .25)))
            supported_seconds = sum(sample[1] for sample in history)
            bursts = sum(i==0 or sample[0]-history[i-1][0]>r.signal_grace_seconds
                         for i,sample in enumerate(history))
            repeated = (len(history)>=4 and bursts>=2 and
                        history[-1][0]-history[0][0]>=r.hold_seconds and
                        supported_seconds>=max(.4,r.hold_seconds*r.strike_evidence_fraction))
            if candidate:
                active_pairs.add(pair)
                self.pending.setdefault(pair, timestamp)
                self.last_positive[pair] = timestamp
            elif near and depth_ok and timestamp-self.last_positive.get(pair, -1e10) <= r.signal_grace_seconds:
                # A punching wrist briefly slows at each reversal. Bridge only short gaps.
                active_pairs.add(pair)
            held = timestamp - self.pending.get(pair, timestamp)
            sustained = held >= r.hold_seconds or repeated
            trigger = candidate and sustained and pair not in self.cooldowns and pair not in self.episodes
            if trigger:
                self.cooldowns[pair] = timestamp
                self.episodes.add(pair)
            state = "possible_fight" if candidate and sustained else "evaluating" if candidate else "observing"
            blockers = []
            if not valid_time: blockers.append("Motion history is starting or the frame gap is too long")
            if not (a.pose_reliable and b.pose_reliable): blockers.append("Both people need a reliable upper-body pose")
            if not near: blockers.append("The tracked people are too far apart in the image")
            if not depth_ok: blockers.append("Same-frame relative depth disagrees with proximity")
            if motion < .5: blockers.append("Not enough image motion to support limb movement")
            if not (hand_contact or foot_contact): blockers.append("No limb reaches the other person's body region")
            if speed < r.wrist_speed*.7 and not (shove or clinch): blockers.append("No sufficiently fast supported strike")
            if score < r.threshold: blockers.append(f"Interaction score {score:.2f} is below the {r.threshold:.2f} threshold")
            if candidate and not sustained: blockers.append("Collecting repeated strikes or sustained interaction")
            if repeated: reasons.append("Repeated image-supported strike bursts within the temporal window")
            assessment = Assessment(round(score, 3), state, reasons, {
                    **best.signals, "proximity": round(proximity, 3), "wrist_speed": round(speed, 3),
                    "held_seconds": round(held, 2), "depth_available": a.depth is not None and b.depth is not None,
                    "pattern": pattern, "pair_flow": round(pair_flow,3), "relative_body_speed": round(relative_body_speed,3),
                    "strike_bursts": bursts, "supported_seconds": round(supported_seconds,2),
                    "strike_cycles": len(cycles), "strike_cycle_times": list(cycles), "grappling": bool(clinch),
                    "contact_motion": round(contact_motion, 3),
                    "body_contact": bool(hand_contact or foot_contact),
                    "candidate": bool(candidate),
                    **slow_signals,
                    "blockers": blockers,
                }, pair, trigger)
            candidates.append(replace(assessment, events=[], candidates=[]))
            if trigger: events.append(replace(assessment, events=[]))
            if state=="possible_fight": active_fights.append(pair)
            if best.pair is None or (state=="possible_fight", score) > (best.state=="possible_fight", best.score) or trigger:
                best=assessment
        best.events=events
        best.candidates=candidates
        best.signals["active_fight_pairs"]=active_fights
        self.pending = {k: v for k, v in self.pending.items() if k in active_pairs}
        self.last_positive = {k: v for k, v in self.last_positive.items() if k in active_pairs}
        self.strike_history = {pair:history for pair,history in self.strike_history.items() if pair in visible_pairs and history}
        self.episodes.intersection_update(active_pairs | self.strike_history.keys())
        self.strike_cycles = {pair: history for pair, history in self.strike_cycles.items() if pair in visible_pairs and history}
        self.limb_phases = {key: phase for key, phase in self.limb_phases.items()
                            if tuple(sorted(key[:2])) in visible_pairs and timestamp-phase['time'] <= .85}
        self.slow_punch_phases = {key: phase for key, phase in self.slow_punch_phases.items()
                                 if tuple(sorted(key[:2])) in visible_pairs and timestamp-phase['last_support'] <= 1.5}
        self.slow_punch_cycles = {pair: cycles for pair, cycles in self.slow_punch_cycles.items()
                                 if pair in visible_pairs and cycles}
        self.slow_punch_strikes = {pair: strikes for pair, strikes in self.slow_punch_strikes.items()
                                  if pair in visible_pairs and strikes}
        return best
