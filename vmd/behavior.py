"""Temporal posture events associated only with the same camera and track."""
from collections import Counter, deque
from dataclasses import replace
from math import hypot, isfinite
from statistics import median

from .heuristics import Assessment, FightHeuristic
from .confirmation import FightConfirmation
from .snatching import SnatchingHeuristic
from .vehicle_snatching import VehicleSnatchingHeuristic

EVENT_LABELS = {
    "fight": "FIGHT DETECTED",
    "possible_fight": "POSSIBLE FIGHT / REVIEW",
    "possible_snatching": "POSSIBLE SNATCHING / REVIEW",
    "snatching_detected": "SNATCHING DETECTED",
    "person_down": "PERSON DOWN",
    "possible_fall": "POSSIBLE FALL",
    "person_down_after_fight": "PERSON DOWN AFTER FIGHT SIGNAL",
    "hands_up": "HANDS UP / REVIEW",
}


def centroid_speed(history):
    """Robust pixel/second displacement over a short window, not one noisy box step."""
    if len(history) < 3 or history[-1][0] - history[0][0] < .4:
        return None
    count = max(1, len(history) // 3)
    first, last = list(history)[:count], list(history)[-count:]
    dt = median(p[0] for p in last) - median(p[0] for p in first)
    if dt <= 0:
        return None
    return hypot(*(median(p[axis] for p in last) - median(p[axis] for p in first)
                   for axis in (1, 2))) / dt


def down_posture_geometry(person):
    """Observed torso/leg geometry, without filling in hidden landmarks.

    One hidden hip is common in a side view. Both shoulders and one real hip
    still anchor that view; a compact or diagonal box needs a visible straight
    leg aligned with the torso, rather than just an outstretched arm.
    """
    points = person.keypoints
    visible = lambda index: points[index][2] >= .55 and all(isfinite(v) for v in points[index])
    if (not person.pose_reliable or not all(visible(i) for i in (5, 6))
            or not isfinite(person.scale) or person.scale <= 0
            or not all(isfinite(v) for v in person.box)):
        return None
    hips = [i for i in (11, 12) if visible(i)]
    if not hips:
        return None
    shoulder = tuple(sum(points[i][axis] for i in (5, 6))/2 for axis in (0, 1))
    hip = tuple(sum(points[i][axis] for i in hips)/len(hips) for axis in (0, 1))
    torso = tuple(hip[axis]-shoulder[axis] for axis in (0, 1))
    dx, dy = abs(torso[0]), abs(torso[1])
    width = person.box[2]-person.box[0]
    height = person.box[3]-person.box[1]
    if min(width, height) <= 0:
        return None
    horizontal_legs, vertical_legs = [], []
    for anchors in ((11, 13, 15), (12, 14, 16)):
        if not all(visible(i) for i in anchors):
            continue
        leg = [points[i] for i in anchors]
        # Off-box landmarks are not evidence for a cropped lower body.
        margin = person.scale*.05
        if any(not (person.box[0]-margin <= p[0] <= person.box[2]+margin
                    and person.box[1]-margin <= p[1] <= person.box[3]+margin) for p in leg):
            continue
        first = tuple(leg[1][axis]-leg[0][axis] for axis in (0, 1))
        second = tuple(leg[2][axis]-leg[1][axis] for axis in (0, 1))
        whole = tuple(leg[2][axis]-leg[0][axis] for axis in (0, 1))
        lengths = hypot(*first)*hypot(*second)
        straight = lengths > 0 and sum(a*b for a, b in zip(first, second))/lengths >= .6
        aligned = sum(a*b for a, b in zip(torso, whole)) > 0
        if (straight and aligned and abs(whole[0]) > abs(whole[1])*1.35
                and abs(whole[0]) > person.scale*.18):
            horizontal_legs.append(anchors[0])
        elif whole[1] > person.scale*.3 and abs(whole[0]) < whole[1]*.5:
            vertical_legs.append(anchors[0])
    upright = dy > dx*1.5 and dy > person.scale*.12
    strong_torso = dx > dy*1.5 and dx > person.scale*.12
    wide = width > height*1.2
    supported_crop = (bool(horizontal_legs) and width > height*.85
                      and dx > dy*1.25 and dx > person.scale*.12)
    prone = (strong_torso and wide or supported_crop) and (not vertical_legs or bool(horizontal_legs))
    return dict(shoulder=shoulder, hip=hip, upright=upright, prone=prone,
                evidence="horizontal_torso" if strong_torso and wide else "torso_and_leg",
                visible_hips=hips, horizontal_legs=horizontal_legs,
                body_aspect_ratio=width/height, span=max(width, height))


class BehaviorHeuristic:
    def __init__(self, rules=None, confirmation=False, person_down_seconds=3.0, snatching_vehicles=False):
        self.person_down_seconds = float(person_down_seconds)
        if not isfinite(self.person_down_seconds) or self.person_down_seconds <= 0:
            raise ValueError("Person-down duration must be a positive, finite number")
        self.fight = FightHeuristic(rules)
        self.confirmation = FightConfirmation(self.fight.rules) if confirmation else None
        self.snatching = SnatchingHeuristic(self.fight.rules)
        self.vehicle_snatching = VehicleSnatchingHeuristic(self.fight.rules) if snatching_vehicles else None
        self.episodes = {}
        self.tracks = {}
        self.hands = {}
        self.recent_fights = {}
        self.last_time = None

    def observe_depth(self, source_time, observations, current_time, confirm_fight=True):
        if self.confirmation:
            # Reaching arms can meet across different torso depth planes.
            # Snatching keeps its original torso/grab rule below.
            fight_observations = [dict(item, torso_status=item['status'],
                                       status=item.get('fight_status', item['status']))
                                  for item in observations]
            # Review-only mode still rejects known separation, but cannot build
            # the compatible-depth history needed to confirm a fight.
            if not confirm_fight:
                fight_observations = [item for item in fight_observations if item.get('status') != 'compatible']
            previous_depth_time = self.confirmation.last_depth_time
            self.confirmation.observe_depth(source_time, fight_observations, current_time)
            if self.confirmation.last_depth_time != previous_depth_time:
                for item in fight_observations:
                    if item.get('status') == 'separated':
                        self.fight.reset_slow_punch_pair(tuple(sorted(item['tracks'])))
        self.snatching.observe_depth(source_time, observations, current_time)

    def depth_unavailable(self):
        if self.confirmation:
            self.confirmation.depth_unavailable()
        self.snatching.depth_unavailable()

    def update(self, people, timestamp, local_flow=0, camera_motion=0, *, raw_people=None, frame_shape=None):
        result = self.fight.update(people, timestamp, local_flow, camera_motion)
        if self.confirmation:
            result = self.confirmation.update(result, people, timestamp, camera_motion)
        snatching = self.snatching.update(people, timestamp, camera_motion)
        vehicle_signals = (self.vehicle_snatching.update(raw_people or people, people, timestamp,
                            frame_shape=frame_shape, camera_motion=camera_motion)
                           if self.vehicle_snatching else [])
        pending_pairs = result.signals.get("pending_pairs", [])
        events = list(result.events)
        events.extend(signal for signal in snatching if signal.trigger)
        events.extend(signal for signal in vehicle_signals if signal.trigger)
        if self.last_time is not None and not 0 < timestamp - self.last_time <= self.fight.rules.max_gap_seconds:
            self.tracks.clear()
            self.hands.clear()
            self.recent_fights.clear()
            self.episodes.clear()
        self.last_time = timestamp
        if result.state == "camera_moving":
            self.tracks.clear()
            self.hands.clear()
            self.recent_fights.clear()
            self.episodes.clear()
            result.signals['snatching'] = self.snatching.snapshot()
            if self.vehicle_snatching:
                result.signals['vehicle_snatching'] = self.vehicle_snatching.snapshot()
            return result
        for pair in result.signals.get("active_fight_pairs", []):
            for track in pair:
                self.recent_fights[track] = timestamp
        for signal in [result, *events]:
            if signal.event_type == "fight" and signal.state in {"possible_fight", "fight_detected"} and signal.pair:
                for track in signal.pair:
                    self.recent_fights[track] = timestamp
        self.recent_fights = {key: time for key, time in self.recent_fights.items() if timestamp - time < 8}
        active = set()
        track_counts = Counter(person.track_id for person in people)
        current_postures = []
        for person in people:
            active.add(person.track_id)
            points = person.keypoints
            # COCO uses wrists 9/10 and nose 0; 15/16 are ankles.
            raised = (person.pose_reliable and all(points[i][2] >= .55 for i in (0, 9, 10))
                      and all(points[i][1] < points[0][1] - person.scale * .03 for i in (9, 10)))
            if raised:
                hands = self.hands.setdefault(person.track_id, {"since": timestamp, "reported": False})
                held = timestamp - hands["since"]
                if held >= 2:
                    signal = Assessment(.75, "hands_up", ["Both wrists stayed above the nose for at least 2 seconds",
                        "Hands-up posture alone does not establish robbery or surrender"],
                        {"track_id": person.track_id, "raised_seconds": round(held, 2)},
                        (person.track_id, person.track_id), not hands["reported"], "hands_up")
                    if signal.trigger:
                        events.append(signal)
                        hands["reported"] = True
                    current_postures.append(replace(signal, trigger=False))
            else:
                self.hands.pop(person.track_id, None)
            previous = self.tracks.get(person.track_id, {})
            geometry = (down_posture_geometry(person)
                        if person.track_id >= 0 and track_counts[person.track_id] == 1 else None)
            if not geometry:
                self.tracks.pop(person.track_id, None)
                continue
            shoulder, hip = geometry["shoulder"], geometry["hip"]
            if previous and (hypot(*(person.center[i]-previous["center"][i] for i in (0, 1)))
                             > max(person.scale, previous["scale"])*.65
                             or not .5 <= geometry["span"]/previous["span"] <= 2):
                # A recycled tracker ID cannot inherit the previous person's
                # down timer, upright transition or fight aftermath context.
                previous = {}
                self.recent_fights.pop(person.track_id, None)
            state = dict(previous)
            state.update(center=person.center, scale=person.scale, span=geometry["span"])
            centers = state.setdefault("centers", deque())
            centers.append((timestamp, *person.center))
            while centers and timestamp - centers[0][0] > 1:
                centers.popleft()
            speed = centroid_speed(centers)
            torsos = state.setdefault("torsos", deque())
            torsos.append((timestamp, *((shoulder[i]+hip[i])/2 for i in (0, 1))))
            while torsos and timestamp - torsos[0][0] > 1:
                torsos.popleft()
            torso_speed = centroid_speed(torsos)
            if geometry["upright"]:
                reference = state.get("upright")
                if not reference or timestamp - reference[0] > 1:
                    state["upright"] = (timestamp, shoulder[1], person.scale)
                for key in ("prone_since", "stationary_since", "reported", "assessment", "fall"):
                    state.pop(key, None)
            elif geometry["prone"]:
                state.setdefault("prone_since", timestamp)
                reference = state.get("upright")
                if reference and 0 < timestamp - reference[0] <= 2 and (shoulder[1] - reference[1]) / reference[2] > .25:
                    state["fall"] = True
                stationary = speed is not None and torso_speed is not None and max(speed, torso_speed) < 5
                if stationary:
                    state.setdefault("stationary_since", timestamp)
                else:
                    state.pop("stationary_since", None)
                still = timestamp - state.get("stationary_since", timestamp)
                if still >= self.person_down_seconds and not state.get("reported"):
                    after_fight = person.track_id in self.recent_fights and state.get("fall", False)
                    event_type = "person_down_after_fight" if after_fight else "possible_fall" if state.get("fall") else "person_down"
                    reasons = ["Horizontal torso and body aspect ratio above 1.2" if geometry["evidence"] == "horizontal_torso"
                               else "Horizontal torso corroborated by a visible straight, aligned leg",
                               f"Box and torso movement below 5 pixels/second for at least {self.person_down_seconds:g} seconds",
                               "Down posture requires operator review; it does not establish injury or cause"]
                    if state.get("fall"):
                        reasons.append("Rapid downward transition from a recently upright posture")
                    if after_fight:
                        reasons.append("The same track was involved in a recent fight signal; causation is unconfirmed")
                    signal = Assessment(.8, event_type, reasons,
                        {"track_id": person.track_id, "prone_seconds": round(timestamp - state["prone_since"], 2),
                         "stationary_seconds": round(still, 2), "centroid_speed_px_s": round(speed, 2)},
                        (person.track_id, person.track_id), True, event_type)
                    events.append(signal)
                    signal.signals.update(torso_speed_px_s=round(torso_speed, 2),
                                          required_stationary_seconds=self.person_down_seconds,
                                          posture_evidence=geometry["evidence"],
                                          visible_hips=geometry["visible_hips"],
                                          horizontal_legs=geometry["horizontal_legs"],
                                          body_aspect_ratio=round(geometry["body_aspect_ratio"], 2))
                    state["reported"] = event_type
                    state["assessment"] = replace(signal, trigger=False, events=[])
                if state.get("assessment") and stationary:
                    current_postures.append(state["assessment"])
            else:
                for key in ("prone_since", "stationary_since", "reported", "assessment", "fall"):
                    state.pop(key, None)
                if timestamp - state.get("upright", (-100, 0, 1))[0] > 2:
                    state.pop("upright", None)
            self.tracks[person.track_id] = state
        self.tracks = {key: value for key, value in self.tracks.items() if key in active}
        self.hands = {key: value for key, value in self.hands.items() if key in active}
        priority = {"hands_up": 0, "possible_fight": 1, "possible_snatching": 2,
                    "fight": 4, "snatching_detected": 5, "person_down": 2, "possible_fall": 2, "person_down_after_fight": 6}
        # A provisional warning and its escalation share one durable incident.
        active_pairs = set(self.snatching.pending)
        if self.confirmation:
            active_pairs.update(self.confirmation.pending)
        self.episodes = {pair: value for pair, value in self.episodes.items()
                         if pair in active_pairs or timestamp-value[1] < 2}
        for signal in [result, *snatching, *events]:
            # Rider checks are an independent review episode; they cannot
            # inherit a standing detector's confirmation/dispatch identity.
            if signal.signals.get('vehicle_context'):
                continue
            proposed = signal.signals.get('episode_id')
            if proposed:
                episode = self.episodes.setdefault(signal.pair, (proposed, timestamp))
                self.episodes[signal.pair] = (episode[0], timestamp)
                signal.signals = {**signal.signals, 'episode_id': episode[0]}
        choices = ([result] if result.state in {"possible_fight", "fight_detected"} else []) + current_postures + snatching + vehicle_signals
        if choices:
            choice = max(choices, key=lambda item: priority[item.event_type])
            result = replace(choice, trigger=any(event.event_type == choice.event_type and event.pair == choice.pair for event in events))
        unique = {}
        for event in events:
            key = event.signals.get('episode_id') or (event.event_type, event.pair)
            if key not in unique or priority[event.event_type] > priority[unique[key].event_type]:
                unique[key] = event
        result.events = [replace(event, events=[]) for event in unique.values()]
        result.signals = {**result.signals, "pending_pairs": pending_pairs, 'snatching': self.snatching.snapshot()}
        if self.vehicle_snatching:
            result.signals['vehicle_snatching'] = self.vehicle_snatching.snapshot()
        return result
