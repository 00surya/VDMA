import threading
import time
import uuid
from dataclasses import replace
from pathlib import Path

import cv2

from .capture import Capture, validate_source
from .depth_worker import DepthWorker
from .eco import EcoGate
from .objects import ObjectWorker, WeaponIncidents
from .unattended import UnattendedObjects
from .heuristics import Rules
from .behavior import BehaviorHeuristic, EVENT_LABELS
from .stabilization import PoseStabilizer
from .storage import EvidenceBuffer
from .spatial import fight_depth_evidence
from .vision import Motion, PoseModel, annotate, annotate_status, choose_device, depth_view, synthetic_frame


class Engine:
    def __init__(self, store, model_dir="models", camera_id="camera-1", name="Camera 1"):
        self.camera_id, self.name = camera_id, name
        self.store, self.model_dir = store, Path(model_dir)
        self.lock = threading.Lock()
        self.lifecycle = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None
        self.capture = None
        self.depth_worker = None
        self.object_worker = None
        self.settings = None
        self.auto_reconnect = False
        self.reconnect_at = 0
        self.reconnect_attempts = 0
        self.scenario = "calm"
        self.frames = {}
        self.preview_source = None
        self.clean_preview = None
        self.preview_lock = threading.Lock()
        self.ai_buffer = EvidenceBuffer(seconds=30, max_bytes=8*1024*1024)
        self.ai_session = uuid.uuid4().hex
        self.state = {"status": "idle", "mode": None, "message": "Connect a source or explore the synthetic demo", "people": 0, "score": 0, "fps": 0, "latency_ms": 0, "reasons": [], "signals": {}, "sequence": 0}

    def snapshot(self):
        with self.lock:
            state = dict(self.state)
        state.update(camera_id=self.camera_id, name=self.name, storage_error=self.store.error, dropped_records=self.store.dropped)
        state.update(auto_reconnect=self.auto_reconnect, reconnect_attempts=self.reconnect_attempts,
                     retry_in_seconds=max(0, round(self.reconnect_at-time.monotonic())))
        state['live_camera'] = False
        if self.settings:
            if self.settings.mode == 'live':
                try:
                    state['live_camera'] = not validate_source(self.settings.source)[1]
                except ValueError:
                    pass
            state['settings'] = {key: getattr(self.settings, key) for key in
                ('device', 'depth', 'depth_fps', 'detection_mode', 'eco_mode', 'object_detection',
                 'object_fps', 'threshold', 'hold_seconds', 'fight_confirmation_seconds', 'target_fps') if hasattr(self.settings, key)}
            state['settings'].update({key: getattr(self.settings, key) for key in
                ('unattended_objects', 'unattended_seconds', 'person_down_seconds', 'snatching_vehicles') if hasattr(self.settings, key)})
            location = getattr(self.settings, 'location', None)
            state['location'] = location.model_dump() if location else None
        age = time.time() - state.get("last_frame_at", time.time())
        state["frame_age_seconds"] = round(age, 1)
        state["stale"] = state["status"] == "running" and age > 3
        state["depth_meta"] = self.depth_metadata(state)
        state["object_meta"] = self.object_metadata(state)
        if state["object_meta"]["stale"] or state["object_meta"]["status"] != "ready":
            state["scene_objects"] = []
            if (state.get('alert') or {}).get('event_type') in {'knife_detected', 'gun_detected', 'unattended_object'}:
                state['alert'] = None
            state['unattended_meta'] = {'threshold_seconds': state.get('unattended_seconds', 60), 'tracks': []}
        state["depth_available"] = bool(state.get("depth_available") and not state["depth_meta"]["stale"] and state["depth_meta"]["status"] != "error")
        state["detection_warning"] = (
            "Depth-confirmed fight alerts are waiting for fresh depth. Pose and posture checks continue."
            if state.get("detection_mode") == "depth_confirmed" and state["status"] == "running"
            and (state["depth_meta"]["status"] not in {"ready", "simulated"} or not state["depth_meta"]["confirmation_fresh"])
            else None)
        if self.capture and getattr(self.capture, 'file', False) and hasattr(self.capture, 'metadata'):
            state['recording'] = self.capture.metadata()
        return state

    def preview_frames(self, overlays=True):
        """Display-only variant: never change inference, evidence or settings."""
        with self.preview_lock:
            with self.lock:
                values, state = dict(self.frames), dict(self.state)
                source, cached = self.preview_source, self.clean_preview
            if overlays:
                return values, state
            values['objects'] = values.get('objects_clean')
            values['pose'] = None
            if source is None or source[0] != state.get('sequence'):
                return values, state
            sequence, frame, people, alert, checking_text = source
            if cached is None or cached[0] != sequence:
                image = annotate_status(annotate(frame, people, overlays=False), alert, checking_text)
                ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 78])
                if not ok:
                    return values, state
                cached = (sequence, encoded.tobytes())
                with self.lock:
                    if self.preview_source is source:
                        self.clean_preview = cached
            values['pose'] = cached[1]
            return values, state

    @staticmethod
    def object_metadata(state):
        submitted = state.get("object_submitted_at")
        age = max(0, time.time()-submitted) if submitted is not None else None
        if age is not None:
            age = max(age, state.get("source_time", 0)-state.get("object_source_time", 0))
        rate = min(state.get("object_fps", 1), .5) if state.get("eco_state") == "quiet" else state.get("object_fps", 1)
        return {"status": state.get("object_status", "off"), "sequence": state.get("object_sequence"),
                "source_time": state.get("object_source_time"), "age_seconds": round(age, 1) if age is not None else None,
                "stale": age is not None and age > max(3, 2/rate), "error": state.get("object_error"),
                "latency_ms": state.get("object_latency_ms"), "target_fps": rate, "context_only": False,
                "knife_status": state.get("knife_status", "off") if state.get("object_status") == "ready" else state.get("object_status", "off"),
                "knife_error": state.get("knife_error")}

    @staticmethod
    def depth_metadata(state):
        submitted = state.get("depth_submitted_at")
        age = max(0, time.time()-submitted) if submitted is not None else None
        if age is not None and state.get("depth_source_time") is not None and state.get("source_time") is not None:
            age = max(age, state["source_time"]-state["depth_source_time"] + max(0, time.time()-state.get("last_frame_at", time.time())))
        rate = min(state.get("depth_fps", .5), .1) if state.get("eco_state") == "quiet" else state.get("depth_fps", .5)
        limit = max(3, 2/rate)
        return {"status": state.get("depth_status", "off"), "sequence": state.get("depth_sequence"),
                "source_time": state.get("depth_source_time"), "age_seconds": round(age, 1) if age is not None else None,
                "stale": age is not None and age > limit, "latency_ms": state.get("depth_latency_ms"),
                "error": state.get("depth_error"), "target_fps": rate,
                "confirmation_fresh": age is not None and age <= 2.5,
                "advisory": state.get("detection_mode") != "depth_confirmed",
                "fight_confirmation_support": state.get("detection_mode") == "depth_confirmed"}

    @staticmethod
    def matched_depth_evidence(sample, pose_frames):
        """Only compare raw poses with depth from their exact captured frame."""
        cached = pose_frames.get(sample.get("sequence"))
        depth = sample.get("depth")
        if (cached is None or depth is None or sample.get("source_time") != cached["time"]
                or depth.shape != cached["shape"]):
            return []
        return fight_depth_evidence(depth, cached["people"])

    @staticmethod
    def matched_object_context(sample, object_frames, source_time, rate, status='ready'):
        """A late object result may only use motion from its exact captured frame."""
        if status != 'ready' or not sample:
            return None
        cached = object_frames.get(sample.get('sequence'))
        limit = max(3, 2/rate)
        if (cached is None or sample.get('source_time') != cached['time']
                or tuple(sample.get('frame_shape', ())) != cached['shape']
                or not 0 <= source_time-sample['source_time'] <= limit
                or not 0 <= time.time()-sample.get('submitted_at', 0) <= limit):
            return None
        return cached

    def ai_frames(self, since):
        with self.lock:
            return [(stamp, jpeg) for stamp, jpeg in self.ai_buffer.snapshot() if stamp > since][-24:]

    def start(self, settings):
        with self.lifecycle:
            if self.thread and self.thread.is_alive():
                raise RuntimeError("Stop the current session before connecting another source")
            self.settings = settings
            self.stop_event = threading.Event()
            self.capture = None
            self.depth_worker = None
            self.object_worker = None
            self.frames = {}
            self.preview_source, self.clean_preview = None, None
            with self.lock:
                self.ai_buffer = EvidenceBuffer(seconds=30, max_bytes=8*1024*1024)
                self.ai_session = uuid.uuid4().hex
            self.scenario = "calm"
            with self.lock:
                self.state = {"status": "starting", "mode": settings.mode, "message": "Loading pipeline", "people": 0, "score": 0, "fps": 0, "latency_ms": 0, "sequence": 0, "reasons": [], "signals": {}, "depth": settings.depth, "device": settings.device, "depth_fps": settings.depth_fps, "depth_status": "off" if settings.depth == "off" else "loading"}
                self.state["detection_mode"] = getattr(settings, "detection_mode", "responsive")
                self.state.update(eco_mode=bool(getattr(settings, "eco_mode", False)), eco_state="starting" if getattr(settings, "eco_mode", False) else "off", eco_skipped_frames=0, processed_frames=0, pose_inference_ms_total=0, depth_submissions=0)
                self.state.update(object_detection=bool(getattr(settings, "object_detection", False)), object_status="loading" if getattr(settings, "object_detection", False) or getattr(settings, 'unattended_objects', False) or getattr(settings, 'snatching_vehicles', False) else "off", object_fps=getattr(settings, "object_fps", 1), scene_objects=[], object_submissions=0)
                self.state.update(knife_status="loading" if getattr(settings, "object_detection", False) else "off", knife_error=None)
                self.state.update(unattended_objects=bool(getattr(settings, 'unattended_objects', False)),
                                  unattended_seconds=getattr(settings, 'unattended_seconds', 60),
                                  unattended_meta={'threshold_seconds': getattr(settings, 'unattended_seconds', 60), 'tracks': []})
            self.thread = threading.Thread(target=self.run, args=(settings,), daemon=True, name="vmd-inference")
            self.thread.start()

    def set_eco_mode(self, enabled):
        with self.lock:
            if self.settings is None or self.settings.mode != "live":
                raise RuntimeError("Eco mode is available for camera and recorded-video inputs")
            self.settings.eco_mode = enabled
            self.state.update(eco_mode=enabled, eco_state="starting" if enabled else "off")

    def stop(self):
        with self.lifecycle:
            self.stop_event.set()
            if self.capture:
                self.capture.stop_event.set()
            if self.thread:
                self.thread.join(timeout=7)
            if self.thread and self.thread.is_alive():
                with self.lock:
                    self.state.update(status="stopping", message="Waiting for the current model operation to finish")
            else:
                with self.lock:
                    self.frames = {}
                    self.preview_source, self.clean_preview = None, None
                    self.ai_buffer = EvidenceBuffer(seconds=30, max_bytes=8*1024*1024)
                    self.ai_session = uuid.uuid4().hex
                    self.state.update(status="idle", mode=None, message="Session stopped", fps=0, sequence=0, people=0, score=0, depth_available=False, buffer_seconds=0, signals={}, reasons=[], alert=None, depth_status="off", depth_sequence=None, depth_submitted_at=None, depth_error=None)
                    self.state.update(object_status="off", object_error=None, knife_status="off", knife_error=None, scene_objects=[], object_submitted_at=None, eco_state="off")
                    self.state['unattended_meta'] = {'threshold_seconds': self.state.get('unattended_seconds', 60), 'tracks': []}

    def run(self, settings):
        buffer = EvidenceBuffer()
        confirmed = getattr(settings, "detection_mode", "responsive") == "depth_confirmed"
        # Every user-visible fight passes the depth gate. Legacy responsive
        # sessions retain review warnings only, even if motion is very strong.
        heuristic = BehaviorHeuristic(Rules(threshold=settings.threshold, hold_seconds=settings.hold_seconds,
            fight_confirmation_seconds=getattr(settings, 'fight_confirmation_seconds', 3.0)), confirmation=True,
            person_down_seconds=getattr(settings, 'person_down_seconds', 3.0),
            snatching_vehicles=bool(getattr(settings, 'snatching_vehicles', False)))
        rider = heuristic.vehicle_snatching
        motion = Motion()
        stabilizer = PoseStabilizer()
        alert, alert_until = None, 0
        seq, previous_source, last_telemetry = 0, None, 0
        pose_frames = {}
        last_depth_sequence = None
        file_depth_time = float('-inf')
        eco_gate = None
        weapon_gate = WeaponIncidents()
        unattended = (UnattendedObjects(getattr(settings, 'unattended_seconds', 60),
                         max_gap=max(3, 2/getattr(settings, 'object_fps', 1)))
                      if getattr(settings, 'unattended_objects', False) else None)
        object_frames, last_object_sequence = {}, None
        object_file_error = None
        file_object_time = float('-inf')
        unattended_alert, unattended_until = None, 0
        skipped, processed, depth_submissions, object_submissions, pose_ms = 0, 0, 0, 0, 0.0
        started = time.monotonic()
        pose = None
        try:
            if settings.mode == "live":
                device = choose_device(settings.device)
                pose = PoseModel(str(self.model_dir / "yolo11n-pose.pt"), device)
                with self.lock:
                    self.state["device"] = device
                if self.stop_event.is_set():
                    return
                self.capture = Capture(settings.source)
                self.capture.target_fps = settings.target_fps
                self.capture.start()
                if settings.depth != "off":
                    try:
                        self.depth_worker = DepthWorker(settings.depth, device, self.model_dir, fps=settings.depth_fps)
                        self.depth_worker.start()
                    except Exception as exc:
                        if self.depth_worker:
                            self.depth_worker.stop()
                        self.depth_worker = None
                        with self.lock:
                            self.state.update(depth_status="error", depth_error=f"Could not start depth worker ({type(exc).__name__})")
                if getattr(settings, "object_detection", False) or unattended or rider:
                    try:
                        context_options = {}
                        if unattended or rider:
                            context_options.update(unattended=bool(unattended),
                                weapons=bool(getattr(settings, 'object_detection', False)))
                        if rider:
                            context_options['riders'] = True
                        self.object_worker = ObjectWorker(self.model_dir, fps=settings.object_fps, **context_options)
                        self.object_worker.start()
                    except Exception as exc:
                        if self.object_worker:
                            self.object_worker.stop()
                        self.object_worker = None
                        with self.lock:
                            self.state.update(object_status="error", object_error=f"Could not start object worker ({type(exc).__name__})")
            while not self.stop_event.is_set():
                cycle = time.monotonic()
                pose_elapsed_ms = 0.0
                depth_jpeg = None
                depth_state = {}
                object_state, object_jpeg, object_clean_jpeg = {}, None, None
                weapon_alert, weapon_events, weapon_sample = None, [], None
                unattended_events, unattended_sample = [], None
                if settings.mode == "demo":
                    timestamp = cycle - started
                    frame, people, depth, flow, camera = synthetic_frame(timestamp, self.scenario)
                    seq += 1
                    ok, encoded_depth = cv2.imencode(".jpg", depth_view(depth))
                    if ok:
                        depth_jpeg = encoded_depth.tobytes()
                    depth_state = dict(depth_status="simulated", depth_sequence=seq, depth_source_time=timestamp, depth_submitted_at=time.time(), depth_latency_ms=0)
                else:
                    packet = self.capture.latest(seq)
                    if packet is None:
                        if self.capture.finished:
                            if self.capture.error:
                                raise RuntimeError(self.capture.error)
                            with self.lock:
                                self.state.update(status="finished", message="Recording analysis complete", fps=0)
                            return
                        self.stop_event.wait(.01)
                        continue
                    seq, timestamp, frame = packet
                    # Recorded analysis must inspect its selected frames rather
                    # than skip subtle actions while the viewer is idle.
                    enabled = bool(getattr(settings, "eco_mode", False)) and not getattr(self.capture, 'file', False)
                    if enabled and eco_gate is None:
                        eco_gate = EcoGate()
                    elif not enabled:
                        eco_gate = None
                    if eco_gate:
                        eco_gate.observe(frame, timestamp)
                        if not eco_gate.should_run("pose", timestamp) and settings.target_fps > 2:
                            skipped += 1
                            with self.lock:
                                self.state.update(eco_skipped_frames=skipped, **eco_gate.snapshot())
                            self.stop_event.wait(min(.05, 1/settings.target_fps))
                            continue
                    pose_started = time.perf_counter()
                    people = pose.infer(frame)
                    pose_elapsed_ms = (time.perf_counter()-pose_started)*1000
                    pose_ms += pose_elapsed_ms
                    motion_started = time.perf_counter()
                    flow, camera = motion.infer(frame, people, timestamp)
                raw_people = people
                if settings.mode == "live":
                    for person in people:
                        person.local_flow = motion.person_flow(person)
                    people = stabilizer.update(people, timestamp, motion.joint_motion)
                else:
                    # Synthetic motion evidence exercises the workflow, not the model.
                    for person in people:
                        person.local_flow = flow
                    people = stabilizer.update(people, timestamp, lambda p, i: 2 if self.scenario == "interaction" else 0)
                motion_elapsed_ms = (time.perf_counter()-motion_started)*1000 if settings.mode == 'live' else 0.0
                file_depth = bool(self.depth_worker and getattr(self.capture, 'file', False)
                                  and hasattr(self.depth_worker, 'analyze_file_frame'))
                def cache_depth_pose():
                    pose_frames[seq] = {'time': timestamp, 'shape': frame.shape[:2],
                        'people': [replace(raw, keypoints=list(raw.keypoints), pose_reliable=stable.pose_reliable)
                                   for raw, stable in zip(raw_people, people)]}
                    while len(pose_frames) > 32:
                        pose_frames.pop(next(iter(pose_frames)))
                if self.depth_worker:
                    if file_depth and timestamp-file_depth_time >= 1/settings.depth_fps-1e-9:
                        cache_depth_pose()
                        sampled = self.depth_worker.analyze_file_frame(seq, timestamp, frame, self.stop_event)
                        if sampled is None:
                            return
                        depth_submissions += 1
                        file_depth_time = timestamp
                    status, sampled, error = self.depth_worker.poll()
                    depth_state = dict(depth_status=status, depth_error=error)
                    if status == "error":
                        heuristic.depth_unavailable()
                    if sampled:
                        depth_jpeg = sampled["jpeg"]
                        depth_state.update(depth_sequence=sampled["sequence"], depth_source_time=sampled["source_time"], depth_submitted_at=sampled["submitted_at"], depth_latency_ms=sampled["latency_ms"])
                        if sampled["sequence"] != last_depth_sequence:
                            observations = self.matched_depth_evidence(sampled, pose_frames)
                            heuristic.observe_depth(sampled["source_time"], observations, timestamp, confirm_fight=confirmed)
                            last_depth_sequence = sampled["sequence"]
                    if not file_depth and (not eco_gate or eco_gate.should_run("depth", timestamp)) and self.depth_worker.submit(seq, timestamp, frame):
                        depth_submissions += 1
                        pose_frames[seq] = {"time": timestamp, "shape": frame.shape[:2],
                            "people": [replace(raw, keypoints=list(raw.keypoints), pose_reliable=stable.pose_reliable)
                                       for raw, stable in zip(raw_people, people)]}
                        while len(pose_frames) > 32:
                            pose_frames.pop(next(iter(pose_frames)))
                elif confirmed and settings.mode == "demo":
                    heuristic.observe_depth(timestamp, fight_depth_evidence(depth, raw_people), timestamp)
                if self.object_worker:
                    def cache_object_frame():
                        object_frames[seq] = {'time': timestamp, 'shape': frame.shape[:2], 'camera_motion': camera,
                            'people': [replace(raw, keypoints=list(raw.keypoints), pose_reliable=stable.pose_reliable)
                                       for raw, stable in zip(raw_people, people)]}
                        while len(object_frames) > 32:
                            object_frames.pop(next(iter(object_frames)))
                    # Source-time sampling/backpressure keeps an uploaded video
                    # from skipping the whole absence interval during fast decode.
                    file_objects = bool((unattended or rider) and getattr(self.capture, 'file', False)
                                        and hasattr(self.object_worker, 'analyze_file_frame'))
                    if file_objects and not object_file_error and timestamp-file_object_time >= 1/settings.object_fps-1e-9:
                        cache_object_frame()
                        try:
                            sampled = self.object_worker.analyze_file_frame(seq, timestamp, frame, self.stop_event)
                            if sampled is None:
                                return
                            object_submissions += 1
                            file_object_time = timestamp
                        except RuntimeError:
                            # This optional worker cannot take down pose/fight checks.
                            self.object_worker.stop()
                            if unattended:
                                unattended.reset()
                            if rider:
                                rider.reset()
                            object_file_error = 'Recording object analysis unavailable; pose and motion remain active'
                    status, sample, error = (('error', None, object_file_error) if object_file_error
                                             else self.object_worker.poll())
                    object_state.update(object_status=status, object_error=error)
                    if object_file_error:
                        object_state.update(knife_status='error' if getattr(settings, 'object_detection', False) else 'off',
                                            knife_error=None)
                    if sample:
                        object_jpeg = sample['jpeg']
                        object_clean_jpeg = sample.get('clean_jpeg')
                        object_state.update(scene_objects=sample['detections'] if getattr(settings, 'object_detection', False) else [], object_sequence=sample['sequence'],
                                            object_source_time=sample['source_time'], object_submitted_at=sample['submitted_at'],
                                            object_latency_ms=sample['latency_ms'],
                                            knife_status=sample.get('knife_status', 'off') if getattr(settings, 'object_detection', False) else 'off',
                                            knife_error=sample.get('knife_error') if getattr(settings, 'object_detection', False) else None)
                    weapon_rate = min(settings.object_fps, .5) if eco_gate and eco_gate.snapshot()['eco_state'] == 'quiet' else settings.object_fps
                    weapons, weapon_events = (weapon_gate.update(sample, timestamp, weapon_rate, status)
                                              if getattr(settings, 'object_detection', False) else ([], []))
                    if weapons:
                        strongest = max(weapons, key=lambda item: item['confidence'])
                        weapon_alert = {'event_type': strongest['label'] + '_detected',
                            'label': strongest['label'].title() + ' detected', 'score': strongest['confidence'],
                            'reasons': ['Visible weapon prediction; review the recorded evidence.']}
                    if weapon_events:
                        weapon_sample = sample
                    if unattended or rider:
                        context = self.matched_object_context(sample, object_frames, timestamp, weapon_rate, status)
                        if context is None or camera > heuristic.fight.rules.camera_speed:
                            if unattended:
                                unattended.reset()
                                unattended_alert = None
                            if rider:
                                rider.reset()
                        elif sample['sequence'] != last_object_sequence:
                            if rider:
                                rider.observe_vehicles(sample.get('context_objects'), context['people'], sample['source_time'],
                                    timestamp, sample['sequence'], context['camera_motion'], context['shape'])
                            if unattended:
                                unattended_events = unattended.update(sample.get('context_objects'), sample['source_time'],
                                    context['camera_motion'], context['shape'])
                            last_object_sequence = sample['sequence']
                            if unattended_events:
                                unattended_sample = sample
                                unattended_alert = {'event_type': 'unattended_object', 'label': 'UNATTENDED ITEM / REVIEW',
                                    'reasons': unattended_events[0].reasons, 'score': unattended_events[0].score}
                                unattended_until = timestamp + 5
                            elif unattended and not any(item['status'] == 'alerted' for item in unattended.snapshot()['tracks']):
                                unattended_alert = None
                    if not file_objects and (not eco_gate or eco_gate.should_run('objects', timestamp)) and self.object_worker.submit(seq, timestamp, frame):
                        object_submissions += 1
                        if unattended or rider:
                            cache_object_frame()
                if timestamp > unattended_until:
                    unattended_alert = None
                # Sampled depth supports pair history only; never attach it to a newer pose.
                for person in people:
                    person.depth = None
                rules_started = time.perf_counter()
                filtered_people = [p for p in people if p.track_id >= 0]
                result = (heuristic.update(filtered_people, timestamp, flow, camera,
                            raw_people=raw_people, frame_shape=frame.shape[:2]) if rider
                          else heuristic.update(filtered_people, timestamp, flow, camera))
                rules_elapsed_ms = (time.perf_counter()-rules_started)*1000
                priority_depth = (self.depth_worker and seq not in pose_frames
                    and (any(state['contact'] == timestamp for state in heuristic.snatching.pending.values())
                         or any(state.since == timestamp for state in heuristic.confirmation.pending.values())
                         or any(strike['contact'] == timestamp for strikes in heuristic.fight.slow_punch_strikes.values()
                                for strike in strikes)))
                if priority_depth:
                    if file_depth:
                        cache_depth_pose()
                        sampled = self.depth_worker.analyze_file_frame(seq, timestamp, frame, self.stop_event)
                        if sampled is None:
                            return
                        heuristic.observe_depth(timestamp, self.matched_depth_evidence(sampled, pose_frames),
                                                timestamp, confirm_fight=confirmed)
                        last_depth_sequence, file_depth_time = seq, timestamp
                        depth_submissions += 1
                    elif self.depth_worker.submit(seq, timestamp, frame, priority=True):
                        depth_submissions += 1
                        cache_depth_pose()
                if result.state in {"possible_fight", "fight_detected", "possible_snatching", "snatching_detected", "person_down", "possible_fall", "person_down_after_fight", "hands_up"} or result.events:
                    label = EVENT_LABELS[result.event_type]
                    alert = {"event_type": result.event_type, "label": label, "reasons": result.reasons, "pair": result.pair,
                             'vehicle_snatching': bool(result.signals.get('vehicle_context'))}
                    alert_until = timestamp + 5
                if (alert and alert['event_type'] in {'possible_fight', 'fight'}
                        and alert.get('pair') in result.signals.get('rejected_fight_pairs', [])):
                    alert = None
                if (alert and alert['event_type'] in {'possible_snatching', 'snatching_detected'}
                        and not alert.get('vehicle_snatching')
                        and heuristic.snatching.pending.get(alert.get('pair'), {}).get('depth_status')
                            in {'uncertain', 'separated'}):
                    alert = None
                if timestamp > alert_until or camera > heuristic.fight.rules.camera_speed:
                    alert = None
                if eco_gate and (alert or weapon_alert or unattended_alert or unattended_events or result.events or result.state in {"evaluating", "checking_interaction"}
                                 or heuristic.snatching.pending or (rider and rider.snapshot().get('candidates'))
                                 or heuristic.hands or any("prone_since" in track for track in heuristic.tracks.values())):
                    eco_gate.keep_active(timestamp)
                render_started = time.perf_counter()
                rendered = annotate(frame, people)
                checking_text = None
                if result.state == "checking_interaction" and not alert:
                    progress = result.signals.get("fight_timer_seconds", 0)
                    required = heuristic.confirmation.required_seconds
                    checking_text = f"CHECKING INTERACTION: {progress:.1f} / {required:g}s"
                display_alert = alert or unattended_alert
                annotate_status(rendered, display_alert, checking_text)
                ok, encoded = cv2.imencode(".jpg", rendered, [cv2.IMWRITE_JPEG_QUALITY, 78])
                if not ok:
                    raise RuntimeError("Could not encode the processed video frame")
                jpeg = encoded.tobytes()
                render_elapsed_ms = (time.perf_counter()-render_started)*1000
                buffer.append(timestamp, jpeg)
                now = time.time()
                location = getattr(settings, 'location', None)
                response_context = {
                    'live_camera': settings.mode == 'live' and self.capture is not None
                                   and not getattr(self.capture, 'file', True),
                    'location': location.model_dump() if location else None,
                }
                for item in weapon_events:
                    sample_time = weapon_sample['source_time']
                    evidence = {stamp: image for stamp, image in buffer.snapshot()
                                if sample_time-10 <= stamp <= timestamp}
                    # Async inference can finish after newer frames: retain that
                    # real footage so even a first-frame detection has a clip.
                    evidence[sample_time] = weapon_sample['jpeg']
                    frames = sorted(evidence.items())
                    self.store.enqueue('incident', ({'id': uuid.uuid4().hex, 'created': now,
                        'mode': settings.mode, 'camera_id': self.camera_id, 'camera_name': self.name,
                        'event_type': item['label'] + '_detected', 'score': item['confidence'],
                        'reasons': [f"{item['label'].title()} detected ({item['confidence']:.0%} model confidence); review required"],
                        'signals': {**response_context, 'object_detection': True, 'object_label': item['label'],
                            'box': item['box'], 'detector': item.get('detector'),
                            'source_seconds': sample_time, 'sequence': weapon_sample['sequence'],
                            'buffer_seconds': sample_time-frames[0][0]}}, frames))
                for event in result.events:
                    self.store.enqueue("incident", ({"id": event.signals.get('episode_id') or uuid.uuid4().hex, "created": now, "mode": settings.mode, "camera_id": self.camera_id, "camera_name": self.name, "event_type": event.event_type, "score": event.score, "reasons": event.reasons, "signals": {**event.signals, **response_context, "pair": event.pair, "source_seconds": timestamp, "buffer_seconds": round(timestamp-buffer.frames[0][0], 2)}}, buffer.snapshot()))
                for event in unattended_events:
                    sample_time = unattended_sample['source_time']
                    evidence = {stamp: image for stamp, image in buffer.snapshot() if sample_time-10 <= stamp <= timestamp}
                    evidence[sample_time] = unattended_sample['jpeg']
                    frames = sorted(evidence.items())
                    self.store.enqueue('incident', ({'id': event.signals['episode_id'], 'created': now,
                        'mode': settings.mode, 'camera_id': self.camera_id, 'camera_name': self.name,
                        'event_type': event.event_type, 'score': event.score, 'reasons': event.reasons,
                        'signals': {**event.signals, **response_context, 'source_seconds': sample_time,
                            'sequence': unattended_sample['sequence'], 'detector': 'YOLO26s + attendance rules',
                            'buffer_seconds': round(sample_time-frames[0][0], 2)}}, frames))
                if now-last_telemetry >= 1:
                    self.store.enqueue("telemetry", (now, len(people), result.score, settings.mode, self.camera_id, response_context['live_camera']))
                    last_telemetry = now
                elapsed = max(.001, time.monotonic()-cycle)
                fps = 1/(timestamp-previous_source) if previous_source is not None and timestamp>previous_source else 0
                previous_source = timestamp
                processed += 1
                with self.lock:
                    # Separate bounded, head-blurred aftermath buffer. Quiet Eco still
                    # processes at 2 FPS, sufficient for this 2 FPS description budget.
                    if settings.mode == 'live' and not getattr(self.capture, 'file', True):
                        if not self.ai_buffer.frames or now-self.ai_buffer.frames[-1][0] >= .5:
                            self.ai_buffer.append(now, jpeg)
                    self.frames = {"pose": jpeg, "depth": depth_jpeg or self.frames.get("depth"), "objects": object_jpeg or self.frames.get("objects"),
                                   'objects_clean': object_clean_jpeg if object_jpeg is not None else self.frames.get('objects_clean')}
                    self.preview_source = (seq, frame.copy(),
                        [replace(person, keypoints=list(person.keypoints)) for person in people],
                        dict(display_alert) if display_alert else None, checking_text)
                    self.clean_preview = None
                    self.state.update(status="running", message="Synthetic inputs" if settings.mode == "demo" else "Processing locally", people=len(people), score=result.score, assessment=result.state, alert=alert, reasons=result.reasons, signals=result.signals, fps=round(fps, 1), latency_ms=round(elapsed*1000), sequence=seq, source_time=timestamp, last_frame_at=now, buffer_seconds=round(timestamp-buffer.frames[0][0], 1), depth_available=self.frames.get("depth") is not None, scenario=self.scenario, sampling_limited=settings.mode == "live" and fps > 0 and fps < 1/heuristic.fight.rules.max_gap_seconds, **depth_state)
                    if getattr(self.capture, 'file', False):
                        self.state['message'] = 'Analyzing recording · selected frames are processed in order'
                    self.state.update(processed_frames=processed, pose_inference_ms_total=round(pose_ms, 1),
                                      stage_ms={key: round(value, 1) for key, value in {'pose': pose_elapsed_ms,
                                          'motion_and_stabilization': motion_elapsed_ms, 'rules': rules_elapsed_ms,
                                          'render_encode': render_elapsed_ms}.items()},
                                      depth_submissions=depth_submissions, object_submissions=object_submissions, eco_mode=bool(eco_gate), eco_skipped_frames=skipped, **object_state,
                                      **(eco_gate.snapshot() if eco_gate else {"eco_state": "off", "eco_motion_score": 0}))
                    if unattended:
                        self.state.update(unattended_objects=True, unattended_seconds=unattended.threshold_seconds,
                                          unattended_meta=unattended.snapshot())
                    if weapon_alert or unattended_alert:
                        self.state['alert'] = weapon_alert or alert or unattended_alert
                self.stop_event.wait(max(0, 1/settings.target_fps - elapsed))
        except Exception as exc:
            message = str(exc) if isinstance(exc, RuntimeError) else f"Pipeline failed ({type(exc).__name__}). Check installed vision dependencies and model files."
            with self.lock:
                self.state.update(status="error", message=message, fps=0)
        finally:
            if self.object_worker:
                self.object_worker.stop()
            if self.depth_worker:
                self.depth_worker.stop()
            if self.capture:
                self.capture.stop()
            if self.stop_event.is_set():
                with self.lock:
                    self.state.update(status="idle", message="Session stopped", fps=0)
