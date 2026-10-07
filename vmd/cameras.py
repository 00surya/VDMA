"""Saved camera settings and bounded retries; manual stops stay stopped."""
import json
import os
import threading
import time
import uuid

from .capture import validate_source
from .engine import Engine


EDITABLE_SETTINGS = ('device', 'depth', 'depth_fps', 'detection_mode', 'eco_mode',
                     'object_detection', 'object_fps', 'threshold', 'hold_seconds',
                     'fight_confirmation_seconds', 'target_fps', 'unattended_objects',
                     'unattended_seconds', 'person_down_seconds', 'snatching_vehicles')


class CameraManager:
    def __init__(self, store, model_dir, settings_type):
        self.store, self.model_dir, self.settings_type = store, model_dir, settings_type
        self.path = store.directory / 'cameras.json'
        self.lock = threading.RLock()
        self.cameras, self.saved = {}, {}
        self.error = None
        self.closed = threading.Event()
        self._restore()
        self.worker = threading.Thread(target=self._supervise, daemon=True, name='camera-recovery')
        self.worker.start()

    def _save(self):
        temporary = self.path.with_suffix('.tmp')
        try:
            fd = os.open(temporary, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
            with os.fdopen(fd, 'w') as stream:
                os.fchmod(stream.fileno(), 0o600)
                json.dump(self.saved, stream)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(self.path)
            self.error = None
        except OSError:
            self.error = 'Camera settings could not be saved. Check disk space and permissions.'
            raise RuntimeError(self.error) from None

    def _restore(self):
        if not self.path.exists():
            return
        try:
            entries = json.loads(self.path.read_text())
            if not isinstance(entries, dict) or len(entries) > 4:
                raise ValueError('Invalid camera registry')
            # Validate the complete registry before starting any camera.
            validated = []
            for camera_id, entry in entries.items():
                settings = self.settings_type(**entry['settings'])
                if (settings.mode != 'live' or validate_source(settings.source)[1]
                        or not isinstance(entry['enabled'], bool)):
                    raise ValueError('Invalid saved camera')
                validated.append((camera_id, entry, settings))
            for camera_id, entry, settings in validated:
                engine = Engine(self.store, self.model_dir, camera_id, settings.name or 'Camera')
                engine.settings = settings
                engine.auto_reconnect = entry['enabled']
                self.cameras[camera_id], self.saved[camera_id] = engine, entry
                if entry['enabled']:
                    engine.start(settings)
        except (OSError, ValueError, TypeError, KeyError):
            self.error = 'Saved camera settings could not be restored. The original file was kept.'

    def connect(self, settings, engine=None):
        with self.lock:
            if self.error and not self.saved and self.path.exists():
                raise RuntimeError(self.error)
            reusable = None
            if (engine is None or engine.camera_id not in self.cameras) and len(self.cameras) >= 4:
                if settings.mode == 'live' and validate_source(settings.source)[1]:
                    reusable = self._finished_recording()
                if reusable is None:
                    raise RuntimeError('Four sessions are configured. Stop and remove a session before adding another.')
            engine = engine or Engine(self.store, self.model_dir, 'cam-' + uuid.uuid4().hex[:10],
                                      settings.name.strip() or f'Camera {len(self.cameras)+1}')
            if engine.thread and engine.thread.is_alive():
                raise RuntimeError('Stop the current session before connecting another source')
            persistent = settings.mode == 'live' and not validate_source(settings.source)[1]
            previous = self.saved.get(engine.camera_id)
            if persistent:
                self.saved[engine.camera_id] = {'settings': settings.model_dump(), 'enabled': True}
            else:
                self.saved.pop(engine.camera_id, None)
            try:
                self._save()
            except RuntimeError:
                if previous is None:
                    self.saved.pop(engine.camera_id, None)
                else:
                    self.saved[engine.camera_id] = previous
                raise
            engine.auto_reconnect = persistent
            engine.reconnect_attempts, engine.reconnect_at = 0, 0
            if settings.name.strip():
                engine.name = settings.name.strip()
            engine.start(settings)
            if reusable is not None:
                self.cameras.pop(reusable.camera_id, None)
            self.cameras[engine.camera_id] = engine
            return engine

    def _finished_recording(self):
        """Only file sessions can release a slot; saved or active cameras never do."""
        for engine in self.cameras.values():
            if (engine.camera_id in self.saved or engine.auto_reconnect or not engine.settings
                    or engine.settings.mode != 'live' or (engine.thread and engine.thread.is_alive())
                    or engine.snapshot()['status'] not in {'idle', 'finished'}):
                continue
            try:
                if validate_source(engine.settings.source)[1]:
                    return engine
            except ValueError:
                continue
        return None

    def recording_slot_available(self):
        with self.lock:
            return len(self.cameras) < 4 or self._finished_recording() is not None

    def stop(self, engine):
        with self.lock:
            engine.auto_reconnect = False
            engine.reconnect_at = 0
            try:
                if engine.camera_id in self.saved:
                    self.saved[engine.camera_id]['enabled'] = False
                    self._save()
            finally:
                engine.stop()

    def remove(self, engine):
        with self.lock:
            self.stop(engine)
            if engine.thread and engine.thread.is_alive():
                raise RuntimeError('Camera is still stopping; retry removal once it has stopped')
            self.saved.pop(engine.camera_id, None)
            self._save()
            self.cameras.pop(engine.camera_id, None)

    def set_eco(self, engine, enabled):
        with self.lock:
            engine.set_eco_mode(enabled)
            if engine.camera_id in self.saved:
                self.saved[engine.camera_id]['settings'] = engine.settings.model_dump()
                self._save()

    def update_settings(self, engine, changes):
        """Apply tuning as one validated change, retaining source and stop/recovery state."""
        with self.lock:
            if self.cameras.get(engine.camera_id) is not engine or engine.settings is None:
                raise RuntimeError('No input settings are available for this camera')
            if set(changes) - set(EDITABLE_SETTINGS):
                raise ValueError('Only camera detection and processing settings can be edited')
            previous_settings = engine.settings
            settings = self.settings_type(**{**previous_settings.model_dump(), **changes})
            if settings == previous_settings:
                return
            previous_entry = self.saved.get(engine.camera_id)
            active = bool(engine.thread and engine.thread.is_alive())
            restart = previous_entry['enabled'] if previous_entry is not None else active
            if restart or active:
                engine.stop()
                if engine.thread and engine.thread.is_alive():
                    raise RuntimeError('Camera is still stopping; retry saving after it has stopped')
            persisted = False
            try:
                if previous_entry is not None:
                    self.saved[engine.camera_id] = {**previous_entry, 'settings': settings.model_dump()}
                    self._save()
                    persisted = True
                if restart:
                    engine.start(settings)
                else:
                    engine.settings = settings
            except (RuntimeError, OSError) as exc:
                if previous_entry is not None:
                    self.saved[engine.camera_id] = previous_entry
                engine.settings = previous_settings
                rollback_error = None
                if persisted:
                    try:
                        self._save()
                    except RuntimeError:
                        rollback_error = 'Saved settings could not be rolled back. Check disk space and permissions.'
                if restart:
                    try:
                        engine.start(previous_settings)
                    except (RuntimeError, OSError):
                        rollback_error = rollback_error or 'Original settings were retained, but the camera could not restart. Reconnect it.'
                raise RuntimeError(rollback_error or str(exc)) from exc
            engine.reconnect_attempts, engine.reconnect_at = 0, 0

    def set_location(self, engine, location):
        with self.lock:
            if engine.settings is None or engine.settings.mode != 'live':
                raise RuntimeError('Select a live camera to set its location')
            previous = engine.settings.location
            engine.settings.location = location
            try:
                if engine.camera_id in self.saved:
                    self.saved[engine.camera_id]['settings'] = engine.settings.model_dump()
                    self._save()
            except RuntimeError:
                engine.settings.location = previous
                self.saved[engine.camera_id]['settings'] = engine.settings.model_dump()
                raise

    def recover_once(self, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            engines = list(self.cameras.values())
        for engine in engines:
            with self.lock:
                if self.closed.is_set() or not engine.auto_reconnect or engine.camera_id not in self.cameras:
                    continue
                state = engine.snapshot()
                healthy = state['status'] == 'running' and not state['stale']
                if healthy:
                    engine.reconnect_attempts, engine.reconnect_at = 0, 0
                    continue
                failed = state['status'] in {'error', 'finished', 'idle', 'stopping'} or (
                    state['stale'] and state['frame_age_seconds'] > 15)
                if not failed:
                    continue
                if not engine.reconnect_at:
                    engine.reconnect_at = now + 5
                if now < engine.reconnect_at:
                    continue
                engine.reconnect_attempts += 1
                engine.reconnect_at = now + min(60, 5 * 2 ** min(engine.reconnect_attempts, 4))
                engine.stop()
                try:
                    engine.start(engine.settings)
                except RuntimeError:
                    pass  # A model call still stopping is retried after the backoff.

    def _supervise(self):
        while not self.closed.wait(1):
            self.recover_once()

    def close(self):
        self.closed.set()
        self.worker.join(timeout=10)
        with self.lock:
            for engine in self.cameras.values():
                engine.stop()
