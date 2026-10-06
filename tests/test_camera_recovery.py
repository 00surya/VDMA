import json
import stat

import pytest

from vmd.api import CameraLocation, StartRequest
from vmd.cameras import CameraManager
from vmd.engine import Engine
from vmd.storage import Store


class Camera:
    def __init__(self, store, models, camera_id, name):
        self.camera_id, self.name = camera_id, name
        self.thread, self.settings = None, None
        self.auto_reconnect, self.reconnect_at, self.reconnect_attempts = False, 0, 0
        self.starts, self.state = [], 'idle'
    def start(self, settings):
        self.settings = settings
        self.starts.append(settings.model_dump())
        self.state = 'starting'
    def stop(self): self.state = 'idle'
    def set_eco_mode(self, enabled): self.settings.eco_mode = enabled
    def snapshot(self):
        return {'status': self.state, 'stale': self.state == 'running', 'frame_age_seconds': 20}


@pytest.fixture
def manager_setup(tmp_path, monkeypatch):
    import vmd.cameras as module
    monkeypatch.setattr(module, 'Engine', Camera)
    monkeypatch.setattr(CameraManager, '_supervise', lambda self: self.closed.wait())
    store = Store(tmp_path)
    managers = []
    def make():
        result = CameraManager(store, tmp_path, StartRequest)
        managers.append(result)
        return result
    yield make
    for manager in managers: manager.close()
    store.close()


def test_recovery_retains_all_settings_and_stop_remove_survive_restart(manager_setup):
    manager = manager_setup()
    settings = StartRequest(mode='live', name='Phone', source='http://user:pass@192.0.2.1:8080/video',
        device='mps', depth='ZipDepth', depth_fps=1.2, detection_mode='depth_confirmed',
        threshold=.73, hold_seconds=1.4, target_fps=9, object_detection=True, object_fps=.8, eco_mode=True,
        location={'place': 'Gate', 'latitude': 28.61, 'longitude': 77.21})
    first = manager.connect(settings)
    second = manager.connect(StartRequest(mode='live', name='Other camera', source='0', threshold=.51))
    assert stat.S_IMODE(manager.path.stat().st_mode) == 0o600
    first.state = 'error'
    manager.recover_once(100)
    manager.recover_once(104)
    assert len(first.starts) == 1
    manager.recover_once(105)
    assert len(first.starts) == 2 and first.starts[-1] == settings.model_dump()
    assert len(second.starts) == 1
    first.state = 'error'
    manager.recover_once(110)
    assert len(first.starts) == 2
    manager.recover_once(115)
    assert len(first.starts) == 3
    manager.set_eco(first, False)
    manager.stop(first)
    first.state = 'error'
    manager.recover_once(1000)
    assert len(first.starts) == 3
    manager.close()
    restored = manager_setup()
    saved_first, saved_second = restored.cameras[first.camera_id], restored.cameras[second.camera_id]
    assert saved_first.starts == [] and not saved_first.auto_reconnect
    assert saved_first.settings.threshold == .73 and not saved_first.settings.eco_mode
    assert saved_second.starts[0]['threshold'] == .51 and saved_second.auto_reconnect
    restored.connect(saved_first.settings, saved_first)
    assert saved_first.auto_reconnect and saved_first.starts[-1]['device'] == 'mps'
    restored.remove(saved_first)
    assert first.camera_id not in json.loads(restored.path.read_text())


def test_recordings_finish_without_auto_restarting_or_being_restored(manager_setup, tmp_path):
    source = tmp_path / 'movie.avi'
    source.write_bytes(b'fixture')
    manager = manager_setup()
    camera = manager.connect(StartRequest(mode='live', source=str(source)))
    camera.state = 'finished'
    manager.recover_once(1000)
    assert len(camera.starts) == 1 and not camera.auto_reconnect
    assert json.loads(manager.path.read_text()) == {}


def test_corrupt_registry_is_kept_and_not_overwritten(manager_setup, tmp_path):
    path = tmp_path / 'cameras.json'
    path.write_text('broken')
    manager = manager_setup()
    assert manager.error and not manager.cameras
    with pytest.raises(RuntimeError, match='restored'):
        manager.connect(StartRequest(mode='live', source='0'))
    assert path.read_text() == 'broken'


def test_browser_location_survives_restart_without_rewriting_incidents(manager_setup):
    manager = manager_setup()
    settings = StartRequest(mode='live', source='0', threshold=.73, eco_mode=True,
                            location={'place': 'Old gate', 'latitude': 10, 'longitude': 20})
    camera = manager.connect(settings)
    manager.stop(camera)
    original = settings.location.model_dump()
    manager.store._incident({'id': 'a'*32, 'created': 1, 'mode': 'live', 'score': .95,
                             'reasons': [], 'signals': {'location': original}}, [])
    new_location = CameraLocation(place='Laptop location', latitude=0, longitude=0,
                                  source='browser', accuracy_meters=12.5, captured_at=1000)
    manager.set_location(camera, new_location)
    manager.close()
    restored = manager_setup().cameras[camera.camera_id]
    assert restored.settings.location == new_location
    assert restored.settings.threshold == .73 and restored.settings.eco_mode
    assert not restored.auto_reconnect and not restored.starts
    assert manager.store.incident('a'*32)['signals']['location'] == original


def test_browser_location_failed_save_rolls_back_settings(manager_setup, monkeypatch):
    manager = manager_setup()
    camera = manager.connect(StartRequest(mode='live', source='0',
        location={'place': 'Gate', 'latitude': 10, 'longitude': 20}))
    before = camera.settings.model_dump()
    def fail_save():
        raise RuntimeError('Storage unavailable')
    monkeypatch.setattr(manager, '_save', fail_save)
    with pytest.raises(RuntimeError, match='Storage unavailable'):
        manager.set_location(camera, CameraLocation(place='Laptop location', latitude=0, longitude=0,
                                                    source='browser', accuracy_meters=20))
    assert camera.settings.model_dump() == before
    assert manager.saved[camera.camera_id]['settings'] == before


def test_snapshot_distinguishes_stopped_camera_from_recordings(tmp_path):
    store = Store(tmp_path)
    try:
        engine = Engine(store, tmp_path)
        assert not engine.snapshot()['live_camera']
        engine.settings = StartRequest(mode='live', source='http://user:password@192.0.2.1/video')
        snapshot = engine.snapshot()
        assert snapshot['live_camera'] and not snapshot['auto_reconnect']
        assert '192.0.2.1' not in json.dumps(snapshot) and 'password' not in json.dumps(snapshot)
        recording = tmp_path / 'recording.mp4'
        recording.write_bytes(b'fixture')
        engine.settings = StartRequest(mode='live', source=str(recording))
        assert not engine.snapshot()['live_camera']
        recording.unlink()
        assert not engine.snapshot()['live_camera']
        engine.settings = StartRequest(mode='demo')
        assert not engine.snapshot()['live_camera']
    finally:
        store.close()


@pytest.mark.parametrize('field,value', [('accuracy_meters', -1), ('accuracy_meters', float('nan')),
                                        ('captured_at', -1), ('captured_at', float('inf')),
                                        ('source', 'fictional')])
def test_location_provenance_rejects_invalid_values(field, value):
    with pytest.raises(ValueError):
        CameraLocation(place='Gate', latitude=0, longitude=0, **{field: value})
