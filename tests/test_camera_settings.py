"""Camera tuning edits use the saved source and never load a model or call a provider."""
import json

import pytest
from fastapi.testclient import TestClient

from conftest import authenticate
from vmd.api import StartRequest, create_app
from vmd.cameras import CameraManager
from vmd.engine import Engine
from vmd.storage import Store


HEADERS = {'x-vmd-client': 'dashboard'}
SOURCE = 'http://fixture-user:fixture-password@192.0.2.1:8080/video'


class Thread:
    alive = True

    def is_alive(self):
        return self.alive


class Camera(Engine):
    def __init__(self, *args):
        super().__init__(*args)
        self.starts = []
        self.stops = 0
        self.block_stop = False
        self.fail_threshold = None

    def start(self, settings):
        if settings.threshold == self.fail_threshold:
            raise RuntimeError('Could not start pipeline')
        assert not self.thread or not self.thread.is_alive()
        self.settings = settings
        self.starts.append(settings.model_dump())
        self.thread = Thread()
        self.state.update(status='running', mode=settings.mode, depth=settings.depth,
                          detection_mode=settings.detection_mode)

    def stop(self):
        self.stops += 1
        if self.thread and not self.block_stop:
            self.thread.alive = False
        self.state['status'] = 'stopping' if self.block_stop else 'idle'


@pytest.fixture(autouse=True)
def no_models_or_recovery(monkeypatch):
    monkeypatch.setattr('vmd.cameras.Engine', Camera)
    monkeypatch.setattr(CameraManager, '_supervise', lambda self: self.closed.wait())


@pytest.fixture
def manager(tmp_path):
    store = Store(tmp_path)
    result = CameraManager(store, tmp_path, StartRequest)
    yield result
    result.close()
    store.close()


def settings():
    return StartRequest(mode='live', name='Fixture camera', source=SOURCE, device='cpu',
                        depth='ZipDepth', depth_fps=1, detection_mode='depth_confirmed',
                        location={'place': 'Fixture gate', 'latitude': 0, 'longitude': 0})


def test_running_edit_persists_and_stopped_edit_stays_stopped(manager):
    camera = manager.connect(settings())
    original = camera.settings.model_dump()
    changes = {'device': 'auto', 'depth': 'MiDaS_small', 'depth_fps': .8,
               'detection_mode': 'responsive', 'eco_mode': True, 'object_detection': True,
               'object_fps': .6, 'threshold': .78, 'hold_seconds': 1.5,
               'fight_confirmation_seconds': 1.0, 'target_fps': 6}
    manager.update_settings(camera, changes)
    expected = {**original, **changes}
    assert len(camera.starts) == 2 and camera.starts[-1] == expected
    assert camera.auto_reconnect
    assert json.loads(manager.path.read_text())[camera.camera_id] == {'enabled': True, 'settings': expected}
    manager.update_settings(camera, changes)
    assert len(camera.starts) == 2  # Saving an unchanged form does not interrupt analysis.
    manager.stop(camera)
    manager.update_settings(camera, {'threshold': .84})
    expected['threshold'] = .84
    assert len(camera.starts) == 2 and camera.state['status'] == 'idle'
    assert not camera.auto_reconnect
    manager.close()
    restored = CameraManager(manager.store, manager.model_dir, StartRequest)
    try:
        saved = restored.cameras[camera.camera_id]
        assert saved.settings.model_dump() == expected
        assert saved.starts == [] and not saved.auto_reconnect
    finally:
        restored.close()


def test_stop_in_progress_is_not_turned_back_on(manager):
    camera = manager.connect(settings())
    camera.block_stop = True
    manager.stop(camera)
    before = manager.path.read_text()
    with pytest.raises(RuntimeError, match='still stopping'):
        manager.update_settings(camera, {'threshold': .8})
    assert manager.path.read_text() == before
    assert camera.settings.threshold == .6 and len(camera.starts) == 1
    camera.block_stop = False
    manager.update_settings(camera, {'threshold': .8})
    assert camera.settings.threshold == .8 and len(camera.starts) == 1
    assert not camera.auto_reconnect
    assert not manager.saved[camera.camera_id]['enabled']


def test_save_failure_restores_running_original(manager, monkeypatch):
    camera = manager.connect(settings())
    original = camera.settings.model_dump()
    saved = manager.path.read_text()

    def fail():
        raise RuntimeError('Storage unavailable')

    monkeypatch.setattr(manager, '_save', fail)
    with pytest.raises(RuntimeError, match='Storage unavailable'):
        manager.update_settings(camera, {'threshold': .8})
    assert manager.path.read_text() == saved
    assert manager.saved[camera.camera_id]['settings'] == original
    assert camera.settings.model_dump() == original
    assert camera.starts == [original, original] and camera.thread.is_alive()


def test_start_failure_rolls_back_registry_and_restarts_original(manager):
    camera = manager.connect(settings())
    original = camera.settings.model_dump()
    camera.fail_threshold = .8
    with pytest.raises(RuntimeError, match='Could not start pipeline'):
        manager.update_settings(camera, {'threshold': .8})
    assert json.loads(manager.path.read_text())[camera.camera_id]['settings'] == original
    assert camera.settings.model_dump() == original
    assert camera.starts == [original, original] and camera.thread.is_alive()


def test_settings_api_auth_validation_and_private_source(tmp_path):
    app = create_app(tmp_path)
    with TestClient(app) as client:
        route = '/api/cameras/missing/settings'
        assert client.patch(route, json={}, headers=HEADERS).status_code == 401
        authenticate(client)
        assert client.patch(route, json={}).status_code == 403
        assert client.patch(route, json={}, headers={**HEADERS, 'origin': 'https://untrusted.example'}).status_code == 403
        assert client.patch(route, json={}, headers=HEADERS).status_code == 404
        response = client.post('/api/cameras', json=settings().model_dump(), headers=HEADERS)
        assert response.status_code == 200
        camera_id = response.json()['camera_id']
        route = f'/api/cameras/{camera_id}/settings'
        camera = app.state.cameras[camera_id]
        original = camera.settings.model_dump()
        for change in ({'depth': 'off'}, {'depth_fps': .1}, {'threshold': 2}, {'hold_seconds': 0},
                       {'source': '0'}, {'name': 'Other'}, {'location': None}, {'eco_mode': 'true'},
                       {'threshold': None}, {'target_fps': 1}, {'object_fps': 3},
                       {'fight_confirmation_seconds': .4}, {'fight_confirmation_seconds': 11}):
            failure = client.patch(route, json=change, headers=HEADERS)
            assert failure.status_code == 422, failure.text
            assert SOURCE not in failure.text and 'fixture-password' not in failure.text
            assert camera.settings.model_dump() == original and len(camera.starts) == 1
        response = client.patch(route, json={'threshold': .81, 'hold_seconds': 1.2,
                                            'fight_confirmation_seconds': .8}, headers=HEADERS)
        assert response.status_code == 200, response.text
        assert response.json()['settings']['threshold'] == .81
        assert response.json()['settings']['fight_confirmation_seconds'] == .8
        assert response.json()['settings']['depth'] == 'ZipDepth'
        assert response.json()['camera_id'] == camera_id
        assert SOURCE not in response.text and 'fixture-password' not in response.text
        assert camera.settings.source == SOURCE and camera.settings.location == settings().location
        assert camera.settings.name == original['name']
        assert len(camera.starts) == 2
