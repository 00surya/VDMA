"""Opt-in rider review settings are strict, camera-local and backward compatible."""
import json

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from conftest import authenticate
from vmd.api import CameraSettingsRequest, StartRequest, create_app
from vmd.cameras import CameraManager, EDITABLE_SETTINGS
from vmd.engine import Engine
from vmd.storage import Store


HEADERS = {'x-vmd-client': 'dashboard'}
SOURCE = 'http://fixture-user:fixture-password@192.0.2.20:8080/video'


class Thread:
    alive = True

    def is_alive(self):
        return self.alive


class Camera(Engine):
    def __init__(self, *args):
        super().__init__(*args)
        self.starts = []

    def start(self, settings):
        assert not self.thread or not self.thread.is_alive()
        self.settings = settings
        self.starts.append(settings.model_dump())
        self.thread = Thread()
        self.state.update(status='running', mode=settings.mode, depth=settings.depth,
                          detection_mode=settings.detection_mode)

    def stop(self):
        if self.thread:
            self.thread.alive = False
        self.state['status'] = 'idle'


@pytest.fixture(autouse=True)
def no_models_or_recovery(monkeypatch):
    monkeypatch.setattr('vmd.cameras.Engine', Camera)
    monkeypatch.setattr(CameraManager, '_supervise', lambda self: self.closed.wait())


def settings(**changes):
    return StartRequest(mode='live', name='Fixture gate', source=SOURCE, device='cpu',
                        location={'place': 'Fixture gate', 'latitude': 0, 'longitude': 0},
                        **changes)


def test_rider_review_is_off_for_new_inputs_and_omitted_patch_fields():
    assert StartRequest().snatching_vehicles is False
    edit = CameraSettingsRequest()
    assert edit.snatching_vehicles is False
    assert edit.model_dump(exclude_unset=True) == {}
    assert 'snatching_vehicles' in EDITABLE_SETTINGS


@pytest.mark.parametrize('model', [StartRequest, CameraSettingsRequest])
@pytest.mark.parametrize('value', ['true', 'false', 0, 1, None, [], {}])
def test_rider_review_rejects_boolean_coercion(model, value):
    with pytest.raises(ValidationError):
        model(**{'snatching_vehicles': value})


@pytest.mark.parametrize('value', [True, False])
def test_rider_review_supports_explicit_bool_without_weapon_or_bag_checks(value):
    result = settings(snatching_vehicles=value)
    assert result.snatching_vehicles is value
    assert result.object_detection is False and result.unattended_objects is False
    assert CameraSettingsRequest(snatching_vehicles=value).model_dump(exclude_unset=True) == {
        'snatching_vehicles': value}


def test_rider_review_requires_camera_or_recorded_video_input():
    with pytest.raises(ValidationError, match='Rider-snatching review'):
        StartRequest(mode='demo', snatching_vehicles=True)
    recording = StartRequest(mode='live', recording_id='a' * 32, snatching_vehicles=True)
    assert recording.snatching_vehicles is True


def test_legacy_registry_restores_off_without_starting_or_rewriting_it(tmp_path):
    store = Store(tmp_path)
    previous = settings().model_dump()
    previous.pop('snatching_vehicles')
    path = store.directory / 'cameras.json'
    original = json.dumps({'old-camera': {'enabled': False, 'settings': previous}})
    path.write_text(original)
    manager = CameraManager(store, tmp_path, StartRequest)
    try:
        assert manager.error is None
        camera = manager.cameras['old-camera']
        assert camera.settings.snatching_vehicles is False
        assert camera.snapshot()['settings']['snatching_vehicles'] is False
        assert camera.starts == [] and not camera.auto_reconnect
        assert path.read_text() == original
    finally:
        manager.close()
        store.close()


def test_rider_patch_merges_camera_settings_and_persists_stopped_roundtrip(tmp_path):
    app = create_app(tmp_path)
    with TestClient(app) as client:
        authenticate(client)
        original_settings = settings(unattended_objects=True, unattended_seconds=90,
                                     person_down_seconds=7, object_fps=.5)
        first = client.post('/api/cameras', json=original_settings.model_dump(), headers=HEADERS)
        second = client.post('/api/cameras', json=settings().model_dump(), headers=HEADERS)
        assert first.status_code == second.status_code == 200
        first_id, second_id = first.json()['camera_id'], second.json()['camera_id']
        camera, other = app.state.cameras[first_id], app.state.cameras[second_id]
        original, other_original = camera.settings.model_dump(), other.settings.model_dump()
        route = f'/api/cameras/{first_id}/settings'
        response = client.patch(route, json={'snatching_vehicles': True}, headers=HEADERS)
        assert response.status_code == 200, response.text
        expected = {**original, 'snatching_vehicles': True}
        assert camera.settings.model_dump() == expected
        assert response.json()['settings']['snatching_vehicles'] is True
        assert other.settings.model_dump() == other_original
        assert len(camera.starts) == 2 and len(other.starts) == 1
        assert SOURCE not in response.text and 'fixture-password' not in response.text
        saved = json.loads(app.state.manager.path.read_text())
        assert saved[first_id] == {'enabled': True, 'settings': expected}
        assert saved[second_id] == {'enabled': True, 'settings': other_original}
        assert client.patch(route, json={'snatching_vehicles': True}, headers=HEADERS).status_code == 200
        assert len(camera.starts) == 2
        app.state.manager.stop(camera)
        response = client.patch(route, json={'snatching_vehicles': False}, headers=HEADERS)
        assert response.status_code == 200, response.text
        expected['snatching_vehicles'] = False
        assert camera.settings.model_dump() == expected
        assert len(camera.starts) == 2 and camera.state['status'] == 'idle'
        assert not camera.auto_reconnect
        assert json.loads(app.state.manager.path.read_text())[first_id] == {
            'enabled': False, 'settings': expected}
    store = Store(tmp_path)
    restored = CameraManager(store, tmp_path, StartRequest)
    try:
        assert restored.error is None
        camera = restored.cameras[first_id]
        assert camera.settings.model_dump() == expected
        assert camera.starts == [] and not camera.auto_reconnect
        assert restored.cameras[second_id].settings.model_dump() == other_original
    finally:
        restored.close()
        store.close()


def test_invalid_rider_patch_and_demo_merge_leave_state_unchanged(tmp_path):
    app = create_app(tmp_path)
    with TestClient(app) as client:
        authenticate(client)
        response = client.post('/api/cameras', json=settings().model_dump(), headers=HEADERS)
        assert response.status_code == 200
        camera_id = response.json()['camera_id']
        camera = app.state.cameras[camera_id]
        original, saved = camera.settings.model_dump(), app.state.manager.path.read_text()
        for value in ('true', 'false', 1, 0, None):
            response = client.patch(f'/api/cameras/{camera_id}/settings',
                                    json={'snatching_vehicles': value}, headers=HEADERS)
            assert response.status_code == 422, response.text
            assert SOURCE not in response.text and 'fixture-password' not in response.text
            assert camera.settings.model_dump() == original and len(camera.starts) == 1
            assert app.state.manager.path.read_text() == saved
        response = client.post('/api/cameras', json={'mode': 'demo'}, headers=HEADERS)
        assert response.status_code == 200
        demo_id = response.json()['camera_id']
        demo = app.state.cameras[demo_id]
        original = demo.settings.model_dump()
        response = client.patch(f'/api/cameras/{demo_id}/settings',
                                json={'snatching_vehicles': True}, headers=HEADERS)
        assert response.status_code == 422, response.text
        assert demo.settings.model_dump() == original and len(demo.starts) == 1
        assert app.state.manager.path.read_text() == saved
