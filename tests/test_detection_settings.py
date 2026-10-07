"""New event settings remain local to a camera and preserve legacy registries."""
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
SOURCE = 'http://fixture-user:fixture-password@192.0.2.10:8080/video'
FIELDS = ('person_down_seconds', 'unattended_objects', 'unattended_seconds')


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


def test_legacy_defaults_do_not_enable_new_object_processing():
    creation, editing = StartRequest(), CameraSettingsRequest()
    for value in (creation, editing):
        assert value.person_down_seconds == 3
        assert value.unattended_seconds == 60
        assert value.unattended_objects is False
        assert value.object_detection is False
    assert editing.model_dump(exclude_unset=True) == {}
    assert set(FIELDS) <= set(EDITABLE_SETTINGS)


@pytest.mark.parametrize('model', [StartRequest, CameraSettingsRequest])
@pytest.mark.parametrize('field,value', [
    ('person_down_seconds', .99), ('person_down_seconds', 60.01),
    ('person_down_seconds', float('nan')), ('person_down_seconds', float('inf')),
    ('person_down_seconds', float('-inf')), ('person_down_seconds', None),
    ('unattended_seconds', 9.99), ('unattended_seconds', 3600.01),
    ('unattended_seconds', float('nan')), ('unattended_seconds', float('inf')),
    ('unattended_seconds', float('-inf')), ('unattended_seconds', None),
    ('unattended_objects', 'true'), ('unattended_objects', 'false'),
    ('unattended_objects', 1), ('unattended_objects', 0), ('unattended_objects', None),
])
def test_event_setting_bounds_nonfinite_numbers_and_strict_boolean(model, field, value):
    with pytest.raises(ValidationError):
        model(**{field: value})


@pytest.mark.parametrize('model', [StartRequest, CameraSettingsRequest])
@pytest.mark.parametrize('person_down_seconds,unattended_seconds', [(1, 10), (60, 3600)])
def test_event_duration_boundaries_are_supported(model, person_down_seconds, unattended_seconds):
    result = model(person_down_seconds=person_down_seconds, unattended_seconds=unattended_seconds)
    assert result.person_down_seconds == person_down_seconds
    assert result.unattended_seconds == unattended_seconds


def test_unattended_processing_is_live_only_and_independent_of_weapon_detection():
    with pytest.raises(ValidationError):
        StartRequest(mode='demo', unattended_objects=True)
    result = settings(unattended_objects=True, object_detection=False)
    assert result.unattended_objects and not result.object_detection
    # A partial PATCH has no mode/source. The saved camera is validated after merge.
    patch = CameraSettingsRequest(unattended_objects=True)
    assert patch.model_dump(exclude_unset=True) == {'unattended_objects': True}


def test_legacy_saved_camera_defaults_restore_without_enabling_or_starting_it(tmp_path):
    store = Store(tmp_path)
    old = settings().model_dump()
    for field in FIELDS:
        old.pop(field)
    path = store.directory/'cameras.json'
    original = json.dumps({'old-camera': {'enabled': False, 'settings': old}})
    path.write_text(original)
    manager = CameraManager(store, tmp_path, StartRequest)
    try:
        assert manager.error is None
        camera = manager.cameras['old-camera']
        assert camera.settings.person_down_seconds == 3
        assert camera.settings.unattended_seconds == 60
        assert camera.settings.unattended_objects is False
        assert not camera.auto_reconnect and camera.starts == []
        assert path.read_text() == original
        assert {field: camera.snapshot()['settings'][field] for field in FIELDS} == {
            'person_down_seconds': 3, 'unattended_objects': False, 'unattended_seconds': 60}
    finally:
        manager.close()
        store.close()


def test_camera_patch_merges_new_values_only_for_target_and_persists_stopped_state(tmp_path):
    app = create_app(tmp_path)
    with TestClient(app) as client:
        authenticate(client)
        first = client.post('/api/cameras', json=settings().model_dump(), headers=HEADERS)
        second = client.post('/api/cameras', json=settings(person_down_seconds=8).model_dump(), headers=HEADERS)
        assert first.status_code == second.status_code == 200
        first_id, second_id = first.json()['camera_id'], second.json()['camera_id']
        camera, other = app.state.cameras[first_id], app.state.cameras[second_id]
        original, other_original = camera.settings.model_dump(), other.settings.model_dump()
        changes = {'person_down_seconds': 6, 'unattended_objects': True, 'unattended_seconds': 120}
        route = f'/api/cameras/{first_id}/settings'
        response = client.patch(route, json=changes, headers=HEADERS)
        assert response.status_code == 200, response.text
        assert {key: response.json()['settings'][key] for key in FIELDS} == changes
        assert response.json()['settings']['object_detection'] is False
        assert SOURCE not in response.text and 'fixture-password' not in response.text
        expected = {**original, **changes}
        assert camera.settings.model_dump() == expected and len(camera.starts) == 2
        assert other.settings.model_dump() == other_original and len(other.starts) == 1
        saved = json.loads(app.state.manager.path.read_text())
        assert saved[first_id] == {'enabled': True, 'settings': expected}
        assert saved[second_id] == {'enabled': True, 'settings': other_original}
        assert client.patch(route, json=changes, headers=HEADERS).status_code == 200
        assert len(camera.starts) == 2
        app.state.manager.stop(camera)
        stopped_change = {'person_down_seconds': 60, 'unattended_seconds': 3600}
        response = client.patch(route, json=stopped_change, headers=HEADERS)
        assert response.status_code == 200, response.text
        expected.update(stopped_change)
        assert camera.settings.model_dump() == expected
        assert camera.state['status'] == 'idle' and not camera.auto_reconnect
        assert len(camera.starts) == 2
        assert json.loads(app.state.manager.path.read_text())[first_id] == {'enabled': False, 'settings': expected}
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


def test_invalid_event_settings_api_patch_is_atomic_and_keeps_private_source_hidden(tmp_path):
    app = create_app(tmp_path)
    with TestClient(app) as client:
        authenticate(client)
        creation = client.post('/api/cameras', json=settings().model_dump(), headers=HEADERS)
        assert creation.status_code == 200
        camera_id = creation.json()['camera_id']
        camera = app.state.cameras[camera_id]
        original = camera.settings.model_dump()
        before = app.state.manager.path.read_text()
        for changes in ({'person_down_seconds': 0}, {'person_down_seconds': 61},
                        {'unattended_seconds': 9}, {'unattended_seconds': 3601},
                        {'unattended_objects': 'true'}, {'unattended_objects': 1},
                        {'person_down_seconds': None}):
            response = client.patch(f'/api/cameras/{camera_id}/settings', json=changes, headers=HEADERS)
            assert response.status_code == 422, response.text
            assert SOURCE not in response.text and 'fixture-password' not in response.text
            assert camera.settings.model_dump() == original and len(camera.starts) == 1
            assert app.state.manager.path.read_text() == before


def test_demo_patch_cannot_turn_on_unattended_objects_after_partial_validation(tmp_path):
    app = create_app(tmp_path)
    with TestClient(app) as client:
        authenticate(client)
        response = client.post('/api/cameras', json={'mode': 'demo'}, headers=HEADERS)
        assert response.status_code == 200
        camera_id = response.json()['camera_id']
        camera = app.state.cameras[camera_id]
        original = camera.settings.model_dump()
        response = client.patch(f'/api/cameras/{camera_id}/settings',
                                json={'unattended_objects': True}, headers=HEADERS)
        assert response.status_code == 422, response.text
        assert camera.settings.model_dump() == original and len(camera.starts) == 1
