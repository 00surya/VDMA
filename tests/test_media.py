from conftest import authenticate
import shutil

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from vmd.api import create_app

HEADERS = {'x-vmd-client': 'dashboard', 'x-file-name': 'Test%20recording.avi'}


def video_bytes(tmp_path):
    path = tmp_path / 'fixture.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (160, 100))
    assert writer.isOpened()
    for i in range(40):
        writer.write(np.full((100, 160, 3), (i*5, 80, 40), np.uint8))
    writer.release()
    return path.read_bytes()


def test_upload_library_playback_ranges_and_analysis_settings(tmp_path, monkeypatch):
    payload = video_bytes(tmp_path)
    app = create_app(tmp_path / 'data')
    with TestClient(app) as client:
        authenticate(client)
        assert client.post('/api/recordings', content=payload).status_code == 403
        response = client.post('/api/recordings', content=payload, headers=HEADERS)
        assert response.status_code == 200, response.text
        item = response.json()
        assert item['name'] == 'Test recording.avi' and item['duration'] == 4
        assert client.get('/api/recordings').json() == [item]
        assert client.get(f"/api/recordings/{item['id']}/download").content == payload
        if not shutil.which('ffmpeg'):
            pytest.importorskip('imageio_ffmpeg')
        response = client.get(f"/api/recordings/{item['id']}/play")
        assert response.status_code == 200 and response.headers['content-type'] == 'video/mp4'
        assert b'ftyp' in response.content[:40]
        ranged = client.get(f"/api/recordings/{item['id']}/play", headers={'Range':'bytes=0-63'})
        assert ranged.status_code == 206 and len(ranged.content) == 64
        assert len(list(app.state.media.cache.glob('*.mp4'))) == 1  # Reuses prepared playback.
        calls = []
        def start(engine, settings):
            engine.settings = settings
            calls.append(settings.model_dump())
        monkeypatch.setattr('vmd.engine.Engine.start', start)
        response = client.post('/api/cameras', json={'mode':'live', 'recording_id':item['id'],
            'threshold':.81, 'hold_seconds':1.3, 'depth':'off'}, headers=HEADERS)
        assert response.status_code == 200, response.text
        assert calls[0]['threshold'] == .81 and calls[0]['hold_seconds'] == 1.3
        assert calls[0]['source'] == str(app.state.media.source(item['id']))
        assert not response.json()['auto_reconnect']
        assert client.post('/api/cameras', json={'mode':'live', 'recording_id':'a'*32}, headers=HEADERS).status_code == 404
    with TestClient(create_app(tmp_path / 'data')) as restored:
        authenticate(restored)
        assert restored.get('/api/recordings').json() == [item]
        assert restored.get('/api/cameras').json()['cameras'] == []


def test_invalid_oversized_and_traversal_uploads_do_not_leave_files(tmp_path):
    app = create_app(tmp_path / 'data')
    with TestClient(app) as client:
        authenticate(client)
        assert client.post('/api/recordings', content=b'bad', headers=HEADERS).status_code == 422
        assert client.post('/api/recordings', content=b'bad', headers={**HEADERS,'x-file-name':'evil.html'}).status_code == 422
        assert client.post('/api/recordings', content=b'bad', headers={**HEADERS,'content-length':str(1024**3+1)}).status_code == 413
        assert not list(app.state.media.directory.iterdir())
        response = client.post('/api/recordings', content=video_bytes(tmp_path),
            headers={**HEADERS,'x-file-name':'..%2F..%2Ftest.avi'})
        assert response.status_code == 200 and response.json()['name'] == 'test.avi'
        assert not (tmp_path/'test.avi').exists()
        assert client.get('/api/recordings/not-found/play').status_code == 404


def test_old_incident_can_be_played_even_outside_the_recent_100(tmp_path, monkeypatch):
    app = create_app(tmp_path)
    with TestClient(app) as client:
        authenticate(client)
        path = app.state.store.clips / 'old.avi'
        path.write_bytes(video_bytes(tmp_path))
        with app.state.store.connect() as conn:
            for i in range(101):
                conn.execute("INSERT INTO incidents(id,created,mode,score,reasons,signals,clip) VALUES(?,?,'live',.8,'[]','{}','old.avi')", (str(i), i))
        assert all(event['id'] != '0' for event in client.get('/api/incidents').json())
        assert client.get('/api/incidents/0/clip').status_code == 200
        # Conversion itself is exercised above; verify this endpoint's exact source.
        def playback(source):
            assert source == path.resolve()
            return path
        monkeypatch.setattr(app.state.media, 'playback', playback)
        assert client.get('/api/incidents/0/play').status_code == 200


@pytest.mark.parametrize('finished_status', ['finished', 'idle'])
def test_analysis_reuses_only_finished_file_capacity_preserving_cameras_and_evidence(tmp_path, monkeypatch, finished_status):
    def start(engine, settings):
        engine.settings = settings
        engine.state.update(status='running', mode=settings.mode)
    monkeypatch.setattr('vmd.engine.Engine.start', start)
    app = create_app(tmp_path / 'data')
    with TestClient(app) as client:
        authenticate(client)
        saved = []
        for index in range(3):
            camera = client.post('/api/cameras', json={'mode': 'live', 'source': str(index),
                'name': f'Saved camera {index}'}, headers=HEADERS).json()
            saved.append(camera['camera_id'])
            client.post(f"/api/cameras/{camera['camera_id']}/stop", json={}, headers=HEADERS)
        registry = app.state.manager.path.read_text()
        recording = client.post('/api/recordings', content=video_bytes(tmp_path), headers=HEADERS).json()
        request = {'mode': 'live', 'recording_id': recording['id'], 'threshold': .72,
                   'hold_seconds': 1.3, 'fight_confirmation_seconds': .8}
        first = client.post('/api/cameras', json=request, headers=HEADERS).json()
        assert not client.get('/api/cameras').json()['recording_slot_available']
        assert client.post('/api/cameras', json=request, headers=HEADERS).status_code == 409
        file_engine = app.state.cameras[first['camera_id']]
        file_engine.state['status'] = finished_status
        app.state.store._incident({'id': 'a'*32, 'created': 1, 'mode': 'live', 'score': .8,
            'camera_id': first['camera_id'], 'camera_name': 'Prior analysis', 'event_type': 'possible_fight',
            'reasons': ['Fixture'], 'signals': {'live_camera': False}}, [])
        assert client.get('/api/cameras').json()['recording_slot_available']
        second = client.post('/api/cameras', json=request, headers=HEADERS)
        assert second.status_code == 200, second.text
        assert second.json()['camera_id'] != first['camera_id']
        assert second.json()['settings']['fight_confirmation_seconds'] == .8
        assert not second.json()['live_camera'] and not second.json()['auto_reconnect']
        assert set(app.state.cameras) == set(saved + [second.json()['camera_id']])
        assert app.state.manager.path.read_text() == registry
        assert client.get('/api/incidents').json()[0]['camera_id'] == first['camera_id']
        assert client.get('/api/recordings').json() == [recording]


def test_analysis_capacity_does_not_replace_saved_cameras_or_lose_previous_on_start_failure(tmp_path, monkeypatch):
    def start(engine, settings):
        engine.settings = settings
        engine.state.update(status='idle', mode=settings.mode)
        if settings.name == 'Fail start':
            raise RuntimeError('Fixture start failure')
    monkeypatch.setattr('vmd.engine.Engine.start', start)
    app = create_app(tmp_path / 'data')
    with TestClient(app) as client:
        authenticate(client)
        recording = client.post('/api/recordings', content=video_bytes(tmp_path), headers=HEADERS).json()
        for index in range(3):
            client.post('/api/cameras', json={'mode': 'live', 'source': str(index)}, headers=HEADERS)
        request = {'mode': 'live', 'recording_id': recording['id']}
        first = client.post('/api/cameras', json=request, headers=HEADERS).json()
        before = set(app.state.cameras)
        failed = client.post('/api/cameras', json={**request, 'name': 'Fail start'}, headers=HEADERS)
        assert failed.status_code == 409 and set(app.state.cameras) == before
        client.post(f"/api/cameras/{first['camera_id']}/remove", json={}, headers=HEADERS)
        client.post('/api/cameras', json={'mode': 'live', 'source': '3'}, headers=HEADERS)
        assert not client.get('/api/cameras').json()['recording_slot_available']
        assert client.post('/api/cameras', json=request, headers=HEADERS).status_code == 409
