"""Expiring video links must not invoke the response workflow."""
import hashlib
import json
import os
import sqlite3
import subprocess
import time
from types import SimpleNamespace
from urllib.parse import urlsplit

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import authenticate
from vmd.api import create_app
from vmd.evidence import configuration, public_origin
from vmd.share import create_share_app

HEADERS = {'x-vmd-client': 'dashboard'}
INCIDENT = 'a' * 32
PATH = f'/api/incidents/{INCIDENT}/share'
PACKAGE = f'/api/incidents/{INCIDENT}/event-package'


@pytest.fixture
def clients(tmp_path, monkeypatch):
    monkeypatch.setenv('VMD_EVIDENCE_BASE_URL', 'https://evidence.example')
    for key in ('TWILIO_ACCOUNT_SID', 'TWILIO_AUTH_TOKEN', 'TWILIO_FROM_NUMBER'):
        monkeypatch.delenv(key, raising=False)
    app = create_app(tmp_path)
    with TestClient(app) as client, TestClient(create_share_app(tmp_path)) as gateway:
        authenticate(client)
        def forbidden(*args, **kwargs):
            pytest.fail('Preparing a video link must never call a provider')
        app.state.delivery.sender = app.state.calling.sender = app.state.calling.fetcher = forbidden
        store, media = app.state.store, app.state.media
        source = store.clips / 'saved.avi'
        source.write_bytes(b'original-recording')
        cache = media.cache / 'saved.mp4'
        cache.write_bytes(b'test-mp4-range-bytes')
        monkeypatch.setattr(media, 'playback', lambda path: cache)
        with store.connect() as conn:
            conn.execute('''INSERT INTO incidents
                (id,created,mode,score,reasons,signals,clip,camera_name,event_type)
                VALUES (?,?,?,?,?,?,?,?,?)''',
                (INCIDENT, time.time(), 'live', .95, '[]',
                 json.dumps({'live_camera': False, 'source_seconds': 4.5}),
                 source.name, 'Uploaded recording', 'fight'))
        yield client, gateway, app, cache


def add_response(store, state='cancelled'):
    with store.connect() as conn:
        conn.execute('''INSERT INTO response_alerts
            (id,event,created,deadline,state,updated) VALUES (?,?,?,?,?,?)''',
            (INCIDENT, json.dumps(store.incident(INCIDENT)), time.time(), time.time()+10, state, time.time()))


def snapshots(store):
    with store.connect() as conn:
        return {table: conn.execute(f'SELECT * FROM {table}').fetchall()
                for table in ('incidents', 'response_alerts', 'response_calls', 'response_deliveries')}


def create_link(client):
    response = client.post(PATH, json={}, headers=HEADERS)
    assert response.status_code == 200, response.text
    return response.json(), urlsplit(response.json()['url']).path


def test_locationless_recording_prepare_play_and_revoke_without_response(clients):
    client, gateway, app, cache = clients
    before = snapshots(app.state.store)
    assert client.get('/api/evidence/status').json()['configured'] is True
    assert client.get(PATH).json() == {'active_count': 0, 'latest_expires_at': None}
    item, path = create_link(client)
    assert item['incident_id'] == INCIDENT
    assert 3595 < item['expires_at'] - time.time() <= 3600
    token = path.rsplit('/', 1)[1]
    assert len(token) == 43
    with app.state.store.connect() as conn:
        row = conn.execute('SELECT * FROM evidence_shares').fetchone()
    assert row == (hashlib.sha256(token.encode()).hexdigest(), INCIDENT, cache.name, item['expires_at'])
    assert token not in repr(row)
    assert client.get(PATH).json() == {'active_count': 1, 'latest_expires_at': item['expires_at']}
    response = gateway.get(path)
    assert response.status_code == 200 and response.content == cache.read_bytes()
    assert response.headers['cache-control'] == 'no-store'
    assert response.headers['referrer-policy'] == 'no-referrer'
    assert response.headers['x-content-type-options'] == 'nosniff'
    assert gateway.head(path).status_code == 200 and gateway.head(path).content == b''
    ranged = gateway.get(path, headers={'range': 'bytes=0-3'})
    assert ranged.status_code == 206 and ranged.content == cache.read_bytes()[:4]
    # Revocation removes all tokens, including one created by the existing SMS path.
    with app.state.store.connect() as conn:
        conn.execute('INSERT INTO evidence_shares VALUES (?,?,?,?)',
                     (hashlib.sha256(('s'*43).encode()).hexdigest(), INCIDENT, cache.name, time.time()+300))
    assert client.get(PATH).json()['active_count'] == 2
    assert client.delete(PATH, headers=HEADERS).json() == {'revoked_count': 2}
    assert gateway.get(path).status_code == 404
    assert gateway.get('/e/'+'s'*43).status_code == 404
    assert client.get(PATH).json() == {'active_count': 0, 'latest_expires_at': None}
    assert snapshots(app.state.store) == before


def test_api_auth_origin_and_header_and_gateway_has_no_registration(clients):
    client, gateway, app, _ = clients
    client.cookies.clear()
    for method, path in [('GET', '/api/evidence/status'), ('GET', PATH), ('POST', PATH),
                         ('DELETE', PATH), ('POST', PACKAGE)]:
        assert client.request(method, path, headers=HEADERS).status_code == 401
    authenticate(client)
    for method in ('POST', 'DELETE'):
        assert client.request(method, PATH).status_code == 403
        assert client.request(method, PATH, headers={**HEADERS, 'origin': 'https://elsewhere.example'}).status_code == 403
    assert client.get(PATH, headers={'origin': 'https://elsewhere.example'}).status_code == 403
    assert client.post(PACKAGE).status_code == 403
    assert client.post(PACKAGE, headers={**HEADERS, 'origin': 'https://elsewhere.example'}).status_code == 403
    # The final static mount returns 404 for paths without a GET handler.
    assert client.get(PACKAGE).status_code == 404
    assert gateway.get('/health').json() == {'status': 'ok', 'service': 'vdma-evidence'}
    for method, path in [('GET', '/'), ('GET', '/docs'), ('GET', '/api/evidence/status'),
                         ('GET', '/api/incidents'), ('POST', '/api/auth/signup')]:
        response = gateway.request(method, path)
        assert response.status_code == 404
        assert response.headers['cache-control'] == 'no-store'
    assert app.state.evidence.summary(INCIDENT)['active_count'] == 0


def test_gateway_database_uri_escapes_directory_characters(clients, tmp_path):
    client, _, app, cache = clients
    _, path = create_link(client)
    directory = tmp_path / 'copy ? #'
    playback = directory / 'playback'
    playback.mkdir(parents=True)
    (playback / cache.name).write_bytes(cache.read_bytes())
    with app.state.store.connect() as source, sqlite3.connect(directory / 'telemetry.sqlite3') as destination:
        source.backup(destination)
        # Initialize the copied WAL database through its writer. The public
        # gateway must remain read-only rather than recover a backup itself.
        destination.execute('SELECT COUNT(*) FROM evidence_shares').fetchone()
    with TestClient(create_share_app(directory)) as gateway:
        response = gateway.get(path)
        assert response.status_code == 200 and response.content == cache.read_bytes()
        assert gateway.get(path, headers={'range': 'bytes=0-3'}).status_code == 206


@pytest.mark.parametrize('mode,signals', [('demo', {'live_camera': True}), ('live', {}),
    ('live', {'live_camera': 1}), ('live', {'live_camera': 0}), ('live', {'live_camera': 'false'}),
    ('live', None), ('live', {'live_camera': True, 'presentation': True})])
def test_unknown_and_synthetic_sources_cannot_be_shared(clients, mode, signals):
    client, _, app, _ = clients
    with app.state.store.connect() as conn:
        conn.execute('UPDATE incidents SET mode=?,signals=?', (mode, json.dumps(signals)))
    assert client.post(PATH, headers=HEADERS).status_code == 409
    assert client.post(PACKAGE, headers=HEADERS).status_code == 409
    assert client.get(PATH).json()['active_count'] == 0


def test_saved_live_input_is_eligible_and_body_cannot_choose_another_file(clients):
    client, gateway, app, cache = clients
    with app.state.store.connect() as conn:
        conn.execute('UPDATE incidents SET signals=?', (json.dumps({'live_camera': True}),))
    before = snapshots(app.state.store)
    result = client.post(PATH, json={'filename': '../private.txt', 'expires': 99999999999}, headers=HEADERS)
    assert result.status_code == 200
    assert gateway.get(urlsplit(result.json()['url']).path).content == cache.read_bytes()
    assert result.json()['expires_at'] < time.time()+3601
    assert snapshots(app.state.store) == before


def test_existing_response_is_unchanged_by_prepare_and_revoke(clients):
    client, _, app, _ = clients
    add_response(app.state.store, 'requested')
    before = snapshots(app.state.store)
    create_link(client)
    assert client.delete(PATH, headers=HEADERS).status_code == 200
    assert snapshots(app.state.store) == before


def test_hosting_and_saved_clip_are_required_before_conversion(clients, monkeypatch):
    client, _, app, _ = clients
    def unexpected(source):
        pytest.fail('Ineligible evidence must not start conversion')
    monkeypatch.setattr(app.state.media, 'playback', unexpected)
    monkeypatch.setenv('VMD_EVIDENCE_BASE_URL', '')
    assert client.post(PATH, headers=HEADERS).status_code == 503
    monkeypatch.setenv('VMD_EVIDENCE_BASE_URL', 'https://evidence.example')
    with app.state.store.connect() as conn:
        conn.execute('UPDATE incidents SET clip=NULL')
    assert client.post(PATH, headers=HEADERS).status_code == 409
    assert client.get(PATH).json()['active_count'] == 0


@pytest.mark.parametrize('change', ['false_positive', 'cancelled', 'expired', 'missing_incident'])
def test_gateway_and_summary_reject_revoked_or_expired_links(clients, change):
    client, gateway, app, _ = clients
    _, path = create_link(client)
    with app.state.store.connect() as conn:
        if change == 'false_positive':
            conn.execute("UPDATE incidents SET review='false_positive'")
        elif change == 'expired':
            conn.execute('UPDATE evidence_shares SET expires=0')
        elif change == 'missing_incident':
            conn.execute('DELETE FROM incidents')
    if change == 'cancelled':
        add_response(app.state.store)
    response = gateway.get(path)
    assert response.status_code == 404 and response.headers['cache-control'] == 'no-store'
    if change in {'false_positive', 'cancelled'}:
        assert client.post(PATH, headers=HEADERS).status_code == 409
        assert client.get(PATH).json() == {'active_count': 0, 'latest_expires_at': None}
        assert client.delete(PATH, headers=HEADERS).json() == {'revoked_count': 1}
    elif change == 'missing_incident':
        for method in ('GET', 'POST', 'DELETE'):
            assert client.request(method, PATH, headers=HEADERS).status_code == 404
    else:
        assert client.get(PATH).json()['active_count'] == 0


@pytest.mark.parametrize('race', ['review', 'cancel', 'clip', 'hosting'])
def test_encoding_rechecks_incident_and_hosting_before_issuing(clients, monkeypatch, race):
    client, _, app, cache = clients
    def during_encoding(source):
        if race == 'cancel':
            add_response(app.state.store)
        elif race == 'hosting':
            monkeypatch.setenv('VMD_EVIDENCE_BASE_URL', '')
        else:
            with app.state.store.connect() as conn:
                if race == 'review':
                    conn.execute("UPDATE incidents SET review='false_positive'")
                else:
                    conn.execute("UPDATE incidents SET clip='newer.avi'")
        return cache
    monkeypatch.setattr(app.state.media, 'playback', during_encoding)
    response = client.post(PATH, headers=HEADERS)
    assert response.status_code == (503 if race == 'hosting' else 409)
    with app.state.store.connect() as conn:
        assert conn.execute('SELECT COUNT(*) FROM evidence_shares').fetchone()[0] == 0


@pytest.mark.parametrize('bad_source', ['missing.avi', '../private.txt', 'escape.avi'])
def test_source_file_path_cannot_escape_clips(clients, tmp_path, bad_source):
    client, _, app, _ = clients
    private = tmp_path / 'private.txt'
    private.write_bytes(b'must stay private')
    (app.state.store.clips / 'escape.avi').symlink_to(private)
    with app.state.store.connect() as conn:
        conn.execute('UPDATE incidents SET clip=?', (bad_source,))
    assert client.post(PATH, headers=HEADERS).status_code == 404


def test_conversion_failure_does_not_issue_link_or_change_response(clients, monkeypatch):
    client, _, app, _ = clients
    before = snapshots(app.state.store)
    def broken(source):
        raise RuntimeError('Playback conversion failed')
    monkeypatch.setattr(app.state.media, 'playback', broken)
    assert client.post(PATH, headers=HEADERS).status_code == 503
    assert client.get(PATH).json()['active_count'] == 0
    assert snapshots(app.state.store) == before


def test_public_path_and_conversion_cache_cannot_escape(clients, tmp_path, monkeypatch):
    client, gateway, app, cache = clients
    _, path = create_link(client)
    private = tmp_path / 'private.mp4'
    private.write_bytes(b'private bytes')
    for filename in ('../private.mp4', str(private)):
        with app.state.store.connect() as conn:
            conn.execute('UPDATE evidence_shares SET filename=?', (filename,))
        assert gateway.get(path).status_code == 404
    cache.unlink()
    cache.symlink_to(private)
    with app.state.store.connect() as conn:
        conn.execute('UPDATE evidence_shares SET filename=?', (cache.name,))
    assert gateway.get(path).status_code == 404
    assert client.post(PATH, headers=HEADERS).status_code == 404
    monkeypatch.setattr(app.state.media, 'playback', lambda source: private)
    assert client.post(PATH, headers=HEADERS).status_code == 404


@pytest.mark.parametrize('value', ['', 'http://example.com', 'https://u:p@example.com',
    'https://example.com/clip', 'https://example.com/?q=1', 'https://example.com/#token',
    'https://example.com:bad', 'https://example.com:99999', 'https://exa mple.com',
    'https://example.com\\evil', 'https://example.com\n'])
def test_invalid_public_origins(value):
    assert public_origin(value) is None


def test_host_configuration_is_independent_dynamic_and_requires_managed_processes(tmp_path, monkeypatch):
    monkeypatch.delenv('VMD_EVIDENCE_BASE_URL', raising=False)
    assert configuration(tmp_path)['configured'] is False
    managed = tmp_path / 'evidence-host.json'
    managed.write_text(json.dumps({'base_url': 'https://evidence.example/',
                                   'gateway_pid': os.getpid(), 'tunnel_pid': os.getpid()}))
    assert configuration(tmp_path)['base_url'] == 'https://evidence.example'
    managed.write_text(json.dumps({'base_url': 'https://evidence.example', 'gateway_pid': os.getpid()}))
    assert configuration(tmp_path)['configured'] is False
    monkeypatch.setenv('VMD_EVIDENCE_BASE_URL', 'https://override.example/')
    assert configuration(tmp_path)['base_url'] == 'https://override.example'
    monkeypatch.setenv('VMD_EVIDENCE_BASE_URL', 'http://bad.example')
    assert configuration(tmp_path)['configured'] is False
    monkeypatch.delenv('VMD_EVIDENCE_BASE_URL')
    managed.write_text('[]')
    assert configuration(tmp_path)['configured'] is False


@pytest.mark.parametrize('output,expected_birth,ready', [
    ('S Sun Sep 27 14:10:00 2026\n', 'Sun Sep 27 14:10:00 2026', True),
    ('S Sun Sep 27 14:11:00 2026\n', 'Sun Sep 27 14:10:00 2026', False),
    ('Z Sun Sep 27 14:10:00 2026\n', 'Sun Sep 27 14:10:00 2026', False),
    ('', 'Sun Sep 27 14:10:00 2026', False),
    ('S Sun Sep 27 14:10:00 2026\n', None, False),
])
def test_managed_host_birth_rejects_reused_pid_and_zombie(tmp_path, monkeypatch, output, expected_birth, ready):
    import vmd.evidence as module
    monkeypatch.delenv('VMD_EVIDENCE_BASE_URL', raising=False)
    (tmp_path / 'evidence-host.json').write_text(json.dumps({
        'base_url': 'https://evidence.example', 'gateway_pid': 12345, 'tunnel_pid': 12346,
        'gateway_birth': expected_birth, 'tunnel_birth': expected_birth}))
    monkeypatch.setattr(module.os, 'kill', lambda pid, signal: None)
    calls = []
    def ps(args, **kwargs):
        calls.append(args)
        assert kwargs == {'capture_output': True, 'text': True, 'timeout': 2}
        return SimpleNamespace(stdout=output, returncode=0)
    monkeypatch.setattr(module.subprocess, 'run', ps)
    status = configuration(tmp_path)
    assert status['configured'] is ready
    if ready:
        assert [call[2] for call in calls] == ['12345', '12346']
        assert 'processes are running' in status['message']
    else:
        assert status['base_url'] is None


def test_managed_host_process_inspection_timeout_fails_closed(tmp_path, monkeypatch):
    import vmd.evidence as module
    monkeypatch.delenv('VMD_EVIDENCE_BASE_URL', raising=False)
    (tmp_path / 'evidence-host.json').write_text(json.dumps({
        'base_url': 'https://evidence.example', 'gateway_pid': 12345, 'tunnel_pid': 12346,
        'gateway_birth': 'Sun Sep 27 14:10:00 2026', 'tunnel_birth': 'Sun Sep 27 14:10:00 2026'}))
    monkeypatch.setattr(module.os, 'kill', lambda pid, signal: None)
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired('ps', 2)
    monkeypatch.setattr(module.subprocess, 'run', timeout)
    assert configuration(tmp_path)['configured'] is False


@pytest.mark.parametrize('endpoint,url_field', [(PATH, 'url'), (PACKAGE, 'video_url')])
def test_actual_h264_preparation_is_playable_without_dispatch(clients, monkeypatch, endpoint, url_field):
    client, gateway, app, _ = clients
    from vmd.media import MediaLibrary
    monkeypatch.setattr(app.state.media, 'playback', MediaLibrary.playback.__get__(app.state.media))
    source = app.state.store.clips / 'saved.avi'
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 8, (64, 48))
    assert writer.isOpened()
    for value in (30, 70, 110, 150):
        writer.write(np.full((48, 64, 3), value, np.uint8))
    writer.release()
    before = snapshots(app.state.store)
    result = client.post(endpoint, headers=HEADERS)
    assert result.status_code == 200, result.text
    path = urlsplit(result.json()[url_field]).path
    response = gateway.get(path)
    assert response.status_code == 200 and response.headers['content-type'] == 'video/mp4'
    with app.state.store.connect() as conn:
        filename = conn.execute('SELECT filename FROM evidence_shares').fetchone()[0]
    video = cv2.VideoCapture(str(app.state.media.cache / filename))
    try:
        assert video.isOpened() and video.read()[0]
        assert video.get(cv2.CAP_PROP_FRAME_COUNT) == 4
    finally:
        video.release()
    assert snapshots(app.state.store) == before


@pytest.mark.parametrize('live', [True, False])
def test_event_package_saved_metadata_and_playable_link_without_dispatch(clients, live):
    client, gateway, app, cache = clients
    location = {'place': 'Example north gate', 'latitude': 28.6, 'longitude': 77.2,
                'source': 'browser', 'captured_at': 1700000000, 'accuracy_meters': 15,
                'private_extra': 'sensitive-location-field'}
    with app.state.store.connect() as conn:
        conn.execute('UPDATE incidents SET created=?,signals=?', (1700000100, json.dumps({
            'live_camera': live, 'location': location, 'source_seconds': 4.5,
            'private_camera_url': 'rtsp://user:secret@private.invalid/video'})))
    before = snapshots(app.state.store)
    response = client.post(PACKAGE, json={'place': 'Spoofed place', 'url': 'https://wrong.example'}, headers=HEADERS)
    assert response.status_code == 200, response.text
    item = response.json()
    assert set(item) == {'incident_id', 'type', 'camera_id', 'camera_name', 'detected_at',
        'occurred_at', 'source_kind', 'source_seconds', 'location', 'review_status',
        'place', 'video_url', 'video_expires_at'}
    assert item['incident_id'] == INCIDENT and item['type'] == 'fight'
    assert item['detected_at'] == '2023-11-14T22:15:00.000Z' and item['occurred_at'] is None
    assert item['place'] == 'Example north gate'
    assert item['location'] == {key: value for key, value in location.items()
                                if key not in {'private_extra', 'captured_at'}} | {
        'verified': False, 'captured_at': '2023-11-14T22:13:20.000Z'}
    assert item['source_kind'] == ('live_camera' if live else 'recording')
    assert item['source_seconds'] == (None if live else 4.5)
    from datetime import datetime
    assert 3595 < datetime.fromisoformat(item['video_expires_at'].replace('Z', '+00:00')).timestamp() - time.time() <= 3600
    assert response.headers['cache-control'] == 'no-store'
    assert all(secret not in response.text for secret in ('rtsp:', 'private.invalid', 'secret', 'saved.avi', 'private_extra', 'Spoofed'))
    path = urlsplit(item['video_url']).path
    assert gateway.get(path).content == cache.read_bytes()
    assert gateway.get(path, headers={'range': 'bytes=0-3'}).status_code == 206
    assert client.delete(PATH, headers=HEADERS).json() == {'revoked_count': 1}
    assert gateway.get(path).status_code == 404
    assert snapshots(app.state.store) == before


def test_event_package_locationless_recording_and_final_snapshot(clients, monkeypatch):
    client, _, app, cache = clients
    first = client.post(PACKAGE, headers=HEADERS).json()
    assert first['location'] is None and first['place'] is None
    assert first['source_kind'] == 'recording' and first['source_seconds'] == 4.5
    def update_metadata(source):
        with app.state.store.connect() as conn:
            conn.execute('UPDATE incidents SET event_type=?,signals=?', ('snatching_detected', json.dumps({
                'live_camera': False, 'source_seconds': 6.5,
                'location': {'place': 'Recorded gate', 'latitude': 28.6, 'longitude': 77.2}})))
        return cache
    monkeypatch.setattr(app.state.media, 'playback', update_metadata)
    second = client.post(PACKAGE, headers=HEADERS).json()
    assert second['type'] == 'snatching_detected' and second['source_seconds'] == 6.5
    assert second['place'] == 'Recorded gate'
    assert second['video_url'] != first['video_url']


@pytest.mark.parametrize('failure,status', [('missing', 404), ('clip', 404),
    ('false_positive', 409), ('cancelled', 409), ('hosting', 503), ('encoding', 503)])
def test_event_package_failure_never_creates_link_or_response(clients, monkeypatch, failure, status):
    client, _, app, _ = clients
    if failure == 'missing':
        with app.state.store.connect() as conn:
            conn.execute('DELETE FROM incidents')
    elif failure == 'clip':
        (app.state.store.clips / 'saved.avi').unlink()
    elif failure == 'false_positive':
        with app.state.store.connect() as conn:
            conn.execute("UPDATE incidents SET review='false_positive'")
    elif failure == 'cancelled':
        add_response(app.state.store)
    elif failure == 'hosting':
        monkeypatch.setenv('VMD_EVIDENCE_BASE_URL', '')
    else:
        def fail(source):
            raise RuntimeError('Video conversion failed')
        monkeypatch.setattr(app.state.media, 'playback', fail)
    before = snapshots(app.state.store)
    result = client.post(PACKAGE, headers=HEADERS)
    assert result.status_code == status, result.text
    with app.state.store.connect() as conn:
        assert conn.execute('SELECT COUNT(*) FROM evidence_shares').fetchone()[0] == 0
    assert snapshots(app.state.store) == before
