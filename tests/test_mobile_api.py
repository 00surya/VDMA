import base64
import json
import sqlite3
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from conftest import TEST_CENTRE
from vmd.api import create_app
from vmd.centre import Centre, CentreSignup
from vmd.mobile import create_mobile_app
from vmd.storage import Store


@pytest.fixture
def saved(tmp_path):
    store = Store(tmp_path)
    centre = Centre(store)
    centre.signup(CentreSignup(**TEST_CENTRE))
    try:
        yield store, centre
    finally:
        store.close()


@pytest.fixture
def client(saved):
    with TestClient(create_mobile_app(saved[0].directory), base_url='https://localhost') as client:
        yield client


def login(client):
    result = client.post('/api/v1/auth/token', json={'password': TEST_CENTRE['password']})
    assert result.status_code == 200, result.text
    assert result.json()['scope'] == 'incidents:read'
    assert result.json()['expires_in'] == 3600
    assert 'set-cookie' not in result.headers
    return {'Authorization': 'Bearer ' + result.json()['access_token']}


def insert(store, identity='incident-1', created=1700000000, *, live=True, mode='live',
           review='unreviewed', location=None, extra=None):
    signals = {'live_camera': live, 'location': location,
               'private_stream': 'rtsp://secret:password@private.invalid/video',
               'source_seconds': 12.5, **(extra or {})}
    with store.connect() as conn:
        conn.execute('''INSERT INTO incidents
            (id,created,mode,score,reasons,signals,review,camera_id,camera_name,event_type,clip)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)''', (identity, created, mode, .99,
            '["private description"]', json.dumps(signals), review, 'camera-1', 'Test entrance',
            'gun_detected', '/private/test/evidence.avi'))


def test_token_required_separate_from_dashboard_and_scoped_to_installation(saved, client, tmp_path):
    store, centre = saved
    insert(store)
    cookie = centre.session()
    for path in ['/api/v1/incidents', '/api/v1/incidents/incident-1']:
        response = client.get(path, headers={'Cookie': f'vmd_session={cookie}'})
        assert response.status_code == 401
        assert response.headers['cache-control'] == 'no-store'
        assert response.headers['www-authenticate'] == 'Bearer'
    wrong = client.post('/api/v1/auth/token', json={'password': 'wrong-password'})
    assert wrong.status_code == 401 and 'wrong-password' not in wrong.text
    headers = login(client)
    assert not centre.authenticated(headers['Authorization'].split(' ', 1)[1])
    assert client.get('/api/v1/incidents', headers=headers).status_code == 200
    # The token is process-local and cannot grant another installation or the dashboard access.
    with TestClient(create_mobile_app(store.directory), base_url='https://localhost') as other:
        assert other.get('/api/v1/incidents', headers=headers).status_code == 401
    with TestClient(create_app(tmp_path / 'other-dashboard')) as dashboard:
        assert dashboard.get('/api/incidents', headers=headers).status_code == 401
    assert client.post('/api/v1/auth/logout', headers=headers).status_code == 204
    assert client.get('/api/v1/incidents', headers=headers).status_code == 401


def test_payload_uses_saved_location_and_detection_clock_only(saved, client):
    store, _ = saved
    location = {'place': 'Fictional gate', 'latitude': 28.6, 'longitude': 77.2,
                'source': 'browser', 'accuracy_meters': 20, 'captured_at': 1699999900,
                'private_extra': 'do not export', 'verified': True}
    insert(store, location=location)
    row = client.get('/api/v1/incidents/incident-1', headers=login(client)).json()
    assert set(row) == {'incident_id', 'type', 'camera_id', 'camera_name', 'detected_at',
                        'occurred_at', 'source_kind', 'source_seconds', 'location', 'review_status'}
    assert row['detected_at'] == '2023-11-14T22:13:20.000Z'
    assert row['occurred_at'] is None and row['source_seconds'] is None
    assert row['location'] == {'place': 'Fictional gate', 'latitude': 28.6, 'longitude': 77.2,
                               'source': 'browser', 'verified': False, 'accuracy_meters': 20,
                               'captured_at': '2023-11-14T22:11:40.000Z'}
    assert row['source_kind'] == 'live_camera'
    assert not any(secret in json.dumps(row) for secret in ('private', 'password', 'rtsp', 'avi', 'authorities'))


def test_source_filters_reviews_recordings_and_corrupt_provenance(saved, client):
    store, _ = saved
    insert(store, 'physical')
    insert(store, 'recording', live=False)
    insert(store, 'false-positive', review='false_positive')
    insert(store, 'synthetic', mode='demo')
    insert(store, 'presentation', extra={'presentation': True})
    insert(store, 'unknown', live=None)
    insert(store, 'boolean-number', live=1)
    insert(store, 'future', created=time.time() + 86400)
    insert(store, 'corrupt')
    with store.connect() as conn:
        conn.execute("UPDATE incidents SET signals='broken' WHERE id='corrupt'")
    headers = login(client)
    listing = client.get('/api/v1/incidents', headers=headers).json()
    assert [r['incident_id'] for r in listing['incidents']] == ['physical']
    assert listing['incidents'][0]['location'] is None
    records = client.get('/api/v1/incidents?source=recording', headers=headers).json()['incidents']
    assert records[0]['source_kind'] == 'recording' and records[0]['source_seconds'] == 12.5
    assert records[0]['occurred_at'] is None and records[0]['location'] is None
    all_rows = client.get('/api/v1/incidents?source=all&include_false_positives=true', headers=headers).json()['incidents']
    assert {r['incident_id'] for r in all_rows} == {'physical', 'recording', 'false-positive'}
    for identity in ('synthetic', 'presentation', 'unknown', 'boolean-number', 'corrupt', 'future', 'absent'):
        assert client.get('/api/v1/incidents/' + identity, headers=headers).status_code == 404
    store.review('physical', 'false_positive')
    assert client.get('/api/v1/incidents', headers=headers).json()['incidents'] == []
    assert client.get('/api/v1/incidents/physical', headers=headers).json()['review_status'] == 'false_positive'


@pytest.mark.parametrize('location', [None, {}, [], {'latitude': True, 'longitude': 1},
    {'latitude': 91, 'longitude': 1}, {'latitude': 1, 'longitude': -181},
    {'latitude': '28.6', 'longitude': 1}])
def test_unknown_or_invalid_locations_are_not_invented(saved, client, location):
    insert(saved[0], location=location)
    assert client.get('/api/v1/incidents/incident-1', headers=login(client)).json()['location'] is None


def test_legacy_location_has_unknown_provenance_and_unverified_recording_location(saved, client):
    insert(saved[0], live=False, location={'latitude': 0, 'longitude': 0, 'source': 'untrusted',
                                         'accuracy_meters': -1, 'captured_at': -1})
    row = client.get('/api/v1/incidents/incident-1', headers=login(client)).json()
    assert row['location']['source'] == 'unknown' and row['location']['verified'] is False
    assert row['location']['latitude'] == row['location']['longitude'] == 0
    assert row['location']['captured_at'] is None and row['location']['accuracy_meters'] is None


def test_pagination_exceeds_dashboard_cap_and_preserves_timestamp_ties(saved, client):
    for index in range(123):
        insert(saved[0], f'event-{index:03}')
    insert(saved[0], 'older', created=1699999999)
    headers, identities, cursor = login(client), [], None
    while True:
        params = {'limit': 40, 'since': '2023-11-14T22:13:20Z'}
        if cursor:
            params['cursor'] = cursor
        response = client.get('/api/v1/incidents', params=params, headers=headers)
        assert response.status_code == 200, response.text
        data = response.json()
        identities.extend(row['incident_id'] for row in data['incidents'])
        cursor = data['next_cursor']
        if not cursor:
            break
    assert identities == [f'event-{index:03}' for index in reversed(range(123))]


@pytest.mark.parametrize('params', [
    {'limit': 0}, {'limit': 101}, {'limit': 'no'}, {'source': 'demo'},
    {'since': '2023-11-14T00:00:00'}, {'since': '1960-01-01T00:00:00Z'},
    {'cursor': 'not json'}, {'cursor': 'a' * 513},
    {'cursor': base64.urlsafe_b64encode(json.dumps([10**400, 'a']).encode()).decode()},
    {'cursor': base64.urlsafe_b64encode(b'[1,"SQL\' injection"]').decode()},
])
def test_query_validation(saved, client, params):
    assert client.get('/api/v1/incidents', params=params, headers=login(client)).status_code == 422


def test_rate_limits_and_expiry(saved, client, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr('vmd.mobile.time', SimpleNamespace(monotonic=lambda: clock[0], time=lambda: 1700000000 + clock[0]))
    headers = login(client)
    for _ in range(120):
        assert client.get('/api/v1/incidents', headers=headers).status_code == 200
    limited = client.get('/api/v1/incidents', headers=headers)
    assert limited.status_code == 429 and limited.headers['retry-after'] == '60'
    clock[0] += 60
    assert client.get('/api/v1/incidents', headers=headers).status_code == 200
    clock[0] += 3540
    assert client.get('/api/v1/incidents', headers=headers).status_code == 401
    for _ in range(5):
        assert client.post('/api/v1/auth/token', json={'password': 'wrong'}).status_code == 401
    assert client.post('/api/v1/auth/token', json={'password': TEST_CENTRE['password']}).status_code == 429
    clock[0] += 60
    assert login(client)


def test_password_change_revokes_mobile_session(saved, client):
    headers = login(client)
    with saved[0].connect() as conn:
        conn.execute("UPDATE centre SET password_hash=? WHERE id=1", ('0' * 128,))
    assert client.get('/api/v1/incidents', headers=headers).status_code == 401


def test_host_transport_validation_no_admin_and_no_input_echo(saved, client):
    headers = login(client)
    assert client.get('/api/v1/incidents', headers={**headers, 'host': 'evil.example'}).status_code == 400
    assert client.get('/api/v1/incidents', headers={**headers, 'origin': 'https://evil.example'}).status_code == 403
    for path in ['/', '/docs', '/openapi.json', '/api/cameras', '/api/centre', '/api/incidents']:
        assert client.get(path, headers=headers).status_code == 404
    assert client.post('/api/v1/incidents', headers=headers).status_code == 405
    assert client.post('/api/v1/incidents/incident-1/response', json={'action': 'dispatch'}, headers=headers).status_code == 404
    malformed = client.post('/api/v1/auth/token', json={'password': {'secret': 'do-not-echo'}})
    assert malformed.status_code == 422 and 'do-not-echo' not in malformed.text
    oversized = client.post('/api/v1/auth/token', content=b'x' * 4097)
    assert oversized.status_code == 413 and 'x' * 100 not in oversized.text
    with TestClient(create_mobile_app(saved[0].directory), base_url='http://localhost', client=('192.0.2.1', 123)) as remote:
        assert remote.post('/api/v1/auth/token', json={'password': TEST_CENTRE['password']}).status_code == 403
    with TestClient(create_mobile_app(saved[0].directory), base_url='http://localhost', client=('127.0.0.1', 123)) as local:
        assert login(local)


def test_gateway_does_not_write_database_or_initialize_missing_data(saved, client, tmp_path):
    store, _ = saved
    insert(store)
    with store.connect() as conn:
        before = '\n'.join(conn.iterdump())
    headers = login(client)
    assert client.get('/api/v1/incidents', headers=headers).status_code == 200
    client.post('/api/v1/auth/logout', headers=headers)
    with store.connect() as conn:
        assert '\n'.join(conn.iterdump()) == before
    missing = tmp_path / 'not-created'
    with TestClient(create_mobile_app(missing), base_url='https://localhost') as empty:
        response = empty.post('/api/v1/auth/token', json={'password': TEST_CENTRE['password']})
        assert response.status_code == 503 and str(missing) not in response.text
    assert not missing.exists()


def test_escaped_database_path_and_exact_host_configuration(saved, tmp_path):
    store, _ = saved
    path = tmp_path / 'copy ? #'
    path.mkdir()
    with store.connect() as source, sqlite3.connect(path / 'telemetry.sqlite3') as target:
        source.backup(target)
        # Finish the copied WAL database's initialization as its writer. A read-only
        # gateway must not attempt recovery or switch to writable mode on its behalf.
        target.execute('SELECT COUNT(*) FROM centre').fetchone()
    with TestClient(create_mobile_app(path, allowed_hosts=['mobile.example']), base_url='https://mobile.example') as client:
        assert client.get('/api/v1/incidents', headers=login(client)).status_code == 200
    for hosts in ([], ['*'], ['*.example'], ['https://example']):
        with pytest.raises(ValueError):
            create_mobile_app(path, allowed_hosts=hosts)
