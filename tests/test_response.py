import hashlib
import io
import json
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import TEST_CENTRE, authenticate
from vmd.alerts import AlertAgent
from vmd.api import create_app, StartRequest
from vmd.centre import Centre, CentreSignup
from vmd.delivery import Delivery
from vmd.media import MediaLibrary
from vmd.share import create_share_app
from vmd.storage import Store

HEADERS = {'x-vmd-client': 'dashboard'}


def event(incident_id='a'*32, created=None, kind='knife_detected', live=True):
    return {'id': incident_id, 'created': time.time() if created is None else created,
            'mode': 'live', 'event_type': kind, 'camera_id': 'camera', 'camera_name': 'Entrance',
            'score': .85, 'reasons': ['Review required'],
            'signals': {'live_camera': live, 'location': {'place': 'Test gate', 'latitude': 28.61, 'longitude': 77.21}}}


def frames():
    jpeg = cv2.imencode('.jpg', np.full((48,64,3), 80, np.uint8))[1].tobytes()
    return [(0, jpeg), (1, jpeg)]


def test_countdown_dedup_ack_review_and_restart(tmp_path):
    store = Store(tmp_path)
    agent = AlertAgent(store, background=False)
    try:
        first = event(created=100)
        agent.register(first)
        agent.register(first)
        agent.tick(109.9)
        assert agent.get(first['id'])['state'] == 'pending'
        # State/deadline survive a service restart.
        restored = AlertAgent(store, background=False)
        assert restored.get(first['id'])['deadline'] == 110
        with ThreadPoolExecutor(2) as pool:
            list(pool.map(lambda _: restored.tick(110), range(2)))
        assert restored.get(first['id'])['state'] == 'requested'
        assert restored.get(first['id'])['trigger'] == 'automatic'
        assert restored.get(first['id'])['updated'] == 110
        restored.act(first['id'], 'acknowledge')
        restored.tick(200)
        assert restored.get(first['id'])['state'] == 'acknowledged'
        second = event('b'*32, created=200, kind='fight')
        store._incident(second, frames())
        restored.register(second)
        assert restored.review(second['id'], 'false_positive')
        restored.tick(220)
        assert restored.get(second['id'])['state'] == 'cancelled'
        with pytest.raises(ValueError): restored.act(second['id'], 'dispatch')
        for item in [event('c'*32, live=False), {**event('d'*32), 'mode': 'demo'}, event('e'*32, kind='gun_detected')]:
            restored.register(item)
            assert restored.get(item['id']) is None
        restored.close()
    finally:
        agent.close()
        store.close()


@pytest.mark.parametrize('action', ['dispatch', 'false_positive'])
def test_first_handling_time_survives_followups_restart_and_next_incident(tmp_path, monkeypatch, action):
    clock = [110.0]
    monkeypatch.setattr('vmd.alerts.time', SimpleNamespace(time=lambda: clock[0]))
    monkeypatch.setattr('vmd.storage.time', SimpleNamespace(time=lambda: clock[0]))
    store = Store(tmp_path)
    agent = AlertAgent(store, background=False)
    try:
        first = event(created=100)
        store._incident(first, frames()); agent.register(first)
        saved_clip = store.incident(first['id'])['clip']
        clip_bytes = (store.clips / saved_clip).read_bytes()
        if action == 'dispatch':
            response = agent.act(first['id'], 'dispatch')
            assert response['incident_saved'] and response['clip_ready']
            for following in ['hospital', 'call_hospital', 'dispatch', 'acknowledge']:
                clock[0] += 20
                agent.act(first['id'], following, True if following == 'hospital' else None)
        else:
            assert agent.review(first['id'], 'false_positive')
            clock[0] += 40
            assert agent.review(first['id'], 'false_positive')
            assert store.review(first['id'], 'false_positive')
        assert agent.get(first['id'])['handled_at'] == 110
        assert store.incident(first['id'])['handled_at'] == 110
        agent.close()
        agent = AlertAgent(store, background=False)
        assert agent.get(first['id'])['handled_at'] == 110
        second = event('b'*32, created=clock[0])
        store._incident(second, frames()); agent.register(second)
        assert store.incident(second['id'])['handled_at'] is None
        assert store.incident(second['id'])['review'] == 'unreviewed'
        assert agent.get(second['id'])['state'] == 'pending'
        assert agent.get(second['id'])['handled_at'] is None
        assert len(store.incidents()) == 2
        assert store.incident(first['id'])['clip'] == saved_clip
        assert (store.clips / saved_clip).read_bytes() == clip_bytes
    finally:
        agent.close(); store.close()


@pytest.mark.parametrize('save_first', [True, False])
def test_automatic_handling_time_covers_both_evidence_save_races(tmp_path, save_first):
    store = Store(tmp_path)
    agent = AlertAgent(store, background=False)
    try:
        item = event(created=100)
        agent.register(item)
        if save_first:
            store._incident(item, frames())
        agent.tick(109.9)
        assert agent.get(item['id'])['handled_at'] is None
        agent.tick(145)  # A delayed timer retains the original dispatch deadline.
        response = agent.describe(agent.get(item['id']))
        assert response['handled_at'] == 110
        assert response['incident_saved'] is save_first
        if not save_first:
            store._incident(item, frames())
        agent.tick(200)
        assert store.incident(item['id'])['handled_at'] == 110
        response = agent.describe(agent.get(item['id']))
        assert response['incident_saved'] and response['clip_ready']
        assert response['handled_at'] == 110
    finally:
        agent.close(); store.close()


@pytest.mark.parametrize('reviewer', ['agent', 'store'])
def test_gun_false_positive_without_response_gets_durable_handling_time(tmp_path, monkeypatch, reviewer):
    clock = [120.0]
    monkeypatch.setattr('vmd.alerts.time', SimpleNamespace(time=lambda: clock[0]))
    monkeypatch.setattr('vmd.storage.time', SimpleNamespace(time=lambda: clock[0]))
    store = Store(tmp_path)
    agent = AlertAgent(store, background=False)
    try:
        item = event(created=100, kind='gun_detected')
        store._incident(item, []); agent.register(item)
        target = agent if reviewer == 'agent' else store
        assert target.review(item['id'], 'false_positive')
        clock[0] = 200
        assert target.review(item['id'], 'false_positive')
        assert agent.get(item['id']) is None
        saved = store.incident(item['id'])
        assert saved['review'] == 'false_positive' and saved['handled_at'] == 120
        assert saved['clip_error']  # A failed clip does not erase the saved incident.
    finally:
        agent.close(); store.close()


@pytest.mark.parametrize('action', ['acknowledge', 'confirmed'])
def test_acknowledging_or_confirming_without_dispatch_does_not_start_camera_reset(tmp_path, action):
    store = Store(tmp_path)
    agent = AlertAgent(store, background=False)
    try:
        item = event(); store._incident(item, frames()); agent.register(item)
        if action == 'confirmed':
            assert agent.review(item['id'], action)
        else:
            agent.act(item['id'], action)
        agent.close()
        agent = AlertAgent(store, background=False)
        assert agent.get(item['id'])['state'] == 'acknowledged'
        assert agent.get(item['id'])['handled_at'] is None
        assert store.incident(item['id'])['handled_at'] is None
    finally:
        agent.close(); store.close()


def test_legacy_handling_migration_keeps_original_known_times_and_unhandled_incidents(tmp_path):
    store = Store(tmp_path)
    cases = [
        ('automatic', 'acknowledged', 'automatic', 'unreviewed', 400, 300, 110),
        ('manual', 'requested', 'manual', 'unreviewed', 500, 150, 150),
        ('false-positive', 'cancelled', None, 'false_positive', 210, None, 210),
        ('acknowledged', 'acknowledged', None, 'confirmed', 220, None, None),
        ('pending', 'pending', None, 'unreviewed', 100, None, None),
        ('gun', None, None, 'false_positive', 100, None, 100),
    ]
    agent = None
    try:
        with store.connect() as conn:
            conn.execute('''CREATE TABLE response_alerts (
                id TEXT PRIMARY KEY, event TEXT NOT NULL, created REAL NOT NULL,
                deadline REAL NOT NULL, state TEXT NOT NULL, hospital INTEGER NOT NULL DEFAULT 0,
                trigger TEXT, updated REAL NOT NULL, voice_requested_at REAL
            )''')
        for identity, state, trigger, review, updated, requested, _ in cases:
            item = event(identity, created=100, kind='gun_detected' if identity == 'gun' else 'fight')
            store._incident(item, [])
            with store.connect() as conn:
                conn.execute('UPDATE incidents SET review=? WHERE id=?', (review, identity))
                if state:
                    conn.execute('''INSERT INTO response_alerts
                        (id,event,created,deadline,state,trigger,updated,voice_requested_at) VALUES (?,?,100,110,?,?,?,?)''',
                        (identity, json.dumps(item), state, trigger, updated, requested))
        agent = AlertAgent(store, background=False)
        for identity, state, _, _, _, _, expected in cases:
            assert store.incident(identity)['handled_at'] == expected
            if state:
                assert agent.get(identity)['handled_at'] == expected
        agent.close()
        agent = AlertAgent(store, background=False)
        assert store.incident('automatic')['handled_at'] == 110
        assert store.incident('false-positive')['review'] == 'false_positive'
    finally:
        if agent:
            agent.close()
        store.close()


def test_signup_signin_protects_api_and_contacts(tmp_path):
    app = create_app(tmp_path)
    with TestClient(app) as client:
        assert client.get('/api/cameras').status_code == 401
        assert client.get('/api/auth/status').json()['centre'] is None
        assert client.post('/api/auth/signup', json=TEST_CENTRE).status_code == 403
        assert client.post('/api/auth/signup', json={**TEST_CENTRE, 'authorities': []}, headers=HEADERS).status_code == 422
        authenticate(client)
        assert client.get('/api/cameras').status_code == 200
        assert client.get('/api/centre').json()['centre']['authorities'][0]['name'] == 'Test authority'
        assert client.post('/api/auth/signup', json=TEST_CENTRE, headers=HEADERS).status_code == 409
        assert client.post('/api/auth/logout', headers=HEADERS).status_code == 200
        assert client.get('/api/centre').status_code == 401
        assert client.post('/api/auth/login', json={'password': 'incorrect'}, headers=HEADERS).status_code == 401
        authenticate(client)
        for location in ({'place': '', 'latitude': 0, 'longitude': 0}, {'place': 'Gate', 'latitude': 91, 'longitude': 0}):
            assert client.post('/api/cameras/missing/location', json=location, headers=HEADERS).status_code == 422
        with app.state.store.connect() as conn:
            row = conn.execute('SELECT password_hash FROM centre').fetchone()
            assert row[0] != TEST_CENTRE['password']
            conn.execute('UPDATE centre_sessions SET expires=0')
        assert client.get('/api/incidents').status_code == 401


def test_api_manual_dispatch_package_and_recording_review(tmp_path):
    app = create_app(tmp_path)
    with TestClient(app) as client:
        authenticate(client)
        # Exercise the request workflow with legacy SMS enabled; no provider is configured.
        client.post('/api/centre', json={**TEST_CENTRE, 'messaging_enabled': True}, headers=HEADERS)
        item = event()
        app.state.store.enqueue('incident', (item, frames()))
        app.state.store.jobs.join()
        response = client.get('/api/response-alerts').json()
        assert response['alerts'][0]['state'] == 'pending'
        assert not response['messaging']['configured']
        route = f"/api/incidents/{item['id']}/response"
        for _ in range(2):
            assert client.post(route, json={'action': 'dispatch', 'hospital': True}, headers=HEADERS).status_code == 200
        assert app.state.alerts.get(item['id'])['hospital']
        package = client.get(route+'-package')
        assert package.status_code == 200
        with zipfile.ZipFile(io.BytesIO(package.content)) as archive:
            manifest = json.loads(archive.read('incident.json'))
            assert manifest['recipients'] == ['authority', 'hospital']
            assert manifest['location']['place'] == 'Test gate'
            assert len(archive.read(manifest['clip'])) > 0
        recording = event('c'*32, live=False)
        recording['signals']['source_seconds'] = 12.5
        recording['signals']['location'] = None
        app.state.store._incident(recording, frames())
        app.state.alerts.register(recording)
        assert app.state.alerts.get(recording['id']) is None
        route = '/api/incidents/'+recording['id']+'/response'
        assert client.post(route, json={'action': 'dispatch'}, headers=HEADERS).status_code == 200
        assert app.state.alerts.get(recording['id'])['trigger'] == 'manual'
        with zipfile.ZipFile(io.BytesIO(client.get(route+'-package').content)) as archive:
            manifest = json.loads(archive.read('incident.json'))
            assert manifest['source_kind'] == 'recording'
            assert manifest['source_seconds'] == 12.5
            assert manifest['location'] is None
            assert 'capture time and location unverified' in manifest['assessment']


@pytest.mark.parametrize('mode, provenance', [('demo', False), ('demo', True), ('presentation', False),
                                            ('live', None), ('live', 0), ('live', 1), ('live', 'false')])
def test_unknown_or_synthetic_source_cannot_manually_dispatch(tmp_path, mode, provenance):
    store = Store(tmp_path)
    agent = AlertAgent(store, background=False)
    try:
        item = event(live=provenance)
        item['mode'] = mode
        if provenance is None:
            del item['signals']['live_camera']
        store._incident(item, frames())
        agent.register(item)
        for action in ('dispatch', 'call_hospital'):
            with pytest.raises(ValueError, match='saved camera or recording'):
                agent.act(item['id'], action)
        agent.tick(time.time()+100)
        assert agent.get(item['id']) is None
    finally:
        agent.close(); store.close()


def test_recording_manual_request_upgrade_and_restart_never_start_countdown(tmp_path, monkeypatch):
    clock = [100000.0]
    monkeypatch.setattr('vmd.alerts.time', SimpleNamespace(time=lambda: clock[0]))
    store = Store(tmp_path)
    agent = AlertAgent(store, background=False)
    try:
        item = event(created=100, kind='possible_fight', live=False)
        store._incident(item, frames()); agent.register(item)
        agent.tick(clock[0])
        assert agent.get(item['id']) is None
        response = agent.act(item['id'], 'dispatch')
        assert response['trigger'] == 'manual'
        assert response['voice_requested_at'] == clock[0]
        assert response['handled_at'] == clock[0]
        item = {**item, 'event_type': 'fight'}
        store._incident(item, frames()); agent.register(item)
        agent.tick(clock[0]+100)
        upgraded = agent.get(item['id'])
        assert upgraded['event']['event_type'] == 'fight'
        for field in ('deadline', 'trigger', 'state', 'handled_at', 'voice_requested_at'):
            assert upgraded[field] == response[field]
        agent.close(); agent = AlertAgent(store, background=False)
        clock[0] += 60
        agent.act(item['id'], 'call_hospital')
        assert agent.get(item['id'])['handled_at'] == response['handled_at']
        assert agent.get(item['id'])['hospital']
        assert agent.review(item['id'], 'false_positive')
        for action in ('dispatch', 'call_hospital'):
            with pytest.raises(ValueError, match='false-positive'):
                agent.act(item['id'], action)
        next_item = event('b'*32, live=False)
        store._incident(next_item, frames()); agent.register(next_item)
        assert agent.get(next_item['id']) is None
        assert len(store.incidents()) == 2
    finally:
        agent.close(); store.close()


@pytest.mark.parametrize('live, located', [(True, True), (False, True), (False, False)])
def test_delivery_recipient_dedup_gateway_expiry_and_false_positive(tmp_path, monkeypatch, live, located):
    import vmd.delivery as module
    monkeypatch.setattr(module, 'configuration', lambda: {'ready': True, 'base': 'https://evidence.example'})
    store = Store(tmp_path)
    agent = AlertAgent(store, background=False)
    centre = Centre(store)
    centre.signup(CentreSignup(**TEST_CENTRE, messaging_enabled=True))
    media = MediaLibrary(store)
    clip = media.cache/'fixture.mp4'
    clip.write_bytes(b'fake-mp4-for-range-test')
    monkeypatch.setattr(media, 'playback', lambda source: clip)
    messages = []
    def send(phone, body):
        messages.append((phone, body))
        return {'status': 'accepted', 'sid': 'SM'+'a'*32, 'error': None}
    delivery = Delivery(store, centre, agent, media, background=False, sender=send)
    try:
        item = event(live=live)
        if not located:
            item['signals']['location'] = None
        store._incident(item, frames())
        agent.register(item)
        agent.act(item['id'], 'dispatch', True)
        delivery.process_once()
        delivery.process_once()
        assert len(messages) == 2
        from vmd.centre import CentreDetails
        centre.update(CentreDetails(**{**TEST_CENTRE, 'messaging_enabled': True,
            'authorities': [{'name': 'Replacement', 'phone': '+12025550103'}]}))
        delivery.process_once()
        assert len(messages) == 2  # Contact edits do not reroute an existing request.
        if located:
            assert all('28.61' in body and 'Test gate' in body for _, body in messages)
        else:
            assert all('Recording location unknown' in body and 'maps.google' not in body
                       and '28.61' not in body for _, body in messages)
        assert all(('Recorded-footage review' in body) is (not live) for _, body in messages)
        token = messages[0][1].rsplit('/e/', 1)[1]
        with TestClient(create_share_app(tmp_path)) as gateway:
            assert gateway.get('/api/cameras').status_code == 404
            assert gateway.get('/').status_code == 404
            assert gateway.get('/e/'+token).content == clip.read_bytes()
            assert gateway.get('/e/'+token, headers={'range': 'bytes=0-3'}).status_code == 206
            assert gateway.get('/e/'+'x'*43).status_code == 404
            with store.connect() as conn:
                conn.execute('UPDATE evidence_shares SET expires=0 WHERE token_hash=?', (hashlib.sha256(token.encode()).hexdigest(),))
            assert gateway.get('/e/'+token).status_code == 404
            other = messages[1][1].rsplit('/e/', 1)[1]
            assert agent.review(item['id'], 'false_positive')
            assert gateway.get('/e/'+other).status_code == 404
    finally:
        delivery.close(); agent.close(); store.close()


def test_uncertain_delivery_not_retried_and_missing_location_blocks(tmp_path, monkeypatch):
    import vmd.delivery as module
    monkeypatch.setattr(module, 'configuration', lambda: {'ready': True, 'base': 'https://evidence.example'})
    store = Store(tmp_path)
    agent = AlertAgent(store, background=False)
    centre = Centre(store)
    centre.signup(CentreSignup(**TEST_CENTRE, messaging_enabled=True))
    media = MediaLibrary(store)
    clip = media.cache/'fixture.mp4'; clip.write_bytes(b'clip')
    monkeypatch.setattr(media, 'playback', lambda _: clip)
    attempts = []
    def timeout(phone, body):
        attempts.append(phone)
        raise TimeoutError()
    delivery = Delivery(store, centre, agent, media, background=False, sender=timeout)
    try:
        item = event()
        store._incident(item, frames()); agent.register(item)
        agent.act(item['id'], 'dispatch')
        delivery.process_once(); delivery.process_once()
        assert len(attempts) == 1 and delivery.records(item['id'])[0]['status'] == 'uncertain'
        missing = event('b'*32); missing['signals']['location'] = None
        store._incident(missing, frames()); agent.register(missing)
        agent.act(missing['id'], 'dispatch'); delivery.process_once()
        assert len(attempts) == 1
    finally:
        delivery.close(); agent.close(); store.close()


def test_twilio_submission_format_and_immediate_failure(monkeypatch):
    import vmd.delivery as module
    from urllib.parse import parse_qs
    monkeypatch.setattr(module, 'configuration', lambda: {'ready': True, 'sid': 'AC'+'a'*32, 'token': 'fake-token', 'sender': '+12025550100'})
    captured = []
    payload = {'sid': 'SM'+'b'*32, 'status': 'queued'}
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, limit): return json.dumps(payload).encode()
    class Opener:
        def open(self, request, timeout):
            captured.append(request)
            assert timeout == 8
            return Response()
    monkeypatch.setattr(module, 'build_opener', lambda *args: Opener())
    assert module.send_sms('+12025550101', 'Test https://evidence.example/e/token')['status'] == 'accepted'
    assert captured[0].full_url == 'https://api.twilio.com/2010-04-01/Accounts/AC'+'a'*32+'/Messages.json'
    assert parse_qs(captured[0].data.decode())['To'] == ['+12025550101']
    assert captured[0].get_header('Authorization').startswith('Basic ')
    payload['status'] = 'failed'
    assert module.send_sms('+12025550101', 'Test')['status'] == 'failed'


def test_preparation_failure_is_visible_and_not_repeated_until_manual_retry(tmp_path, monkeypatch):
    import vmd.delivery as module
    monkeypatch.setattr(module, 'configuration', lambda: {'ready': True, 'base': 'https://evidence.example'})
    store = Store(tmp_path); agent = AlertAgent(store, background=False); centre = Centre(store)
    centre.signup(CentreSignup(**TEST_CENTRE, messaging_enabled=True))
    media = MediaLibrary(store)
    attempts = []
    def broken(source):
        attempts.append(source)
        raise RuntimeError('encoder failed')
    monkeypatch.setattr(media, 'playback', broken)
    delivery = Delivery(store, centre, agent, media, background=False, sender=lambda *args: pytest.fail('No SMS without a clip'))
    try:
        item = event(); store._incident(item, frames()); agent.register(item); agent.act(item['id'], 'dispatch')
        delivery.process_once(); delivery.process_once()
        assert len(attempts) == 1
        assert any('prepare' in issue for issue in agent.describe(agent.get(item['id']))['delivery_issues'])
        agent.act(item['id'], 'dispatch'); delivery.process_once()
        assert len(attempts) == 2
    finally:
        delivery.close(); agent.close(); store.close()


@pytest.mark.parametrize('kind,strong', [('possible_fight', 'fight'), ('possible_snatching', 'snatching_detected')])
@pytest.mark.parametrize('handled', [None, 'dispatch', 'false_positive'])
def test_escalation_updates_one_incident_without_duplicate_dispatch(tmp_path, kind, strong, handled):
    store = Store(tmp_path)
    agent = AlertAgent(store, background=False)
    store.on_incident = agent.register
    try:
        first = event(created=100, kind=kind)
        store.enqueue('incident', (first, frames()))
        store.jobs.join()
        original = store.incident(first['id'])
        assert original['clip'] and agent.get(first['id']) is None
        if handled == 'dispatch':
            agent.act(first['id'], 'dispatch')
        elif handled == 'false_positive':
            agent.review(first['id'], 'false_positive')
        before = agent.get(first['id'])
        upgraded = {**first, 'created': 104, 'event_type': strong}
        store.enqueue('incident', (upgraded, frames()))
        store.jobs.join()
        current = store.incident(first['id'])
        assert len(store.incidents()) == 1 and current['event_type'] == strong
        assert current['created'] == 100 and current['signals']['escalated_at'] == 104
        assert current['clip'] != original['clip']
        assert (store.clips / original['clip']).exists(), 'retain original evidence'
        assert current['signals']['stages'][0]['event_type'] == kind
        after = agent.get(first['id'])
        if handled == 'false_positive':
            assert after is None and current['review'] == 'false_positive'
        elif handled == 'dispatch':
            assert after['state'] == 'requested' and after['deadline'] == before['deadline']
            assert after['voice_requested_at'] == before['voice_requested_at']
            assert after['handled_at'] == before['handled_at']
        else:
            assert after['state'] == 'pending' and after['deadline'] == 114
            agent.tick(114)
            assert agent.get(first['id'])['state'] == 'requested'
        # Out-of-order provisional updates never downgrade an already escalated row.
        store.enqueue('incident', (first, frames()))
        store.jobs.join()
        assert store.incident(first['id'])['event_type'] == strong
        assert store.error is None
    finally:
        agent.close()
        store.close()
