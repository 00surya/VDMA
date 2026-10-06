import io
import json
import time
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit
from xml.etree.ElementTree import fromstring

import pytest
from fastapi.testclient import TestClient

from conftest import TEST_CENTRE, authenticate
from test_response import event, frames, HEADERS
from vmd.alerts import AlertAgent
from vmd.api import create_app
from vmd.calling import Calling, call_instructions, configuration, request_call
from vmd.centre import Centre, CentreDetails, CentreSignup
from vmd.storage import Store


FAKE_CONFIG = {'ready': True, 'sid': 'AC'+'a'*32, 'token': 'test-token', 'sender': '+12025550100'}


@pytest.fixture
def voice(tmp_path, monkeypatch):
    monkeypatch.setattr('vmd.calling.configuration', lambda directory=None: FAKE_CONFIG)
    store = Store(tmp_path)
    alerts = AlertAgent(store, background=False)
    centre = Centre(store)
    centre.signup(CentreSignup(**TEST_CENTRE, calling_enabled=True))
    attempts = []
    def send(phone, instructions):
        attempts.append((phone, instructions))
        return {'status': 'queued', 'sid': 'CA'+f'{len(attempts):032x}', 'error': None}
    calling = Calling(store, centre, alerts, background=False, sender=send)
    yield store, alerts, centre, calling, attempts
    calling.close(); alerts.close(); store.close()


def test_countdown_dynamic_contacts_hospital_immediate_and_dedup(voice):
    store, alerts, centre, calling, attempts = voice
    centre.update(CentreDetails(**{**TEST_CENTRE, 'calling_enabled': True,
        'authorities': TEST_CENTRE['authorities'] + [{'name': 'Second authority', 'phone': '+12025550103'}]}))
    first = event(created=time.time()-11)
    first['signals']['location'] = None  # Calls do not require a clip, coordinates or a public URL.
    alerts.register(first)
    calling.process_once()
    assert not attempts
    alerts.tick(); calling.process_once()
    assert [phone for phone, _ in attempts] == ['+12025550101', '+12025550103']
    assert 'not been configured' in attempts[0][1]
    alerts.act(first['id'], 'call_hospital'); calling.process_once()
    assert attempts[-1][0] == TEST_CENTRE['hospital']['phone']
    for _ in range(2):
        alerts.act(first['id'], 'call_hospital'); calling.process_once()
    assert len(attempts) == 3
    # Editing contacts cannot redirect a request whose contacts were frozen.
    centre.update(CentreDetails(**{**TEST_CENTRE, 'calling_enabled': True,
        'authorities': [{'name': 'New authority', 'phone': '+12025550104'}]}))
    calling.process_once()
    assert len(attempts) == 3
    second = event('b'*32)
    alerts.register(second); alerts.act(second['id'], 'call_hospital'); calling.process_once()
    assert [phone for phone, _ in attempts[3:]] == ['+12025550104', '+12025550102']
    assert '28.61' in attempts[-1][1] and 'Test gate' in attempts[-1][1]


def test_disabled_acknowledged_expired_and_old_requests_do_not_call(voice):
    store, alerts, centre, calling, attempts = voice
    item = event(); alerts.register(item); alerts.act(item['id'], 'dispatch')
    centre.update(CentreDetails(**TEST_CENTRE))
    calling.process_once(); assert not attempts
    centre.update(CentreDetails(**TEST_CENTRE, calling_enabled=True))
    alerts.act(item['id'], 'acknowledge')
    calling.process_once(); assert not attempts
    alerts.act(item['id'], 'dispatch')
    with store.connect() as conn:
        conn.execute('UPDATE response_alerts SET voice_requested_at=?', (time.time()-301,))
    calling.process_once(); assert not attempts
    with store.connect() as conn:
        conn.execute('UPDATE response_alerts SET voice_requested_at=NULL')  # Pre-voice upgrade.
    calling.process_once(); assert not attempts
    alerts.act(item['id'], 'dispatch'); calling.process_once()
    assert len(attempts) == 1


def test_false_positive_or_disable_stops_unsubmitted_contacts(voice):
    store, alerts, centre, calling, attempts = voice
    item = event(); store._incident(item, frames()); alerts.register(item)
    alerts.act(item['id'], 'call_hospital')
    original = calling.sender
    def cancel_after_first(phone, instructions):
        result = original(phone, instructions)
        alerts.review(item['id'], 'false_positive')
        return result
    calling.sender = cancel_after_first
    calling.process_once()
    assert len(attempts) == 1
    with pytest.raises(ValueError): alerts.act(item['id'], 'call_hospital')
    second = event('b'*32); alerts.register(second); alerts.act(second['id'], 'call_hospital')
    def disable_after_first(phone, instructions):
        result = original(phone, instructions)
        centre.update(CentreDetails(**TEST_CENTRE))
        return result
    calling.sender = disable_after_first
    calling.process_once()
    assert len(attempts) == 2


def test_timeout_is_not_retried_and_poll_status_is_honest(voice):
    store, alerts, centre, calling, attempts = voice
    item = event(); alerts.register(item); alerts.act(item['id'], 'call_hospital')
    original = calling.sender
    def timeout(phone, instructions):
        if phone == TEST_CENTRE['hospital']['phone']:
            attempts.append((phone, instructions))
            raise TimeoutError('sensitive provider detail')
        return original(phone, instructions)
    calling.sender = timeout
    calling.process_once(); calling.process_once()
    assert len(attempts) == 2
    records = calling.records(item['id'])
    assert [r['status'] for r in records] == ['queued', 'uncertain']
    assert 'sensitive' not in records[1]['error']
    for status in ['ringing', 'in-progress', 'completed']:
        with store.connect() as conn:
            conn.execute('UPDATE response_calls SET updated=0')
        calling.fetcher = lambda sid: {'sid': sid, 'status': status, 'error': None}
        calling.poll_once()
        assert calling.records(item['id'])[0]['status'] == status
    with store.connect() as conn:
        conn.execute("UPDATE response_calls SET status='queued',created=0")
    calling.poll_once()
    assert all(r['status'] == 'uncertain' for r in calling.records(item['id']))


def test_api_hospital_action_accepts_saved_recordings_but_excludes_synthetic_or_unknown_sources(tmp_path, monkeypatch):
    monkeypatch.setattr('vmd.calling.configuration', lambda directory=None: FAKE_CONFIG)
    app = create_app(tmp_path)
    with TestClient(app) as client:
        authenticate(client)
        # No real Twilio sender or status fetcher can run in this test.
        app.state.calling.close()
        app.state.calling.closed.clear()
        sent = []
        def fake_send(phone, instructions):
            sent.append(phone)
            return {'sid': 'CA'+'a'*32, 'status': 'queued', 'error': None}
        app.state.calling.sender = fake_send
        assert client.post('/api/centre', headers=HEADERS,
                           json={**TEST_CENTRE, 'calling_enabled': True}).status_code == 200
        item = event(kind='gun_detected')
        app.state.store._incident(item, frames())
        route = '/api/incidents/'+item['id']+'/response'
        assert client.post(route, json={'action': 'call_hospital'}, headers=HEADERS).status_code == 200
        assert app.state.alerts.get(item['id'])['hospital']
        app.state.calling.process_once()
        assert sent == ['+12025550101', '+12025550102']
        data = client.get('/api/response-alerts').json()
        assert data['calling'] == {'configured': True, 'enabled': True, 'mode': 'incident', 'error': None}
        assert data['alerts'][0]['calls_submitted']
        assert not data['alerts'][0]['call_issues']
        assert 'test-token' not in json.dumps(data)
        recording = event('c'*32, live=False)
        recording['signals']['source_seconds'] = 7.625
        app.state.store._incident(recording, frames())
        assert client.post('/api/incidents/'+recording['id']+'/response', headers=HEADERS,
                           json={'action': 'call_hospital'}).status_code == 200
        app.state.calling.process_once()
        assert sent == ['+12025550101', '+12025550102'] * 2
        for index, (kind, live) in enumerate([('demo', True), ('demo', False),
                                            ('live', None), ('live', 'false')], start=4):
            excluded = {**event(f'{index:032x}', live=live), 'mode': kind}
            if live is None:
                excluded['signals'].pop('live_camera')
            app.state.store._incident(excluded, frames())
            assert client.post('/api/incidents/'+excluded['id']+'/response', headers=HEADERS,
                               json={'action': 'call_hospital'}).status_code == 409


def test_old_recording_calls_only_after_manual_request_and_preserves_recipient_dedup(voice):
    store, alerts, centre, calling, attempts = voice
    item = event(created=time.time()-86400, kind='possible_fight', live=False)
    item['signals']['source_seconds'] = 4.5
    store._incident(item, frames())
    alerts.register(item); alerts.tick(); calling.process_once()
    assert alerts.get(item['id']) is None and not attempts
    alerts.act(item['id'], 'dispatch')
    handled = alerts.get(item['id'])['handled_at']
    # Upgrade before the provider worker consumes the request. Its latest
    # description changes, while the operator's request and recipients do not.
    upgraded = {**item, 'event_type': 'fight', 'created': item['created']+3,
                'signals': {**item['signals'], 'source_seconds': 7.5}}
    store._incident(upgraded, frames()); alerts.register(upgraded)
    calling.process_once()
    assert len(attempts) == 1
    assert 'A fight was identified in recorded footage' in fromstring(attempts[0][1]).find('Say').text
    alerts.act(item['id'], 'call_hospital'); calling.process_once()
    assert [phone for phone, _ in attempts] == ['+12025550101', '+12025550102']
    alerts.act(item['id'], 'dispatch'); calling.process_once()
    assert len(attempts) == 2
    assert alerts.get(item['id'])['handled_at'] == handled
    assert alerts.get(item['id'])['trigger'] == 'manual'


def test_call_api_uses_saved_target_echo_url_and_reports_rejection(monkeypatch):
    import vmd.calling as module
    captured = []
    payload = {'sid': 'CA'+'b'*32, 'status': 'queued'}
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
    instructions = call_instructions('Centre <&> + 50% # प्रवेश', event())
    speech = fromstring(instructions).find('Say')
    assert speech.text.startswith('Hello, we are calling from Centre <&> + 50% # प्रवेश')
    assert speech.attrib == {'voice': 'Polly.Joanna', 'language': 'en-US', 'loop': '2'}
    assert request_call(FAKE_CONFIG, '+12025550102', instructions)['status'] == 'queued'
    request = captured[0]
    assert request.full_url.endswith('/Calls.json')
    assert request.get_method() == 'POST'
    params = parse_qs(request.data.decode())
    assert set(params) == {'To', 'From', 'Url'}
    assert params['To'] == ['+12025550102'] and params['From'] == ['+12025550100']
    url = urlsplit(params['Url'][0])
    assert (url.scheme, url.netloc, url.path, url.fragment) == ('https', 'twimlets.com', '/echo', '')
    assert parse_qs(url.query) == {'Twiml': [instructions]}
    assert FAKE_CONFIG['token'] not in params['Url'][0]
    assert FAKE_CONFIG['sid'] not in params['Url'][0]
    request_call({**FAKE_CONFIG, 'mode': 'trial_template'}, '+12025550103', instructions)
    assert parse_qs(captured[-1].data.decode()) == {'To': ['+12025550103'], 'From': ['+12025550100'],
        'Url': ['https://webhooks.twilio.com/v1/Voice/Template/voice_text_to_speech']}
    request_call(FAKE_CONFIG, sid=payload['sid'])
    assert captured[-1].get_method() == 'GET' and captured[-1].data is None
    with pytest.raises(RuntimeError): request_call(FAKE_CONFIG, sid='CA'+'c'*32)
    class Rejected:
        code = 400
        def open(self, request, timeout):
            raise HTTPError(request.full_url, self.code, 'error', {},
                            io.BytesIO(b'{"code":21211,"message":"sensitive provider detail"}'))
    monkeypatch.setattr(module, 'build_opener', lambda *args: Rejected())
    result = request_call(FAKE_CONFIG, '+12025550102', instructions)
    assert result['status'] == 'failed' and '21211' in result['error'] and 'sensitive' not in result['error']
    Rejected.code = 503
    assert request_call(FAKE_CONFIG, '+12025550102', instructions)['status'] == 'uncertain'
    class TrialRejected:
        def open(self, request, timeout):
            raise HTTPError(request.full_url, 400, 'error', {},
                            io.BytesIO(b'{"code":0,"message":"trial accounts have limited parameter access"}'))
    monkeypatch.setattr(module, 'build_opener', lambda *args: TrialRejected())
    result = request_call(FAKE_CONFIG, '+12025550102', instructions)
    assert result['status'] == 'failed' and result['sid'] is None
    assert 'no call was created' in result['error'] and 'upgraded' not in result['error']


def test_local_credentials_and_environment_override(tmp_path, monkeypatch):
    for name in ['TWILIO_ACCOUNT_SID', 'TWILIO_AUTH_TOKEN', 'TWILIO_FROM_NUMBER', 'TWILIO_VOICE_MODE']:
        monkeypatch.delenv(name, raising=False)
    assert not configuration(tmp_path)['ready']
    (tmp_path/'twilio.json').write_text(json.dumps(FAKE_CONFIG))
    assert configuration(tmp_path)['ready']
    monkeypatch.setenv('TWILIO_FROM_NUMBER', '+12025550103')
    assert configuration(tmp_path)['sender'] == '+12025550103'
    assert not configuration(tmp_path/'different-installation')['ready']
    monkeypatch.setenv('TWILIO_VOICE_MODE', 'trial_template')
    assert configuration(tmp_path)['mode'] == 'trial_template' and configuration(tmp_path)['ready']
    monkeypatch.setenv('TWILIO_VOICE_MODE', 'typo')
    assert not configuration(tmp_path)['ready']


def test_disabled_manual_call_shows_setup_error_without_queueing(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        authenticate(client)
        for action in ['dispatch', 'call_hospital']:
            response = client.post('/api/incidents/'+'a'*32+'/response', json={'action': action}, headers=HEADERS)
            assert response.status_code == 409
            assert 'Voice calls are off' in response.json()['detail']
        assert not client.get('/api/response-alerts').json()['alerts']


def test_incident_speech_uses_original_detection_time_place_and_centre(monkeypatch):
    import os
    from datetime import datetime, timezone
    # The spoken timestamp must be the event time, not the later dispatch time.
    previous = os.environ.get('TZ')
    try:
        monkeypatch.setenv('TZ', 'Asia/Kolkata')
        time.tzset()
        stamp = datetime(2026, 9, 26, 8, 17, tzinfo=timezone.utc).timestamp()
        item = event(created=stamp, kind='fight')
        item['signals']['location']['place'] = 'North gate <&>'
        speech = fromstring(call_instructions('Campus centre', item)).find('Say').text
        assert 'calling from Campus centre' in speech
        assert 'A fight was detected at North gate <&>' in speech
        assert '1:47 PM on 26 September 2026 IST' in speech
        assert 'Latitude 28.61' in speech
    finally:
        if previous is None:
            os.environ.pop('TZ', None)
        else:
            os.environ['TZ'] = previous
        time.tzset()


def test_recording_speech_distinguishes_analysis_time_offset_and_configured_location():
    from datetime import datetime, timezone
    stamp = datetime(2026, 9, 26, 8, 17, tzinfo=timezone.utc).timestamp()
    item = event(created=stamp, kind='fight', live=False)
    item['camera_name'] = 'Reviewed entrance <&>'
    item['signals']['source_seconds'] = 7.64878
    speech = fromstring(call_instructions('Campus centre', item)).find('Say').text
    when = datetime.fromtimestamp(stamp, tz=timezone.utc).astimezone().strftime('%I:%M %p on %d %B %Y %Z').lstrip('0')
    assert 'A fight was identified in recorded footage labelled Reviewed entrance <&>' in speech
    assert f'The footage was analyzed at {when}' in speech
    assert 'About 7.6 seconds into the recording' in speech
    assert 'Configured location: Test gate' in speech
    assert 'Latitude 28.61, longitude 77.21' in speech
    assert 'original recording time and capture location have not been verified' in speech
    assert 'was detected at' not in speech


@pytest.mark.parametrize('offset', [None, -1, float('nan'), float('inf'), '7.5', True])
def test_recording_speech_omits_invalid_offsets_and_does_not_invent_a_location(offset):
    item = event(live=False)
    item['signals'].update(source_seconds=offset, location=None)
    speech = fromstring(call_instructions('Campus centre', item)).find('Say').text
    assert 'No location has been configured for this recording' in speech
    assert 'seconds into the recording' not in speech
    assert 'Latitude' not in speech


def test_recording_speech_accepts_a_zero_source_offset():
    item = event(live=False)
    item['signals']['source_seconds'] = 0
    speech = fromstring(call_instructions('Campus centre', item)).find('Say').text
    assert 'About 0.0 seconds into the recording' in speech
