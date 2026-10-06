from conftest import authenticate
import time
from fastapi.testclient import TestClient
from vmd.api import create_app
from vmd.api import CameraSettingsRequest, StartRequest
from pydantic import ValidationError
import pytest

HEADERS={'x-vmd-client':'dashboard'}


def test_explicit_fight_confirmation_duration_is_independent_of_review_observation():
    assert StartRequest().fight_confirmation_seconds == 3
    for seconds in (.5, 1, 3, 10):
        settings = StartRequest(hold_seconds=5, fight_confirmation_seconds=seconds)
        assert settings.fight_confirmation_seconds == seconds and settings.hold_seconds == 5
        assert CameraSettingsRequest(fight_confirmation_seconds=seconds).model_dump(exclude_unset=True) == {
            'fight_confirmation_seconds': seconds}
    for seconds in (0, .49, 10.01, float('nan'), float('inf')):
        with pytest.raises(ValidationError):
            StartRequest(fight_confirmation_seconds=seconds)


def test_local_api_validation_and_origin(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        authenticate(client)
        assert client.get('/').status_code==200
        assert client.get('/api/state').json()['status']=='idle'
        assert client.post('/api/start',json={'mode':'demo'}).status_code==403
        assert client.post('/api/start',json={'mode':'demo'},headers={**HEADERS,'origin':'https://untrusted.example'}).status_code==403
        assert client.get('/api/state',headers={'host':'untrusted.example'}).status_code==400
        assert client.post('/api/start',json={'mode':'live','source':'file:///etc/passwd'},headers=HEADERS).status_code==422
        assert client.post('/api/start',json={'threshold':2},headers=HEADERS).status_code==422
        for settings in ({'detection_mode':'unknown'},
                         {'detection_mode':'depth_confirmed','depth':'off'},
                         {'detection_mode':'depth_confirmed','depth':'ZipDepth','depth_fps':.1}):
            assert client.post('/api/cameras',json=settings,headers=HEADERS).status_code==422
        assert client.post('/api/incidents/missing/review',json={'decision':'confirmed'},headers=HEADERS).status_code==404


def test_synthetic_session_alert_review_and_clip(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        authenticate(client)
        assert client.post('/api/start',json={'mode':'demo','hold_seconds':.3},headers=HEADERS).status_code==200
        assert client.post('/api/start',json={'mode':'demo'},headers=HEADERS).status_code==409
        assert client.post('/api/scenario',json={'scenario':'interaction'},headers=HEADERS).status_code==200
        deadline=time.monotonic()+6
        events=[]
        while time.monotonic()<deadline:
            events=client.get('/api/incidents').json()
            if events:break
            time.sleep(.1)
        assert events,client.get('/api/state').json()
        event=events[0]
        assert event['mode']=='demo'
        frames=client.get('/api/frames').json()
        assert frames['pose'] and frames['depth'] and frames['sequence']>0
        assert client.get(f"/api/incidents/{event['id']}/clip").status_code==200
        assert client.post(f"/api/incidents/{event['id']}/review",json={'decision':'false_positive'},headers=HEADERS).status_code==200
        assert client.get('/api/incidents').json()[0]['review']=='false_positive'
        assert client.get('/api/analytics').json()==[]
        assert client.post('/api/stop',json={},headers=HEADERS).json()['status']=='idle'


def test_street_map_assets_and_scoped_resource_policy(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        authenticate(client)
        response = client.get('/')
        assert response.headers['cache-control'] == 'no-cache'
        policy = response.headers['content-security-policy']
        assert "img-src 'self' data: https://tile.openstreetmap.org;" in policy
        assert "script-src 'self';" in policy
        assert "connect-src 'self';" in policy
        assert response.headers['referrer-policy'] == 'no-referrer'
        for path in ('/map.mjs', '/map-location.mjs', '/map.css', '/leaflet.js', '/leaflet.css', '/leaflet-LICENSE.txt'):
            assert client.get(path).status_code == 200
