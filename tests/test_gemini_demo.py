import json
import numpy as np
from fastapi.testclient import TestClient
from scripts.gemini_demo import create_demo_app,scene_frame,DEMO_ID
from test_gemini import Provider


def test_generated_scene_changes_without_camera_or_external_assets():
    first,contact,fallen=scene_frame(0),scene_frame(4),scene_frame(8)
    assert first.shape==(360,640,3)
    assert not np.array_equal(first,contact)
    assert not np.array_equal(contact,fallen)


def test_demo_uses_isolated_recording_and_never_instantiates_response_services(tmp_path):
    with TestClient(create_demo_app(tmp_path,provider=Provider())) as client:
        assert client.get('/').status_code==200
        assert client.get('/api/status').json()['configured']
        store=client.app.state.store
        event=store.incident(DEMO_ID)
        assert event['signals']['live_camera'] is False and event['signals']['generated_demo']
        assert event['signals']['location'] is None
        assert 'no detector' in event['reasons'][0].lower()
        with store.connect() as conn:
            names={row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert not {'response_alerts','response_calls','response_deliveries'} & names
        assert client.post('/api/analyze',json={'quality':False}).status_code==403
        assert client.post('/api/analyze',json={'quality':False},headers={'x-vmd-client':'dashboard','Origin':'https://other.example'}).status_code==403
        assert client.post('/api/ask',json={'question':' '},headers={'x-vmd-client':'dashboard'}).status_code==503
        assert client.get('/api/incidents').status_code==404
        assert client.get('/demo.mjs').headers['content-security-policy'].startswith("default-src 'self'")
