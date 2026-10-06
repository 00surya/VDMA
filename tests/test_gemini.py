import json
import queue
import time
from types import SimpleNamespace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from conftest import authenticate
from test_response import event, frames, HEADERS
from vmd.api import create_app
from vmd.engine import Engine
from vmd.gemini import IncidentAI, IncidentReport, Answer, sample_clip, configuration, version
from vmd.storage import Store
from vmd.calling import call_instructions


REPORT = {'summary': 'Two people interact; one moves away.', 'assessment': 'unclear',
          'subjects': ['Person wearing a dark shirt'], 'timeline': [{'seconds': 0, 'observation': 'Two people visible'}],
          'uncertainties': ['The object is not clear'], 'briefing': 'Two people interact. Please verify the footage.',
          'subjects_visible': True}


class Provider:
    def __init__(self):
        self.calls = []
        self.callback = None
    def generate(self, prompt, images, schema, quality):
        self.calls.append((prompt, images, schema, quality))
        if self.callback:
            self.callback()
        return (dict(REPORT) if schema is IncidentReport else {'answer': 'Two people are visible.',
            'evidence_seconds': [0], 'limitations': 'Identity is unverified.'}), {'model': 'fake', 'input_tokens': 10}


@pytest.fixture
def service(tmp_path):
    store = Store(tmp_path)
    item = event()
    store._incident(item, frames())
    provider = Provider()
    engines = {}
    ai = IncidentAI(store, lambda key: engines.get(key), provider=provider, background=False)
    yield ai, store, provider, item, engines
    ai.close(); store.close()


def complete(ai, incident_id):
    ai.start(incident_id)
    ai.process(ai.jobs.get_nowait())
    return ai.report(incident_id)


def test_analysis_persists_bounded_frames_metadata_and_no_dispatch(service):
    ai, store, provider, item, _ = service
    data = complete(ai, item['id'])
    assert data['status'] == 'ready' and not data['approved']
    assert data['report']['assessment'] == 'unclear'
    prompt, images, schema, quality = provider.calls[0]
    assert len(images) <= 48 and images[0][0] == 0
    assert 'source_url' not in prompt and 'phone' not in prompt and 'clip' not in json.loads(prompt.split('\n',1)[1])
    assert schema is IncidentReport and quality
    with store.connect() as conn:
        assert not conn.execute("SELECT 1 FROM sqlite_master WHERE name='response_alerts'").fetchone()
    assert store.incident(item['id'])['review'] == 'unreviewed'


def test_duplicate_queue_approval_and_version_upgrade(service):
    ai, store, _, item, _ = service
    ai.start(item['id']); ai.start(item['id'])
    assert ai.jobs.qsize() == 1
    ai.process(ai.jobs.get_nowait())
    assert ai.briefing(item['id']) is None
    ai.approve(item['id'], True)
    assert ai.briefing(item['id']) == REPORT['briefing']
    with store.connect() as conn:
        conn.execute("UPDATE incidents SET event_type='snatching_detected' WHERE id=?", (item['id'],))
    assert ai.report(item['id'])['status'] == 'outdated'
    assert ai.briefing(item['id']) is None
    with pytest.raises(ValueError): ai.approve(item['id'], True)


def test_synthetic_false_positive_and_unsafe_clip_rejected(service):
    ai, store, provider, item, _ = service
    store.review(item['id'], 'false_positive')
    with pytest.raises(ValueError): ai.start(item['id'])
    store.review(item['id'], 'unreviewed')
    with store.connect() as conn:
        conn.execute("UPDATE incidents SET clip='../secret.avi' WHERE id=?", (item['id'],))
    assert complete(ai, item['id'])['status'] == 'error'
    assert not provider.calls
    with store.connect() as conn:
        conn.execute("UPDATE incidents SET mode='demo' WHERE id=?", (item['id'],))
    with pytest.raises(ValueError): ai.ask(item['id'], 'Who is visible?')


def test_review_race_discards_output_and_redacts_error(service):
    ai, store, provider, item, _ = service
    provider.callback = lambda: store.review(item['id'], 'false_positive')
    ai.start(item['id']); ai.process(ai.jobs.get_nowait())
    with store.connect() as conn:
        row = conn.execute('SELECT status,report,error FROM ai_reports WHERE incident_id=?', (item['id'],)).fetchone()
    assert row[0] == 'error' and row[1] is None
    assert 'credential' not in row[2]


def test_questions_timestamp_validation_and_rate_limit(service, monkeypatch):
    ai, _, _, item, _ = service
    assert ai.ask(item['id'], 'What is visible?')['evidence_seconds'] == [0]
    with pytest.raises(ValueError): ai.ask(item['id'], ' ')
    with pytest.raises(ValueError): ai.ask(item['id'], 'x'*601)
    monkeypatch.setenv('VMD_AI_HOURLY_LIMIT', '1')
    with pytest.raises(RuntimeError): ai.ask(item['id'], 'Again?')
    assert ai.gate.acquire(False)
    ai.gate.release()


def test_provider_error_never_returns_private_exception(service):
    ai, _, provider, item, _ = service
    def fail(): raise RuntimeError('SECRET https://private-camera/password')
    provider.callback = fail
    data = complete(ai, item['id'])
    assert data['status'] == 'error' and 'SECRET' not in json.dumps(data)


def test_schema_bounds():
    with pytest.raises(ValidationError): IncidentReport.model_validate({**REPORT, 'subjects': ['x'*401]})
    with pytest.raises(ValidationError): IncidentReport.model_validate({**REPORT, 'timeline': [{'seconds': float('nan'), 'observation': 'x'}]})


def test_follow_is_session_bound_and_stops_on_reconnect(service):
    ai, store, provider, item, engines = service
    class Camera:
        ai_session = 'first'
        def snapshot(self): return {'status': 'running', 'stale': False}
        def ai_frames(self, since): return [(since+1, frames()[0][1]), (since+2, frames()[1][1])]
    camera = Camera(); engines[item['camera_id']] = camera
    complete(ai, item['id'])
    ai.follow(item['id'], True)
    ai.follows[item['id']]['next'] = 0
    ai._follow_tick()
    assert len(ai.report(item['id'])['observations']) == 1
    camera.ai_session = 'second'
    ai.follows[item['id']]['next'] = 0
    ai._follow_tick()
    assert not ai.report(item['id'])['following']


def test_follow_stop_during_inference_drops_late_result(service):
    ai, _, provider, item, engines = service
    engines[item['camera_id']] = SimpleNamespace(ai_session='first',
        snapshot=lambda: {'status':'running','stale':False},
        ai_frames=lambda since: [(since+1,frames()[0][1]), (since+2,frames()[1][1])])
    complete(ai, item['id'])
    ai.follow(item['id'], True); ai.follows[item['id']]['next'] = 0
    provider.callback = lambda: ai.follow(item['id'], False)
    ai._follow_tick()
    assert ai.report(item['id'])['observations'] == []


def test_recordings_cannot_follow_and_sampling_rejects_missing(service):
    ai, store, _, item, _ = service
    with store.connect() as conn:
        conn.execute('UPDATE incidents SET signals=? WHERE id=?', (json.dumps({'live_camera':False}),item['id']))
    with pytest.raises(ValueError): ai.follow(item['id'], True)
    with pytest.raises(ValueError): sample_clip(Path('/no-such-clip'))


def test_announcements_only_use_explicit_approved_briefing(service):
    ai, _, _, item, _ = service
    complete(ai,item['id'])
    assert 'AI observation' not in call_instructions('Centre',item,ai.briefing(item['id']))
    ai.approve(item['id'],True)
    assert 'Operator-approved AI observation' in call_instructions('Centre',item,ai.briefing(item['id']))


def test_configuration_legacy_alias_and_vertex_no_api_key(monkeypatch):
    monkeypatch.delenv('VMD_AI_VERTEX_PROJECT',raising=False)
    monkeypatch.setenv('CLIMBRK_AI_VERTEX_PROJECT','test-project')
    assert configuration()['project']=='test-project'
    assert configuration()['backend']=='vertex'


def test_api_auth_validation_and_report_download(tmp_path, monkeypatch):
    # Exercise real routes in isolated storage without external providers.
    with TestClient(create_app(tmp_path)) as client:
        assert client.get('/api/ai/status').status_code==401
        authenticate(client)
        store=client.app.state.store
        item=event(); store._incident(item,frames())
        client.app.state.ai.injected=True
        client.app.state.ai.provider=Provider()
        path=f"/api/incidents/{item['id']}/ai"
        assert client.post(path,json={}).status_code==403
        assert client.post(path,json={'quality':'yes'},headers=HEADERS).status_code==422
        assert client.get('/api/ai/status').json()['configured']
        assert client.get(path).json()['status']=='not_started'
        assert client.post(path,json={'quality':False},headers=HEADERS).status_code==200
        for _ in range(100):
            if client.get(path).json()['status'] not in ('queued','running'): break
            time.sleep(.02)
        assert client.get(path).json()['status']=='ready'
        assert client.post(path+'/approve',json={'enabled':True},headers=HEADERS).status_code==200
        response=client.get(path+'/report')
        assert response.status_code==200 and 'attachment' in response.headers['content-disposition']
        pdf = client.get(path+'/report.pdf')
        assert pdf.status_code == 200 and pdf.headers['content-type'] == 'application/pdf'
        assert pdf.content.startswith(b'%PDF-')
        assert client.post(path+'/ask',json={'question':'What happened?'},headers=HEADERS).json()['answer']
        assert client.post(path+'/follow',json={'enabled':True},headers=HEADERS).status_code==409
        assert client.post(path,json={},headers={**HEADERS,'Origin':'https://evil.example'}).status_code==403


def test_vertex_client_uses_cloud_credentials_without_api_key(monkeypatch):
    from vmd.gemini import make_client
    from google import genai
    captured={}
    creds=object()
    monkeypatch.setenv('VMD_AI_ENABLED','true')
    monkeypatch.setenv('VMD_AI_VERTEX_PROJECT','synthetic-project')
    monkeypatch.setenv('VMD_AI_BACKEND','vertex')
    monkeypatch.delenv('GEMINI_API_KEY',raising=False)
    monkeypatch.setattr('vmd.gemini.google_credentials',lambda project:creds)
    def client(**kwargs): captured.update(kwargs); return object()
    monkeypatch.setattr(genai,'Client',client)
    make_client()
    assert captured['vertexai'] and captured['credentials'] is creds
    assert 'api_key' not in captured


def test_follow_first_starts_initial_clip_analysis(service):
    ai, _, _, item, engines = service
    engines[item['camera_id']] = SimpleNamespace(ai_session='session',
        snapshot=lambda: {'status': 'running', 'stale': False})
    data = ai.follow(item['id'], True)
    assert data['following'] and data['status'] == 'queued'
    assert ai.jobs.qsize() == 1


def test_automatic_analysis_waits_for_clip_and_deduplicates_restart(service, monkeypatch):
    ai, store, provider, item, _ = service
    monkeypatch.setenv('VMD_AI_AUTO_ANALYZE', 'true')
    saved = store.incident(item['id'])['clip']
    with store.connect() as conn:
        conn.execute('UPDATE incidents SET clip=NULL WHERE id=?', (item['id'],))
    ai._automatic()
    assert ai.jobs.empty()
    # Clip readiness can arrive more than two minutes after detection.
    ai.auto_since = item['created'] - 400
    with store.connect() as conn:
        conn.execute('UPDATE incidents SET clip=?,created=? WHERE id=?', (saved, item['created']-300, item['id']))
    ai._automatic()
    ai.process(ai.jobs.get_nowait())
    assert ai.report(item['id'])['status'] == 'ready'
    assert len(list((store.directory/'ai-reports').glob('*.pdf'))) == 1
    ai.auto_seen.clear()  # restart loses in-memory deduplication
    ai._automatic()
    assert ai.jobs.empty() and len(provider.calls) == 1


def test_pdf_contains_image_and_authenticated_video_link(service):
    import io
    from pypdf import PdfReader
    ai, _, _, item, _ = service
    complete(ai, item['id'])
    reader = PdfReader(io.BytesIO(ai.pdf(item['id'], 'http://127.0.0.1:8875')))
    text = '\n'.join(page.extract_text() for page in reader.pages)
    assert item['id'] in text and REPORT['summary'] in text
    assert 'Evidence frame at' in text and 'signed-in session' in text
    assert sum(len(page.images) for page in reader.pages) >= 1
    links = [annotation.get_object()['/A']['/URI'] for page in reader.pages for annotation in page.get('/Annots', []) if '/A' in annotation.get_object()]
    assert f"http://127.0.0.1:8875/api/incidents/{item['id']}/play" in links
    ai.store.review(item['id'], 'false_positive')
    with pytest.raises(ValueError): ai.pdf(item['id'])
