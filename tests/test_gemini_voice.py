import asyncio
import base64
import hashlib
import json
import secrets
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace
from xml.etree.ElementTree import fromstring

import numpy as np
import pytest
from fastapi.testclient import TestClient
from twilio.request_validator import RequestValidator

from test_gemini import service, complete
from vmd.alerts import AlertAgent
from vmd.voice import create_app, VoiceDatabase, pcm_from_mulaw, mulaw_from_pcm, public_origin

SID='CA'+'1'*32
STREAM='MZ'+'2'*32
TOKEN='test-auth-token'
ORIGIN='https://voice.example'


@pytest.fixture
def gateway(service, monkeypatch):
    ai, store, provider, item, engines=service
    alerts=AlertAgent(store,background=False)
    alerts.act(item['id'],'dispatch')
    monkeypatch.setenv('VMD_VOICE_PUBLIC_URL',ORIGIN)
    monkeypatch.setattr('vmd.voice.calling_configuration',lambda directory=None: {'ready':True,'token':TOKEN})
    token=secrets.token_urlsafe(32)
    with store.connect() as conn:
        conn.execute('INSERT INTO ai_voice_sessions(token_hash,incident_id,phone,expires) VALUES(?,?,?,?)',
                     (hashlib.sha256(token.encode()).hexdigest(),item['id'],'+12025550101',time.time()+300))
    yield ai,store,item,token
    alerts.close()


def signature(path,params=None):
    return RequestValidator(TOKEN).compute_signature(ORIGIN+path,params or {})


def bind(client,token):
    params={'CallSid':SID,'To':'+12025550101'}
    return client.post('/twiml/'+token,content='CallSid='+SID+'&To=%2B12025550101',
                       headers={'X-Twilio-Signature':signature('/twiml/'+token,params),
                                'Content-Type':'application/x-www-form-urlencoded'})


def test_signed_twiml_binds_call_and_exposes_only_voice(gateway):
    ai,store,item,token=gateway
    with TestClient(create_app(store.directory)) as client:
        assert client.get('/api/incidents').status_code==404
        assert client.get('/docs').status_code==404
        assert client.post('/twiml/'+token,content='CallSid='+SID).status_code==403
        response=bind(client,token)
        assert response.status_code==200
        root=fromstring(response.text)
        assert root.find('Connect/Stream').attrib['url']=='wss://voice.example/stream'
        assert root.find('Connect/Stream/Parameter').attrib['value']==token
    database=VoiceDatabase(store.directory)
    assert database.consume(token,SID)==item['id']
    with pytest.raises(ValueError): database.consume(token,SID)


def test_expired_and_cancelled_sessions_rejected(gateway):
    ai,store,item,token=gateway
    with store.connect() as conn: conn.execute('UPDATE ai_voice_sessions SET expires=0')
    with TestClient(create_app(store.directory)) as client: assert bind(client,token).status_code==404
    with store.connect() as conn: conn.execute('UPDATE ai_voice_sessions SET expires=?',(time.time()+100,))
    store.review(item['id'],'false_positive')
    with TestClient(create_app(store.directory)) as client: assert bind(client,token).status_code==404


def test_codec_roundtrip_and_silence():
    assert np.frombuffer(pcm_from_mulaw(bytes([255])*160),dtype='<i2').shape==(320,)
    assert not np.frombuffer(pcm_from_mulaw(bytes([255])*160),dtype='<i2').any()
    pcm=np.full(480,5000,dtype='<i2')
    decoded=np.frombuffer(pcm_from_mulaw(mulaw_from_pcm(pcm.tobytes())),dtype='<i2')
    assert len(decoded)==320 and np.max(np.abs(decoded-5000))<200


def test_context_approved_report_and_freshness(gateway):
    ai,store,item,token=gateway
    complete(ai,item['id']); ai.approve(item['id'],True)
    context=VoiceDatabase(store.directory).context(item['id'])
    assert context['approved'] and context['report']['summary']
    assert 'phone' not in json.dumps(context)
    assert context['metadata']['occurred_at'] is None


def test_live_bridge_audio_tool_and_stop(gateway):
    ai,store,item,token=gateway
    received=[]
    class Session:
        def __init__(self): self.turn=0
        async def send_client_content(self,**kwargs): received.append(('greeting',kwargs))
        async def send_realtime_input(self,**kwargs): received.append(('input',kwargs))
        async def send_tool_response(self,**kwargs): received.append(('tool',kwargs))
        async def receive(self):
            await asyncio.sleep(.02)
            self.turn+=1
            if self.turn==1:
                yield SimpleNamespace(data=None, server_content=None,tool_call=SimpleNamespace(function_calls=[SimpleNamespace(name='get_incident_context',id='tool1')]))
            else:
                yield SimpleNamespace(data=np.zeros(480,dtype='<i2').tobytes(),server_content=None,tool_call=None)
    @asynccontextmanager
    async def factory(**kwargs):
        received.append(('connect',kwargs)); yield Session()
    with TestClient(create_app(store.directory,session_factory=factory)) as client:
        assert bind(client,token).status_code==200
        with client.websocket_connect('/stream',headers={'X-Twilio-Signature':signature('/stream')}) as ws:
            ws.send_json({'event':'connected'})
            ws.send_json({'event':'start','start':{'streamSid':STREAM,'callSid':SID,
                'customParameters':{'session':token},'mediaFormat':{'encoding':'audio/x-mulaw','sampleRate':8000}}})
            ws.send_json({'event':'media','media':{'payload':base64.b64encode(bytes([255])*160).decode()}})
            assert ws.receive_json()['event']=='media'
            ws.send_json({'event':'stop'})
    assert any(kind=='input' for kind,_ in received)
    assert any(kind=='tool' for kind,_ in received)
    config=received[0][1]['config']
    assert 'You cannot send messages or initiate calls' in config['system_instruction']


def test_origin_validation(monkeypatch):
    for invalid in ['http://voice.example','https://user:password@voice.example','https://voice.example/path']:
        monkeypatch.setenv('VMD_VOICE_PUBLIC_URL',invalid)
        assert public_origin() is None


def test_gateway_absence_keeps_announcement_fallback(service,monkeypatch):
    from vmd.voice import issue_voice_url
    ai,store,_,item,_=service
    monkeypatch.setenv('VMD_AI_ENABLED','true')
    monkeypatch.setenv('VMD_VOICE_PUBLIC_URL',ORIGIN)
    def unavailable(*args,**kwargs): raise OSError('offline')
    monkeypatch.setattr('vmd.voice.urlopen',unavailable)
    assert issue_voice_url(store,item['id'],'+12025550101') is None
    with store.connect() as conn:
        assert conn.execute('SELECT COUNT(*) FROM ai_voice_sessions').fetchone()[0]==0


def test_signed_hook_cannot_bind_another_recipient(gateway):
    ai,store,item,token=gateway
    params={'CallSid':SID,'To':'+12025550199'}
    with TestClient(create_app(store.directory)) as client:
        response=client.post('/twiml/'+token,content='CallSid='+SID+'&To=%2B12025550199',
            headers={'X-Twilio-Signature':signature('/twiml/'+token,params)})
        assert response.status_code==404
