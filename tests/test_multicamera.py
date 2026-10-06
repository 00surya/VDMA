from conftest import authenticate
import sqlite3
import time
import numpy as np
from fastapi.testclient import TestClient
from vmd.api import create_app
from vmd.storage import Store

HEADERS={'x-vmd-client':'dashboard'}


def test_two_cameras_independent_alerts_stop_and_capacity(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        authenticate(client)
        def add(name):
            response=client.post('/api/cameras',json={'mode':'demo','name':name,'hold_seconds':.3},headers=HEADERS)
            assert response.status_code==200
            return response.json()['camera_id']
        first,second=add('Stage'),add('Calm room')
        assert first!=second
        assert client.post(f'/api/cameras/{first}/scenario',json={'scenario':'interaction'},headers=HEADERS).status_code==200
        deadline=time.monotonic()+6
        while time.monotonic()<deadline:
            events=client.get('/api/incidents').json()
            if events:break
            time.sleep(.1)
        assert events and all(event['camera_id']==first for event in events)
        assert events[0]['camera_name']=='Stage' and events[0]['event_type']=='possible_fight'
        states={state['camera_id']:state for state in client.get('/api/cameras').json()['cameras']}
        assert states[first]['alert']['label']=='POSSIBLE FIGHT / REVIEW'
        assert states[second]['alert'] is None
        frames=client.get('/api/camera-frames').json()
        assert frames[first]['pose'] and frames[second]['pose']
        before=states[second]['sequence']
        assert client.post(f'/api/cameras/{first}/stop',json={},headers=HEADERS).json()['status']=='idle'
        time.sleep(.2)
        states={state['camera_id']:state for state in client.get('/api/cameras').json()['cameras']}
        assert states[second]['sequence']>before
        assert client.post(f'/api/cameras/{first}/restart',json={},headers=HEADERS).status_code==200
        assert client.post(f'/api/cameras/{first}/restart',json={},headers=HEADERS).status_code==409
        add('Third');add('Fourth')
        assert client.post('/api/cameras',json={'mode':'demo'},headers=HEADERS).status_code==409
        assert client.post('/api/start',json={'mode':'demo'},headers=HEADERS).status_code==409
        assert client.post(f'/api/cameras/{first}/remove',json={},headers=HEADERS).json()['evidence_preserved']
        assert client.get('/api/incidents').json()[0]['camera_id']==first
        assert client.get(f'/api/cameras/{first}/frames').status_code==404
        add('Replacement')


def test_schema_upgrade_preserves_old_evidence_and_groups_telemetry(tmp_path):
    db=tmp_path/'telemetry.sqlite3'
    with sqlite3.connect(db) as conn:
        conn.executescript("""
        CREATE TABLE incidents (id TEXT PRIMARY KEY,created REAL,mode TEXT,score REAL,reasons TEXT,signals TEXT,review TEXT,clip TEXT,clip_error TEXT);
        CREATE TABLE telemetry (created REAL,people INTEGER,score REAL,mode TEXT);
        INSERT INTO incidents VALUES ('old',100,'live',.8,'[]','{}','confirmed','old.avi',NULL);
        INSERT INTO telemetry VALUES (100,3,.1,'live');
        """)
    store=Store(tmp_path)
    try:
        event=store.incidents()[0]
        assert event['review']=='confirmed' and event['clip']=='old.avi'
        assert event['camera_id']=='camera-1' and event['event_type']=='fight'
        store.enqueue('telemetry',(100,9,.1,'live','second'))
        store.jobs.join()
        rows={row['camera_id']:row for row in store.analytics(0)}
        assert rows['camera-1']['average']==3 and rows['second']['average']==9
    finally:store.close()


def test_real_bytetrack_counter_is_independent_between_camera_instances(monkeypatch,tmp_path):
    # Exercise the installed tracker implementation; no model or network is needed.
    monkeypatch.setenv('YOLO_CONFIG_DIR',str(tmp_path/'ultralytics'))
    monkeypatch.setenv('YOLO_OFFLINE','true')
    monkeypatch.setenv('YOLO_AUTOINSTALL','false')
    from ultralytics.engine.results import Boxes
    from vmd.tracking import CameraTracker
    def boxes(extra=False):
        data=[[10,10,60,180,.99,0]]
        if extra:data.append([250,10,300,180,.99,0])
        return Boxes(np.array(data,dtype=np.float32),(360,640)).numpy()
    image=np.zeros((360,640,3),np.uint8)
    a=CameraTracker()
    assert list(a.update(boxes(),image)[:,4])==[1]
    b=CameraTracker()
    assert list(b.update(boxes(),image)[:,4])==[1]
    a.update(boxes(True),image)
    tracks=a.update(boxes(True),image)
    assert set(tracks[:,4])=={1,2}
    assert list(b.update(boxes(),image)[:,4])==[1]
