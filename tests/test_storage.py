import cv2
import numpy as np
from vmd.storage import EvidenceBuffer, Store


def test_buffer_bounded_by_time_and_bytes():
    buffer=EvidenceBuffer(seconds=10,max_bytes=25)
    for t in range(20):buffer.append(t,b'12345')
    assert buffer.bytes<=25
    assert len(buffer.snapshot())==5
    buffer.append(50,b'12')
    assert buffer.snapshot()==[(50,b'12')]


def test_clip_roundtrip_review_and_demo_exclusion(tmp_path):
    store=Store(tmp_path)
    image=np.full((100,160,3),80,np.uint8)
    jpeg=cv2.imencode('.jpg',image)[1].tobytes()
    event={'id':'testclip','created':1000,'mode':'demo','score':.8,'reasons':['Fast limbs'],'signals':{}}
    store.enqueue('incident',(event,[(0,jpeg),(.2,jpeg),(.8,jpeg),(1,jpeg)]))
    store.enqueue('telemetry',(1000,2,.8,'demo'))
    store.enqueue('telemetry',(1001,4,.2,'live'))
    store.jobs.join()
    incident=store.incidents()[0]
    assert incident['clip']=='testclip.avi'
    cap=cv2.VideoCapture(str(store.clips/incident['clip']))
    assert cap.isOpened() and cap.read()[0]
    assert 1<=cap.get(cv2.CAP_PROP_FRAME_COUNT)/cap.get(cv2.CAP_PROP_FPS)<=1.2
    cap.release()
    assert store.review('testclip','false_positive')
    assert store.incidents()[0]['review']=='false_positive'
    assert store.analytics(0)[0]['average']==4
    store.close()
