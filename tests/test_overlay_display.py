"""Display variants keep privacy, provenance and evidence separate from inference."""
import base64
from copy import deepcopy
from types import SimpleNamespace
import time

import cv2
import numpy as np
from fastapi.testclient import TestClient

from conftest import authenticate
from vmd.api import create_app
from vmd.engine import Engine
from vmd.heuristics import Person
from vmd.objects import annotate_objects
from vmd.vision import annotate, annotate_status


def scene():
    gray = np.random.default_rng(12).integers(0, 256, (240, 320), dtype=np.uint8)
    frame = np.repeat(gray[:, :, None], 3, axis=2)
    points = [(0, 0, 0)] * 17
    for i, point in {5:(75,75),6:(105,75),7:(70,95),8:(110,95),9:(65,115),10:(115,115),
                     11:(75,125),12:(105,125),13:(75,150),14:(105,150),15:(75,175),16:(105,175)}.items():
        points[i] = (*point, .99)
    return frame, Person(1, (60,25,120,190), points)


def jpeg(image):
    return cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 78])[1].tobytes()


def decode(encoded):
    return cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)


def prepared_engine(store):
    frame, person = scene()
    engine = Engine(store, camera_id='display-fixture')
    engine.state.update(status='running', sequence=7, processed_frames=7, source_time=1,
                        object_status='ready', object_sequence=3, object_source_time=1,
                        object_submitted_at=time.time(), object_fps=1,
                        scene_objects=[{'label':'knife', 'confidence':.99}])
    engine.frames = {'pose':jpeg(annotate(frame,[person])), 'depth':b'depth',
                     'objects':jpeg(annotate_objects(frame,[{'label':'knife','confidence':.99,'box':[170,90,200,140]}],
                                                    [{'label':'person','box':list(person.box)}])),
                     'objects_clean':jpeg(annotate_objects(frame,[],[{'label':'person','box':list(person.box)}],overlays=False))}
    engine.preview_source = (7, frame.copy(), [person], None, None)
    return engine


def test_hiding_pose_and_boxes_keeps_head_blur_and_does_not_mutate_source():
    frame, person = scene()
    original = frame.copy()
    hidden = annotate(frame,[person],overlays=False)
    shown = annotate(frame,[person])
    np.testing.assert_array_equal(frame,original)
    np.testing.assert_array_equal(hidden[85:],frame[85:])
    assert np.any(shown[85:] != hidden[85:])
    assert hidden[40:65,75:105].var() < frame[40:65,75:105].var()*.1
    np.testing.assert_array_equal(shown[40:65,75:105],hidden[40:65,75:105])


def test_object_variant_hides_weapon_bag_and_rider_boxes_but_retains_privacy():
    frame, person = scene()
    people = [{'label':'person','box':list(person.box)}]
    weapons = [{'label':'knife','confidence':.99,'box':[170,90,200,140]}]
    context = [{'label':'motorcycle','confidence':.8,'box':[140,120,250,210]},
               {'label':'backpack','confidence':.8,'box':[220,85,260,125]}]
    hidden = annotate_objects(frame,weapons,people,context,overlays=False)
    np.testing.assert_array_equal(hidden[85:],frame[85:])
    assert hidden[40:65,75:105].var() < frame[40:65,75:105].var()*.1
    assert np.any(annotate_objects(frame,weapons,people,context)[85:] != hidden[85:])


def test_clean_preview_is_lazy_cached_and_cannot_change_pipeline_state_or_evidence(monkeypatch):
    engine = prepared_engine(SimpleNamespace(error=None,dropped=0))
    original_frames, original_state = deepcopy(engine.frames), deepcopy(engine.state)
    calls, encode = [], cv2.imencode
    monkeypatch.setattr(cv2,'imencode',lambda *args: (calls.append(1),encode(*args))[1])
    shown, _ = engine.preview_frames()
    assert calls == [] and shown['pose'] == original_frames['pose']
    hidden, state = engine.preview_frames(False)
    second, _ = engine.preview_frames(False)
    assert len(calls) == 1 and hidden['pose'] == second['pose']
    assert hidden['pose'] != shown['pose'] and hidden['objects'] == original_frames['objects_clean']
    assert state == original_state and engine.frames == original_frames and engine.state == original_state
    assert engine.preview_frames(True)[0]['pose'] == original_frames['pose']
    engine.stop()
    assert engine.preview_source is None and engine.clean_preview is None
    assert engine.preview_frames(False)[0]['pose'] is None


def test_alert_banner_is_retained_while_pose_geometry_is_hidden():
    engine = prepared_engine(SimpleNamespace(error=None,dropped=0))
    seq, frame, people, _, _ = engine.preview_source
    alert = {'event_type':'possible_snatching','label':'POSSIBLE SNATCHING / REVIEW'}
    engine.preview_source = (seq,frame,people,alert,None)
    hidden = decode(engine.preview_frames(False)[0]['pose'])
    expected = decode(jpeg(annotate_status(annotate(frame,people,overlays=False),alert)))
    np.testing.assert_array_equal(hidden,expected)
    assert not np.array_equal(hidden[:66],frame[:66])


def test_hidden_frame_api_is_read_only_keeps_metadata_and_stale_object_rules(tmp_path):
    app = create_app(tmp_path)
    with TestClient(app) as client:
        assert client.get('/api/camera-frames?overlays=false').status_code == 401
        authenticate(client)
        engine = prepared_engine(app.state.store)
        app.state.cameras[engine.camera_id] = engine
        app.state.engine = engine
        initial = deepcopy(engine.state)
        for route in ['/api/cameras/display-fixture/frames','/api/frames']:
            shown = client.get(route).json()
            hidden = client.get(route+'?overlays=false').json()
            assert shown['sequence'] == hidden['sequence'] == 7
            assert hidden['pose'] != shown['pose']
            if 'objects' in hidden:
                assert hidden['object_meta'] == shown['object_meta']
                assert hidden['scene_objects'] == shown['scene_objects']
                assert hidden['objects'] != shown['objects']
        grid = client.get('/api/camera-frames?overlays=false').json()[engine.camera_id]
        assert grid['pose'] == base64.b64encode(engine.preview_frames(False)[0]['pose']).decode()
        assert client.get('/api/frame/pose?overlays=false').content == engine.preview_frames(False)[0]['pose']
        assert engine.state == initial and app.state.store.incidents() == []
        engine.state['object_submitted_at'] = time.time()-10
        expired = client.get('/api/cameras/display-fixture/frames?overlays=false').json()
        assert expired['objects'] is None and expired['scene_objects'] == []


def test_unavailable_hidden_variant_never_falls_back_to_an_annotated_image():
    engine = prepared_engine(SimpleNamespace(error=None,dropped=0))
    engine.preview_source = None
    engine.frames.pop('objects_clean')
    hidden, _ = engine.preview_frames(False)
    assert hidden['pose'] is None and hidden['objects'] is None
    assert engine.preview_frames()[0]['pose']
