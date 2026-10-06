from dataclasses import replace
import math

import numpy as np

from vmd.behavior import BehaviorHeuristic
from vmd.heuristics import Person, FightHeuristic, Rules
from vmd.stabilization import PoseStabilizer
from vmd.vision import Motion, synthetic_frame


def person(track=1, x=200, posture='seated', jitter=0):
    points=[(x,90,.99)]*17
    points[5:7]=[(x-25,120,.99),(x+25,120,.99)]
    points[11:13]=[(x-20,180,.99),(x+20,180,.99)]
    points[9:11]=[(x+40+jitter,155,.99),(x-40-jitter,155,.99)]
    points[13:17]=[(x-65,200,.99),(x+65,200,.99),(x+35,215,.99),(x-35,215,.99)]
    box=(x-85,75,x+85,230)
    if posture=='upright':
        box=(x-50,75,x+50,340)
        points[15:17]=[(x-20,325,.99),(x+20,325,.99)]
    elif posture=='prone':
        box=(x-100,230,x+140,310)
        points[5:7]=[(x-55,260,.99),(x-55,280,.99)]
        points[11:13]=[(x+45,260,.99),(x+45,280,.99)]
    return Person(track,box,points,body_scale=250)


def test_cross_legged_jitter_has_no_velocity_or_alert_without_image_motion():
    stabilizer=PoseStabilizer(); behavior=BehaviorHeuristic()
    motion=Motion(); image=np.zeros((400,640,3),np.uint8)
    for n in range(100):
        raw=[person(jitter=30*math.sin(n*2)),person(2,280,jitter=30*math.cos(n*2))]
        flow,camera=motion.infer(image,raw,n*.1)
        for p in raw:p.local_flow=motion.person_flow(p)
        filtered=stabilizer.update(raw,n*.1,motion.joint_motion)
        assert all(max(p.limb_speeds.values())==0 for p in filtered)
        result=behavior.update(filtered,n*.1,flow,camera)
        assert not result.events and not result.trigger
        assert result.state!='possible_fight'


def test_deadband_reacquisition_and_jump_rejection():
    stabilizer=PoseStabilizer()
    for n in range(10):
        p=stabilizer.update([person(jitter=n%2)],n*.1,lambda p,i:1)[0]
        assert max(p.limb_speeds.values())==0
    raw=person(jitter=300)
    p=stabilizer.update([raw],1.,lambda p,i:2)[0]
    assert p.limb_speeds[9]==0 and p.keypoints[9][2]==0
    p=stabilizer.update([person(jitter=40)],1.1,lambda p,i:2)[0]
    assert p.limb_speeds[9]==0
    p=stabilizer.update([person(jitter=-40)],4.,lambda p,i:2)[0]
    assert max(p.limb_speeds.values())==0
    p=stabilizer.update([person(track=3,jitter=40)],4.1,lambda p,i:2)[0]
    assert max(p.limb_speeds.values())==0


def test_image_supported_repeated_strikes_survive_filter_and_trigger():
    stabilizer=PoseStabilizer(); behavior=BehaviorHeuristic()
    events=[]; peak=0
    for n in range(100):
        _,people,_,flow,camera=synthetic_frame(n/12,'interaction')
        for p in people:p.local_flow=flow
        filtered=stabilizer.update(people,n/12,lambda p,i:2)
        peak=max(peak,max(p.limb_speeds[9] for p in filtered))
        result=behavior.update(filtered,n/12,flow,camera)
        events.extend(result.events)
    assert peak>1
    assert len(events)==1 and events[0].event_type=='fight'
    assert events[0].events==[]


def posture_sequence(before=None, after_fight=False, other_track=False):
    behavior=BehaviorHeuristic(); events=[]
    if before:
        for n in range(6):behavior.update([person(posture=before)],n*.1)
    if after_fight:behavior.recent_fights[2 if other_track else 1]=.5
    for n in range(55):
        result=behavior.update([person(posture='prone')],.6+n*.1)
        events.extend(result.events)
    return events


def test_separate_prone_fall_and_fight_aftermath_labels():
    assert [e.event_type for e in posture_sequence()]==['person_down']
    assert [e.event_type for e in posture_sequence('upright')]==['possible_fall']
    assert [e.event_type for e in posture_sequence('upright',True)]==['person_down_after_fight']
    assert [e.event_type for e in posture_sequence('upright',True,True)]==['possible_fall']


def test_cross_legged_seated_is_not_prone_even_with_recent_fight():
    behavior=BehaviorHeuristic();behavior.recent_fights[1]=0
    for n in range(60):
        assert not behavior.update([person()],n*.1).events


def test_posture_gaps_camera_movement_and_missing_torso_reset_history():
    behavior=BehaviorHeuristic()
    for n in range(18):behavior.update([person(posture='prone')],n*.1)
    assert not behavior.update([person(posture='prone')],4).events
    assert not behavior.update([person(posture='prone')],4.1,camera_motion=.2).events
    for n in range(15):assert not behavior.update([person(posture='prone')],4.2+n*.1).events
    bad=person(posture='prone');bad.keypoints[5]=(100,100,.1)
    assert not behavior.update([bad],5.7).events
    for n in range(15):assert not behavior.update([person(posture='prone')],5.8+n*.1).events


def test_fast_noncontact_limb_does_not_borrow_contact_from_stationary_hand():
    h=FightHeuristic()
    a,b=person(),person(2,280)
    a.limb_speeds={9:0,10:2};b.limb_speeds={9:0,10:0}
    a.local_flow=b.local_flow=.1
    # Right hand is touching B, but the moving left hand points away.
    for n in range(30):
        result=h.update([a,b],n*.1,.1)
        assert not result.trigger


def test_kick_and_close_contact_rules_require_motion_and_persistence():
    for pattern in ('kick','clinch'):
        h=FightHeuristic();events=[]
        for n in range(40):
            a,b=person(),person(2,290)
            for p in (a,b):p.local_flow=.1;p.limb_speeds={9:.45,10:.45,15:0,16:0}
            # Reciprocal hands reach the other's torso, rather than empty space
            # inside a wide person box.
            a.keypoints[9]=(275,155,.99)
            b.keypoints[10]=(215,155,.99)
            if pattern=='kick':
                a.keypoints[15]=(280,150,.99);a.limb_speeds={15:2,9:0,10:0}
                b.limb_speeds={9:0,10:0}
            result=h.update([a,b],n*.1,.1)
            events.extend(result.events)
        assert len(events)==1
        assert ('kick' if pattern=='kick' else 'struggle') in events[0].signals['pattern']


def test_fight_then_fall_preserves_same_track_context_and_active_posture():
    behavior=BehaviorHeuristic();events=[]
    for n in range(35):
        a,b=person(posture='upright'),person(2,290,posture='upright')
        a.keypoints[9]=(275,155,.99)
        b.keypoints[10]=(215,155,.99)
        for p in (a,b):p.local_flow=.1;p.limb_speeds={9:2,10:2}
        events.extend(behavior.update([a,b],n*.1,.1).events)
    assert events and events[0].event_type=='fight'
    for n in range(80):
        result=behavior.update([person(posture='prone')],3.5+n*.1)
        events.extend(result.events)
    assert [e.event_type for e in events]==['fight','person_down_after_fight']
    assert result.state=='person_down_after_fight' and not result.trigger


def test_repeated_strike_bursts_trigger_without_continuous_arm_speed():
    behavior = BehaviorHeuristic()
    events = []
    for n in range(100):
        active = n % 10 < 3  # Punch, then a 0.7-second withdrawal/pause.
        a, b = person(), person(2, 280)
        for p in (a, b):
            p.local_flow = .10 if active else .002
            p.limb_speeds = {9: 1.4 if active else 0, 10: 0, 15: 0, 16: 0}
        result = behavior.update([a, b], n * .1, .10 if active else .002)
        events.extend(result.events)
    assert len(events) == 1 and events[0].event_type == 'fight'
    assert events[0].signals['strike_bursts'] >= 2
    assert events[0].signals['supported_seconds'] >= max(.4, behavior.fight.rules.hold_seconds * .6)


def test_one_short_strike_and_unsupported_pose_movement_do_not_accumulate_alerts():
    for image_motion in (0, .1):
        behavior = BehaviorHeuristic()
        for n in range(70):
            active = 5 <= n <= 7
            a, b = person(), person(2, 280)
            for p in (a, b):
                p.local_flow = image_motion
                p.limb_speeds = {9: 2 if active else 0, 10: 0}
            assert not behavior.update([a, b], n * .1, image_motion).events


def test_partially_occluded_hip_keeps_pose_but_anchor_changes_reset_velocity():
    stabilizer = PoseStabilizer()
    for n in range(20):
        raw = person(jitter=20 * math.sin(n))
        raw.keypoints[12] = (*raw.keypoints[12][:2], .1)
        filtered = stabilizer.update([raw], n * .1, lambda p, i: 0)[0]
        assert filtered.pose_reliable
        assert max(filtered.limb_speeds.values()) == 0
    filtered = stabilizer.update([person(jitter=50)], 2, lambda p, i: 2)[0]
    assert max(filtered.limb_speeds.values()) == 0


def test_moving_horizontal_person_is_not_reported_as_stationary_person_down():
    behavior = BehaviorHeuristic()
    for n in range(60):
        assert not behavior.update([person(x=200 + n * 4, posture='prone')], n * .1).events
    events = []
    for n in range(55):
        events.extend(behavior.update([person(x=436, posture='prone')], 6 + n * .1).events)
    assert len(events) == 1
    assert events[0].signals['stationary_seconds'] >= 3
    assert events[0].signals['centroid_speed_px_s'] < 5


def test_hands_up_uses_wrists_and_persistence_and_reports_once():
    behavior = BehaviorHeuristic()
    raised = person()
    raised.keypoints[9:11] = [(170, 50, .99), (230, 50, .99)]
    events = []
    for n in range(50):
        result = behavior.update([raised], n * .1)
        if n < 20:
            assert not result.events
        events.extend(result.events)
    assert [e.event_type for e in events] == ['hands_up']
    assert result.state == 'hands_up' and not result.trigger
    behavior.update([person()], 5)
    for n in range(19):
        assert not behavior.update([raised], 5.1 + n * .1).events


def test_hands_up_rejects_one_hand_bad_nose_and_high_ankles():
    cases = [person() for _ in range(3)]
    cases[0].keypoints[9] = (170, 50, .99)
    cases[1].keypoints[9:11] = [(170, 50, .99), (230, 50, .99)]
    cases[1].keypoints[0] = (200, 90, .1)
    cases[2].keypoints[15:17] = [(170, 50, .99), (230, 50, .99)]
    for p in cases:
        behavior = BehaviorHeuristic()
        for n in range(50):
            assert not behavior.update([p], n * .1).events


def test_rule_diagnostics_explain_missing_people_and_camera_motion():
    behavior = BehaviorHeuristic()
    assert 'two visible tracked people' in behavior.update([person()], 0).signals['blockers'][0]
    result = behavior.update([person(), person(2)], .1, camera_motion=.3)
    assert result.state == 'camera_moving' and result.signals['blockers']


def test_hands_up_reaches_engine_overlay_and_saved_incident(monkeypatch, tmp_path):
    import threading
    import time
    from types import SimpleNamespace
    import vmd.engine as module
    from vmd.engine import Engine
    from vmd.storage import Store

    class Capture:
        def __init__(self, source):
            self.finished = False
            self.error = None
            self.stop_event = threading.Event()
        def start(self): pass
        def stop(self): pass
        def latest(self, after):
            return after + 1, (after + 1) * .1, np.zeros((400, 640, 3), np.uint8)

    class Pose:
        def __init__(self, *args): pass
        def infer(self, frame):
            p = person()
            p.keypoints[9:11] = [(170, 50, .99), (230, 50, .99)]
            return [p]

    monkeypatch.setattr(module, 'Capture', Capture)
    monkeypatch.setattr(module, 'PoseModel', Pose)
    monkeypatch.setattr(module, 'choose_device', lambda device: 'cpu')
    store = Store(tmp_path)
    engine = Engine(store)
    settings = SimpleNamespace(mode='live', source='fixture', device='cpu', depth='off',
        depth_fps=.5, threshold=.68, hold_seconds=1.2, target_fps=30)
    try:
        engine.start(settings)
        deadline = time.monotonic() + 5
        events = []
        while time.monotonic() < deadline:
            events = store.incidents()
            if events:
                break
            time.sleep(.03)
        assert len(events) == 1 and events[0]['event_type'] == 'hands_up'
        assert engine.snapshot()['alert']['label'] == 'HANDS UP / REVIEW'
    finally:
        engine.stop()
        store.close()
