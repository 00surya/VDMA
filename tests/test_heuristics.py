from vmd.heuristics import FightHeuristic, Person, Rules


def pair(offset=0, distance=90, translate=0, confidence=.99, depth=(None, None)):
    result=[]
    for i,x in enumerate((200+translate, 200+distance+translate)):
        points=[(x,100,confidence)]*17
        points[5:7]=[(x-25,130,confidence),(x+25,130,confidence)]
        points[11:13]=[(x-20,190,confidence),(x+20,190,confidence)]
        points[9]=(x+offset,160,confidence)
        points[10]=(x-offset,160,confidence)
        result.append(Person(i+1,(x-50,50,x+50,250),points,depth[i]))
    return result


def test_calm_crowd_and_whole_body_translation_do_not_trigger():
    h=FightHeuristic()
    for n in range(50):
        result=h.update(pair(translate=n*30),n*.1,.3)
        assert not result.trigger
        assert result.score < h.rules.threshold


def test_sustained_fast_close_interaction_triggers_once_and_cools_down():
    h=FightHeuristic()
    triggers=[]
    for n in range(180):
        result=h.update(pair(offset=65 if n%2 else -65),n*.1,.25)
        if result.trigger:triggers.append(n*.1)
    assert len(triggers)==1
    assert triggers[0]>=h.rules.hold_seconds
    assert result.state=='possible_fight'


def test_isolated_spike_and_far_people_do_not_trigger():
    for distance in (90,500):
        h=FightHeuristic()
        for n in range(50):
            result=h.update(pair(offset=50 if n==10 else 0,distance=distance),n*.1,.25)
            assert not result.trigger


def test_camera_motion_suppresses_even_fast_interaction():
    h=FightHeuristic()
    for n in range(30):
        result=h.update(pair(offset=n%2*80),n*.1,.25,.2)
        assert not result.trigger
        assert result.state=='camera_moving'


def test_missing_confidence_and_depth_disagreement_suppress():
    for kwargs in ({'confidence':.1},{'depth':(.1,.9)}):
        h=FightHeuristic()
        for n in range(30):
            result=h.update(pair(offset=n%2*80,**kwargs),n*.1,.25)
            assert not result.trigger


def test_stream_gap_and_changed_tracks_reset_persistence():
    h=FightHeuristic()
    for n in range(10):h.update(pair(offset=n%2*80),n*.1,.25)
    result=h.update(pair(offset=80),5,.25)
    assert not result.trigger
    assert not h.pending
    for n in range(10):
        people=pair(offset=n%2*80)
        people[1].track_id=100+n
        assert not h.update(people,5.1+n*.1,.25).trigger


def test_frame_rate_normalization():
    speeds=[]
    for dt in (.05,.1,.2):
        h=FightHeuristic()
        h.update(pair(),0,.25)
        result=h.update(pair(offset=400*dt),dt,.25)
        speeds.append(result.signals['wrist_speed'])
    assert speeds==[2,2,2]
