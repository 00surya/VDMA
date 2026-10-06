import time
import cv2
import numpy as np
from vmd.capture import Capture
from vmd.heuristics import Person
from vmd.vision import Motion, annotate


def test_background_translation_is_removed_from_local_motion():
    rng=np.random.default_rng(12)
    frame=rng.integers(0,256,(180,320,3),dtype=np.uint8)
    frame=cv2.GaussianBlur(frame,(5,5),0)
    person=Person(1,(100,40,170,160),[(120,50,.9)]*17)
    motion=Motion()
    motion.infer(frame,[person],0)
    translated=cv2.warpAffine(frame,np.float32([[1,0,4],[0,1,0]]),(320,180))
    local,camera=motion.infer(translated,[person],.1)
    assert camera>.09
    assert local<.025


def test_head_blur_works_when_face_keypoints_are_missing():
    rng=np.random.default_rng(4)
    frame=rng.integers(0,256,(240,320,3),dtype=np.uint8)
    person=Person(1,(100,40,200,220),[(0,0,0)]*17)
    output=annotate(frame,[person])
    assert output[50:80,120:180].var()<frame[50:80,120:180].var()*.1
    assert np.array_equal(output[120:150,120:180],frame[120:150,120:180])


def test_capture_file_timestamps_and_eof(tmp_path):
    path=tmp_path/'fixture.avi'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),20,(160,100))
    for n in range(6):writer.write(np.full((100,160,3),n*25,np.uint8))
    writer.release()
    capture=Capture(str(path), target_fps=20)
    capture.start()
    deadline=time.monotonic()+3
    packets=[]
    after=0
    try:
        while not capture.finished and time.monotonic()<deadline:
            packet=capture.latest(after)
            if packet is not None:
                packets.append(packet)
                after=packet[0]
            else:
                time.sleep(.005)
    finally:
        capture.stop()
    assert capture.error is None and capture.finished
    assert len(packets)==6 and packets[0][1]==0
    packet=packets[-1]
    assert packet[0]==6 and abs(packet[1]-.25)<.001
    assert packet[2].shape==(100,160,3)
