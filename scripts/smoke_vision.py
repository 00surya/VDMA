"""Check real model plumbing on a bundled public sample, without opening a camera."""
from pathlib import Path
import cv2
import numpy as np
from vmd.vision import PoseModel, DepthModel

pose = PoseModel('models/yolo11n-pose.pt', 'cpu')
import ultralytics
sample = Path(ultralytics.__file__).parent / 'assets' / 'bus.jpg'
frame = cv2.imread(str(sample))
assert frame is not None, 'Bundled Ultralytics sample image missing'
frame = cv2.resize(frame, (384, 512))
people = pose.infer(frame)
assert people and all(len(person.keypoints) == 17 for person in people)
depth = DepthModel('MiDaS_small', 'cpu', 'models').infer(frame)
assert depth.shape == frame.shape[:2] and np.isfinite(depth).all()
print(f'PASS: {len(people)} people with 17-keypoint poses; real MiDaS depth {depth.shape}.')
print('This verifies inference plumbing, not fight-detection accuracy.')
