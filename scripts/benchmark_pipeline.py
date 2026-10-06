"""Short local throughput check on recorded sample imagery (not an accuracy benchmark)."""
import json
import importlib.util
import statistics
import tempfile
import time
from pathlib import Path

import cv2
from vmd.api import StartRequest
from vmd.engine import Engine
from vmd.storage import Store


def main():
    image=Path(importlib.util.find_spec('ultralytics').origin).parent/'assets/bus.jpg'
    frame=cv2.resize(cv2.imread(str(image)),(384,512))
    measurements=[]
    with tempfile.TemporaryDirectory() as temporary:
        path=Path(temporary)/'sample.avi'
        writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),20,(384,512))
        for _ in range(1200):writer.write(frame)
        writer.release()
        for depth in ('off','MiDaS_small'):
            store=Store(Path(temporary)/depth)
            engine=Engine(store,'models')
            seen=set()
            samples=[]
            depth_frames=set()
            first_pose=None
            first_depth=None
            started=time.monotonic()
            try:
                engine.start(StartRequest(mode='live',source=str(path),device='cpu',depth=depth,depth_fps=.5,target_fps=12))
                while time.monotonic()-started<25:
                    state=engine.snapshot()
                    assert state['status']!='error',state
                    if state['sequence'] and first_pose is None:first_pose=time.monotonic()
                    if state['sequence'] and state['sequence'] not in seen:
                        seen.add(state['sequence'])
                        if first_pose and time.monotonic()-first_pose>3:samples.append(state['fps'])
                    meta=state.get('depth_meta',{})
                    assert meta.get('status')!='error',meta
                    if meta.get('sequence') is not None:
                        depth_frames.add(meta['sequence'])
                        if first_depth is None:first_depth=time.monotonic()
                    if first_pose and time.monotonic()-first_pose>12:break
                    time.sleep(.03)
                assert samples and (depth=='off' or depth_frames),state
                measurements.append({'depth':depth,'processed_frames':len(seen),'median_pose_fps':round(statistics.median(samples),2),
                                     'depth_frames':len(depth_frames),'first_pose_seconds':round(first_pose-started,2),
                                     'first_depth_seconds':round(first_depth-started,2) if first_depth else None})
            finally:
                engine.stop()
                store.close()
    print(json.dumps(measurements,indent=2))


if __name__=='__main__':
    main()
