"""Real-model functional/compute check on public imagery, not an accuracy or power test."""
import argparse
import importlib.util
import json
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from vmd.api import StartRequest
from vmd.engine import Engine
from vmd.storage import Store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('tmp/eco-benchmark.json'))
    args = parser.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    sample = Path(importlib.util.find_spec('ultralytics').origin).parent/'assets/bus.jpg'
    frame = cv2.resize(cv2.imread(str(sample)), (384, 512))
    results = []
    with tempfile.TemporaryDirectory() as temporary:
        video = Path(temporary)/'quiet-then-moving.avi'
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 12, (384, 512))
        for tick in range(24*12):
            image = frame if tick < 21*12 else cv2.warpAffine(frame, np.float32([[1,0,(tick%8)*5],[0,1,0]]), (384,512))
            writer.write(image)
        writer.release()
        for eco in (False, True):
            store = Store(Path(temporary)/str(eco))
            engine = Engine(store)
            before = after = None
            quiet_seen = wake_seen = objects_seen = depth_seen = False
            started = time.monotonic()
            try:
                engine.start(StartRequest(mode='live', source=str(video), device='cpu',
                    depth='ZipDepth', depth_fps=1, detection_mode='depth_confirmed',
                    object_detection=True, object_fps=1, eco_mode=eco))
                while time.monotonic()-started < 50:
                    state = engine.snapshot()
                    assert state['status'] != 'error', state
                    assert state['depth_meta']['status'] != 'error', state['depth_meta']
                    assert state['object_meta']['status'] != 'error', state['object_meta']
                    stamp = state.get('source_time', 0)
                    quiet_seen |= state.get('eco_state') == 'quiet'
                    wake_seen |= stamp >= 21 and state.get('eco_state') == 'active'
                    objects_seen |= bool(state.get('scene_objects'))
                    depth_seen |= bool(state.get('depth_available'))
                    if stamp >= 12 and before is None:
                        before = state
                    if stamp >= 20 and after is None:
                        after = state
                    if state['status'] == 'finished':
                        break
                    time.sleep(.04)
                assert before and after and state['status'] == 'finished', state
                assert objects_seen and depth_seen
                if eco:
                    assert quiet_seen and wake_seen
                measurements = {key: round(after[key]-before[key], 1) for key in
                    ('processed_frames', 'pose_inference_ms_total', 'depth_submissions', 'object_submissions')}
                results.append(dict(eco=eco, quiet_window_seconds=round(after['source_time']-before['source_time'], 2),
                    **measurements, eco_skipped_frames=state['eco_skipped_frames'],
                    quiet_seen=quiet_seen, woke_on_motion=wake_seen,
                    real_object_results=objects_seen, real_depth_results=depth_seen))
                print(json.dumps(results[-1]), flush=True)
            finally:
                engine.stop()
                store.close()
    assert results[1]['processed_frames'] < results[0]['processed_frames']
    report = {'sample':'Ultralytics bus.jpg, still for 21s then translated for 3s',
              'device':'cpu', 'pose_threads':1, 'accuracy_evaluation':False, 'measurements':results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
