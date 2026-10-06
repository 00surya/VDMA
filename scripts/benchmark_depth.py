"""Measure depth latency on identical images; this does not measure accuracy."""
import argparse
import importlib.util
import json
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from vmd.depth import DepthModel
from vmd.vision import DepthModel as MidasModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compare-midas', action='store_true', help='Requires the previous MiDaS assets and timm==0.6.13')
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error('--runs must be positive')
    cv2.setNumThreads(1)
    if args.compare_midas:
        import torch
        torch.set_num_threads(1)
    sample = Path(importlib.util.find_spec('ultralytics').origin).parent / 'assets/bus.jpg'
    frame = cv2.imread(str(sample))
    measurements = []
    models = [('MiDaS Small', lambda device, folder: MidasModel('MiDaS_small', device, folder))] if args.compare_midas else []
    models.append(('ZipDepth ONNX', DepthModel))
    for name, factory in models:
        started = time.perf_counter()
        model = factory('cpu', Path(__file__).resolve().parents[1] / 'models')
        load_ms = (time.perf_counter()-started)*1000
        for width, height in [(384, 512), (960, 540)]:
            image = cv2.resize(frame, (width, height))
            for _ in range(2):
                model.infer(image)
            durations = []
            for _ in range(args.runs):
                started = time.perf_counter()
                depth = model.infer(image)
                durations.append((time.perf_counter()-started)*1000)
                assert depth.shape == image.shape[:2] and np.isfinite(depth).all()
            measurements.append(dict(model=name, frame=[width, height], runs=args.runs,
                                     load_ms=round(load_ms, 2), median_ms=round(statistics.median(durations), 2)))
    report = dict(platform=platform.platform(), cpu_threads=1, warmups=2,
                  sample='Ultralytics bundled bus.jpg (resized)', measurements=measurements)
    encoded = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded)


if __name__ == '__main__':
    main()
