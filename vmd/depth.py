"""Portable ZipDepth inference. Output is normalized inverse depth, never metres."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

MODEL_NAME = "ZipDepth"
REVISION = "91f3fd21e131641f51e8d35736d1958350180e3a"
WEIGHTS = "zipdepth-91f3fd2.onnx"
MANIFEST = "zipdepth-91f3fd2.json"
NOTICE = "ZipDepth-LICENSE.txt"
CHECKPOINT_SHA256 = "627c04fda584133ead4310074884a4a037061b4c01ba86e73e492ea30fab570d"
INPUT_SHORT_SIDE = 256
MAX_INPUT_SIDE = 448


def normalize_inverse_depth(depth):
    """Normalize one frame; a flat/invalid map contains no usable Z information."""
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2 or not np.isfinite(depth).all():
        raise ValueError("Depth model returned an invalid map")
    low, high = np.percentile(depth, (2, 98))
    if high - low <= 1e-6:
        return np.full(depth.shape, .5, dtype=np.float32)
    return np.clip((depth - low) / (high - low), 0, 1).astype(np.float32)


def prepare_input(frame):
    """RGB [0,1], aspect-preserving resize, then pad to multiples of 32."""
    height, width = frame.shape[:2]
    scale = min(INPUT_SHORT_SIDE / min(height, width), MAX_INPUT_SIDE / max(height, width))
    h, w = max(1, round(height * scale)), max(1, round(width * scale))
    resized = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(resized, 0, (-h) % 32, 0, (-w) % 32, cv2.BORDER_REPLICATE)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255, (h, w)


class DepthModel:
    name = MODEL_NAME
    backend = "ONNX Runtime CPU"

    def __init__(self, device, model_dir):
        # A bounded CPU session works across desktop platforms without GPU drivers.
        # Pose's device selection is deliberately independent of the depth runtime.
        root = Path(model_dir)
        path = root / WEIGHTS
        try:
            manifest = json.loads((root / MANIFEST).read_text())
            if (manifest["revision"] != REVISION or manifest["checkpoint_sha256"] != CHECKPOINT_SHA256
                    or hashlib.sha256(path.read_bytes()).hexdigest() != manifest["onnx_sha256"]):
                raise ValueError("Depth assets failed integrity verification")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise RuntimeError("Install verified ZipDepth assets: python scripts/download_depth_model.py") from exc
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("Install depth runtime: python -m pip install -e '.[vision]'") from exc
        ort.disable_telemetry_events()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])

    def infer(self, frame):
        tensor, (h, w) = prepare_input(frame)
        output = self.session.run(["depth"], {"image": tensor})[0]
        if output.shape != (1, 1, *tensor.shape[2:]):
            raise ValueError("Depth model returned an unexpected shape")
        # Remove padding before resizing so torso pixels retain their coordinates.
        depth = cv2.resize(output[0, 0, :h, :w], (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        return normalize_inverse_depth(depth)
