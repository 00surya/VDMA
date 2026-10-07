"""Sampled weapons and optional bag/rider context, on their own captured frame."""
import hashlib
import os
import queue
import time
from pathlib import Path

from .depth_worker import DepthWorker, replace_latest

WEIGHTS = 'yolo26s.pt'
MODEL_URL = 'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt'
SHA256 = '646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b'
KNIFE_WEIGHTS = 'assalim-normal-compressed-best.pt'
KNIFE_REVISION = '3d641e6001abdaa1afa3cd45d0fe02a9554f3d0e'
KNIFE_REPOSITORY = 'JoaoAssalim/Weapons-and-Knives-Detector-with-YOLOv8'
# Immutable Git blob for runs/detect/Normal_Compressed/weights/best.pt.
KNIFE_URL = f'https://api.github.com/repos/{KNIFE_REPOSITORY}/git/blobs/27bc7a0b92d72282ec7c12472395e3c6d0c0edbd'
KNIFE_SHA256 = '21d61ad8068caca3062d33fd8d05da445ef9ae81a2bb817249e085b0f1307cf0'
KNIFE_LICENSE = 'Assalim-GPL-3.0.txt'
KNIFE_LICENSE_URL = f'https://api.github.com/repos/{KNIFE_REPOSITORY}/git/blobs/f288702d2fa16d3cdf0035b15a9fcbc552cd88e7'
KNIFE_LICENSE_SHA256 = '3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986'
WEAPON_CONFIDENCE = .90

COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')


def limit_cpu_threads(_predictor):
    # YOLO's lazy device setup overwrites the worker's limit before its first
    # prediction. This callback runs after setup, before warmup and inference.
    import torch
    if torch.get_num_threads() != 1:
        torch.set_num_threads(1)


class ObjectModel:
    weights, checksum, labels = WEIGHTS, SHA256, COCO_CLASSES
    confidence, class_ids, detector = .4, [0], 'YOLO26s'

    def __init__(self, model_dir, device='cpu', *, class_ids=None):
        path = Path(model_dir).resolve() / self.weights
        if not path.is_file():
            raise RuntimeError(f'{self.detector} weights missing. Run scripts/download_object_model.py.')
        if hashlib.sha256(path.read_bytes()).hexdigest() != self.checksum:
            raise RuntimeError(f'{self.detector} weights checksum mismatch. Re-download the pinned model.')
        runtime = path.parent / '.runtime'
        (runtime / 'ultralytics').mkdir(parents=True, exist_ok=True)
        os.environ['YOLO_CONFIG_DIR'] = str(runtime / 'ultralytics')
        os.environ['YOLO_OFFLINE'] = 'true'
        os.environ['YOLO_AUTOINSTALL'] = 'false'
        os.environ.setdefault('MPLCONFIGDIR', str(runtime / 'matplotlib'))
        self.model = self.load_model(path)
        self.device = device
        if class_ids is not None:
            self.class_ids = list(class_ids)
        names = self.model.names
        labels = tuple(names[i] for i in range(len(names)))
        if labels != self.labels:
            raise RuntimeError(f'Unexpected {self.detector} class mapping; detector disabled.')
        if device == 'cpu':
            self.model.add_callback('on_predict_start', limit_cpu_threads)

    def load_model(self, path):
        from ultralytics import YOLO
        return YOLO(str(path), task='detect')

    def infer(self, frame):
        result = self.model.predict(frame, imgsz=640, conf=self.confidence, classes=self.class_ids, device=self.device,
                                    max_det=50, verbose=False)[0]
        if result.boxes is None:
            return []
        return [{'label': str(result.names[int(cls)]).lower(), 'confidence': float(score),
                 'box': [round(float(x), 1) for x in box], 'context_only': True, 'detector': self.detector}
                for box, score, cls in zip(result.boxes.xyxy.cpu().tolist(),
                                          result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist())
                if self.class_ids is None or int(cls) in self.class_ids]


class WeaponModel(ObjectModel):
    """Assalim's Normal_Compressed best checkpoint for guns and knives."""
    weights, checksum = KNIFE_WEIGHTS, KNIFE_SHA256
    labels = ('guns', 'knife')
    confidence, class_ids, detector = WEAPON_CONFIDENCE, [0, 1], 'Assalim Normal_Compressed best · YOLOv8n'

    def load_model(self, path):
        import torch
        from ultralytics import YOLO
        from ultralytics.nn.tasks import DetectionModel
        from ultralytics.nn.modules import Bottleneck, SPPF, DFL, Concat, Detect, Conv, C2f
        from ultralytics.utils import IterableSimpleNamespace
        from ultralytics.utils.loss import v8DetectionLoss, BboxLoss
        from ultralytics.utils.tal import TaskAlignedAssigner
        # Community checkpoint: allow only the installed network classes, never
        # arbitrary pickle globals or dynamically imported checkpoint code.
        allowed = [DetectionModel, Bottleneck, SPPF, DFL, Concat, Detect, Conv, C2f,
                   torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.SiLU, torch.nn.MaxPool2d,
                   torch.nn.Upsample, torch.nn.ModuleList, torch.nn.Sequential,
                   IterableSimpleNamespace, v8DetectionLoss, BboxLoss, TaskAlignedAssigner,
                   torch.nn.BCEWithLogitsLoss]
        with torch.serialization.safe_globals(allowed):
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        model = YOLO('yolov8n.yaml', task='detect', verbose=False)
        model.model = checkpoint['model'].float().eval()
        model.model.args = {}
        return model


def weapon_detections(detections):
    """Only specialist knife/gun predictions strictly above 90% are public."""
    return [{**item, 'label': 'gun' if item['label'] == 'guns' else item['label'], 'context_only': False}
            for item in detections if item['label'] in ('knife', 'gun', 'guns')
            and WEAPON_CONFIDENCE < item['confidence'] <= 1]


class WeaponIncidents:
    """One incident per visible weapon class; brief misses do not re-arm it."""
    def __init__(self):
        self.sequence = None
        self.seen, self.saved = {}, {}

    def update(self, sample, source_time, fps=1, status='ready'):
        limit = max(3, 2 / fps)
        if (status != 'ready' or not sample or sample.get('knife_status') != 'ready'
                or sample.get('knife_error') or time.time() - sample['submitted_at'] > limit
                or not 0 <= source_time - sample['source_time'] <= limit):
            return [], []
        detections = weapon_detections(sample['detections'])
        if sample['sequence'] == self.sequence:
            return detections, []
        self.sequence = sample['sequence']
        strongest = {}
        for item in detections:
            if item['confidence'] > strongest.get(item['label'], {}).get('confidence', 0):
                strongest[item['label']] = item
        timestamp, events = sample['source_time'], []
        for label, item in strongest.items():
            if (timestamp - self.seen.get(label, float('-inf')) > limit
                    and timestamp - self.saved.get(label, float('-inf')) >= 10):
                events.append(item)
                self.saved[label] = timestamp
            self.seen[label] = timestamp
        return detections, events


def annotate_objects(frame, detections, people=(), monitored_objects=(), *, overlays=True):
    """Draw detections on their own sampled frame with best-effort head blur."""
    import cv2
    image = frame.copy()
    h, w = image.shape[:2]
    for item in people:
        if item['label'] == 'person':
            x1, y1, x2, y2 = item['box']
            left, right = max(0, int(x1)), min(w, int(x2))
            top, bottom = max(0, int(y1)), min(h, int(y1+(y2-y1)*.30))
            if right > left and bottom > top:
                region = image[top:bottom, left:right]
                region[:] = cv2.GaussianBlur(region, (0, 0), sigmaX=max(15, (right-left)/3))
    if not overlays:
        return image
    for item in monitored_objects:
        if item.get('label') not in {'backpack', 'handbag', 'suitcase', 'motorcycle'} or item.get('confidence', 0) < .5:
            continue
        x1, y1, x2, y2 = map(int, item['box'])
        color = (180, 205, 70)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        suffix = 'rider context' if item['label'] == 'motorcycle' else 'monitoring'
        cv2.putText(image, f"{item['label']} / {suffix}", (max(0, x1), max(16, y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1)
    for item in weapon_detections(detections):
        x1, y1, x2, y2 = map(int, item['box'])
        color = (20, 170, 245)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"{item['label']} {item['confidence']:.0%}",
                    (max(0, x1), max(16, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
    return image


def run_objects(requests, responses, stop, model_type, device, model_dir):
    try:
        import cv2
        import torch
        cv2.setNumThreads(1)
        torch.set_num_threads(1)
        unattended = model_type in {'objects_and_bags', 'bags', 'objects_and_bags_riders', 'bags_riders'}
        riders = model_type in {'objects_riders', 'riders', 'objects_and_bags_riders', 'bags_riders'}
        weapons_enabled = model_type not in {'bags', 'bags_riders', 'riders'}
        context_enabled = unattended or riders
        class_ids = [0, *([3] if riders else []), *([24, 26, 28] if unattended else [])]
        model = (ObjectModel(model_dir, device, class_ids=class_ids)
                 if context_enabled else ObjectModel(model_dir, device))
        if context_enabled:
            # Weak person detections still interrupt the absence claim. Bags
            # must independently clear the attendance rule's stricter cutoff.
            model.confidence = .2
        weapon, weapon_error = None, None
        if weapons_enabled:
            try:
                weapon = WeaponModel(model_dir, device)
            except Exception as exc:
                weapon_error = str(exc) if isinstance(exc, RuntimeError) else f'Weapon detector failed ({type(exc).__name__})'
        replace_latest(responses, {'status': 'ready'})
        while not stop.is_set():
            try:
                sequence, source_time, submitted_at, frame = requests.get(timeout=.1)
            except queue.Empty:
                continue
            if stop.is_set():
                break
            started = time.monotonic()
            context = model.infer(frame)
            detections = []
            if weapon is not None:
                try:
                    detections = weapon_detections(weapon.infer(frame))
                except Exception as exc:
                    weapon_error = f'Weapon detector failed ({type(exc).__name__})'
                    weapon = None
            ok, jpeg = cv2.imencode('.jpg', annotate_objects(frame, detections, context,
                                                           context if context_enabled else ()))
            if not ok:
                raise RuntimeError('Could not encode object preview')
            clean_ok, clean = cv2.imencode('.jpg', annotate_objects(frame, [], context, overlays=False))
            replace_latest(responses, {'status': 'ready', 'sequence': sequence,
                'source_time': source_time, 'submitted_at': submitted_at,
                'jpeg': jpeg.tobytes(), 'clean_jpeg': clean.tobytes() if clean_ok else None, 'detections': detections,
                'context_objects': context if context_enabled else [], 'frame_shape': list(frame.shape[:2]),
                # Keep existing API keys for clients; status covers both weapon classes.
                'knife_status': ('ready' if weapon is not None else 'error') if weapons_enabled else 'off',
                'knife_error': weapon_error,
                'latency_ms': round((time.monotonic()-started)*1000)})
    except Exception as exc:
        error = str(exc) if isinstance(exc, RuntimeError) else f'Object detection failed ({type(exc).__name__})'
        replace_latest(responses, {'status': 'error', 'error': error})


class ObjectWorker(DepthWorker):
    def __init__(self, model_dir, fps=1, *, unattended=False, weapons=True, riders=False):
        model_type = ('objects_and_bags' if weapons else 'bags') if unattended else 'objects'
        if riders:
            model_type = model_type+'_riders' if unattended or weapons else 'riders'
        super().__init__(model_type, 'cpu', model_dir, fps=fps, runner=run_objects)
        self.process.name = 'vmd-objects'

    def poll(self):
        status, result, error = super().poll()
        if error == 'Depth worker exited; pose and motion remain active':
            self.error = error = 'Object worker exited; pose and motion remain active'
        return status, result, error
