"""Adapters keep model loading out of the web server and heuristic tests."""
from pathlib import Path
from contextlib import nullcontext
import os
import threading
import cv2
import numpy as np

from .heuristics import Person

SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


_MODEL_LOAD_LOCK = threading.Lock()
# PyTorch MPS shares a Metal command stream across model instances. Concurrent
# camera calls can abort the process, so serialize GPU work and its CPU copies.
_MPS_INFERENCE_LOCK = threading.Lock()


class PoseModel:
    def __init__(self, weights, device):
        if not Path(weights).is_file():
            raise RuntimeError("Pose weights missing. Run scripts/download_models.py first.")
        runtime = Path(weights).resolve().parent / ".runtime"
        runtime.mkdir(parents=True, exist_ok=True)
        (runtime / "ultralytics").mkdir(exist_ok=True)
        os.environ["YOLO_CONFIG_DIR"] = str(runtime / "ultralytics")
        os.environ["YOLO_OFFLINE"] = "true"
        os.environ["YOLO_AUTOINSTALL"] = "false"
        os.environ.setdefault("MPLCONFIGDIR", str(runtime / "matplotlib"))
        from ultralytics import YOLO, settings
        from .tracking import CameraTracker
        with _MODEL_LOAD_LOCK:
            settings.update({"sync": False})
            self.model = YOLO(weights, task="pose")
        self.tracker = CameraTracker()
        self.device = device

    def infer(self, frame):
        mps = self.device.startswith("mps")
        with _MPS_INFERENCE_LOCK if mps else nullcontext():
            try:
                result = self.model.predict(frame, device=self.device, imgsz=640, conf=.35, verbose=False)[0].cpu()
            finally:
                if mps:
                    import torch
                    torch.mps.synchronize()
        if result.boxes is None or result.keypoints is None:
            return []
        boxes = result.boxes.xyxy.cpu().numpy()
        tracks = self.tracker.update(result.boxes.cpu().numpy(), frame)
        # Preserve untracked detections for masking; negative IDs cannot trigger alerts.
        ids = [-1-i for i in range(len(boxes))]
        for track in tracks:
            index = int(track[-1])
            ids[index] = int(track[4])
            boxes[index] = track[:4]
        points = result.keypoints.data.cpu().numpy()
        return [Person(int(i), tuple(map(float, box)), [tuple(map(float, p)) for p in keypoints]) for i, box, keypoints in zip(ids, boxes, points)]


class DepthModel:
    def __init__(self, model_type, device, model_dir):
        import torch
        self.torch = torch
        self.device = device
        # Local repository + local weights only: network setup is a separate explicit command.
        root = Path(model_dir)
        repo = root / "MiDaS"
        weight = root / {"DPT_Hybrid": "dpt_hybrid_384.pt", "DPT_Large": "dpt_large_384.pt", "MiDaS_small": "midas_v21_small_256.pt"}[model_type]
        if not repo.is_dir() or not weight.is_file():
            raise RuntimeError("Depth assets missing. Follow the MiDaS setup in README.md or disable depth.")
        transforms = torch.hub.load(str(repo), "transforms", source="local")
        if model_type == "MiDaS_small":
            # Upstream Small fetches its architecture via torch.hub even with pretrained=False.
            # Redirect only that factory to a pinned local repository and load full weights below.
            import midas.blocks as blocks
            efficientnet = root / "efficientnet"
            if not efficientnet.is_dir():
                raise RuntimeError("Local EfficientNet sources missing. Run download_models.py --depth small.")
            original = blocks._make_pretrained_efficientnet_lite3
            def local_backbone(use_pretrained, exportable=False):
                net = torch.hub.load(str(efficientnet), "tf_efficientnet_lite3", source="local", pretrained=False, exportable=exportable)
                return blocks._make_efficientnet_backbone(net)
            blocks._make_pretrained_efficientnet_lite3 = local_backbone
            try:
                self.model = torch.hub.load(str(repo), model_type, source="local", pretrained=False)
            finally:
                blocks._make_pretrained_efficientnet_lite3 = original
        else:
            self.model = torch.hub.load(str(repo), model_type, source="local", pretrained=False)
        self.model.load_state_dict(torch.load(weight, map_location="cpu", weights_only=True))
        self.model.to(device).eval()
        self.transform = transforms.dpt_transform if model_type.startswith("DPT") else transforms.small_transform

    def infer(self, frame):
        torch = self.torch
        with torch.inference_mode():
            batch = self.transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).to(self.device)
            prediction = self.model(batch)
            output = torch.nn.functional.interpolate(prediction.unsqueeze(1), size=frame.shape[:2], mode="bicubic", align_corners=False).squeeze().cpu().numpy()
        low, high = np.percentile(output, (2, 98))
        return np.clip((output - low) / max(high-low, 1e-6), 0, 1)


def choose_device(requested):
    import torch
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Motion:
    def __init__(self):
        self.previous = None
        self.timestamp = None
        self.residual = None
        self.dt = 0
        self.shape = None
        self.people = {}
        self.old_people = {}

    def infer(self, frame, people, timestamp):
        gray = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY)
        dt = timestamp - self.timestamp if self.timestamp is not None else 0
        local, camera = 0.0, 0.0
        self.residual = None
        self.dt, self.shape = dt, frame.shape
        self.old_people, self.people = self.people, {p.track_id:p for p in people}
        if self.previous is not None and 0 < dt <= .65:
            flow = cv2.calcOpticalFlowFarneback(self.previous, gray, None, .5, 3, 15, 3, 5, 1.2, 0)
            mask = np.zeros(gray.shape, dtype=np.uint8)
            sx, sy = 320/frame.shape[1], 180/frame.shape[0]
            for person in people:
                x1, y1, x2, y2 = person.box
                cv2.rectangle(mask, (int(max(0, x1*sx)), int(max(0, y1*sy))), (int(min(319, x2*sx)), int(min(179, y2*sy))), 1, -1)
            background = flow[mask == 0]
            global_vector = np.median(background, axis=0) if len(background) else np.median(flow.reshape(-1, 2), axis=0)
            diagonal = np.hypot(320, 180)
            camera = float(np.linalg.norm(global_vector) / (dt*diagonal))
            residual = np.linalg.norm(flow-global_vector, axis=2)
            self.residual = residual
            if np.any(mask):
                local = float(np.percentile(residual[mask == 1], 80) / (dt*diagonal))
        self.previous, self.timestamp = gray, timestamp
        return local, camera

    def _region(self, box):
        if self.residual is None or self.dt <= 0:
            return None
        sx,sy=320/self.shape[1],180/self.shape[0]
        x1,y1,x2,y2=box
        return self.residual[max(0,int(y1*sy)):min(180,int(y2*sy)+1),max(0,int(x1*sx)):min(320,int(x2*sx)+1)]

    def person_flow(self, person):
        region=self._region(person.box)
        return float(np.percentile(region,90)/(self.dt*np.hypot(320,180))) if region is not None and region.size else 0.0

    def joint_motion(self, person, index):
        point=person.keypoints[index]
        previous=self.old_people.get(person.track_id)
        if previous is None or min(point[2],previous.keypoints[index][2])<.55:
            return 0.0
        old=previous.keypoints[index]
        pad=person.height*.06
        region=self._region((min(old[0],point[0])-pad,min(old[1],point[1])-pad,max(old[0],point[0])+pad,max(old[1],point[1])+pad))
        return float(np.percentile(region,85)/(self.dt*person.height*(180/self.shape[0]))) if region is not None and region.size else 0.0


def annotate(frame, people, privacy=True):
    image = frame.copy()
    height, width = image.shape[:2]
    for person in people:
        x1, y1, x2, y2 = [int(v) for v in person.box]
        if privacy:
            # Cover an expanded head region even when face keypoints are missing.
            head = [p for p in person.keypoints[:5] if p[2] > .35]
            hx1, hx2 = x1, x2
            hy1, hy2 = y1, y1 + int(person.height*.30)
            if head:
                margin = int(person.height*.09)
                hx1 = min(hx1, int(min(p[0] for p in head))-margin)
                hx2 = max(hx2, int(max(p[0] for p in head))+margin)
                hy1 = min(hy1, int(min(p[1] for p in head))-margin)
                hy2 = max(hy2, int(max(p[1] for p in head))+margin)
            hx1, hx2 = max(0, hx1), min(width, hx2)
            hy1, hy2 = max(0, hy1), min(height, hy2)
            if hx2 > hx1 and hy2 > hy1:
                roi = image[hy1:hy2, hx1:hx2]
                roi[:] = cv2.GaussianBlur(roi, (0, 0), sigmaX=max(15, (hx2-hx1)/3))
        color = (176, 224, 93)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
        cv2.putText(image, f"TRACK {person.track_id}" if person.track_id >= 0 else "ACQUIRING", (max(x1, 0), max(15, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1)
        for a, b in SKELETON:
            p, q = person.keypoints[a], person.keypoints[b]
            if min(p[2], q[2]) > .45:
                cv2.line(image, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 2, cv2.LINE_AA)
    return image


def depth_view(depth):
    return cv2.applyColorMap((depth*255).astype(np.uint8), cv2.COLORMAP_INFERNO)


def synthetic_frame(t, scenario):
    """Explicit fixture data exercises UI/state/evidence, never validates detection accuracy."""
    width, height = 960, 540
    image = np.full((height, width, 3), (23, 21, 19), np.uint8)
    for x in range(-600, 1600, 120):
        cv2.line(image, (480, 170), (x, 540), (42, 39, 35), 1)
    for y in (220, 265, 325, 405, 510):
        cv2.line(image, (0, y), (960, y), (42, 39, 35), 1)
    people = []
    active = scenario == "interaction"
    # The interaction fixture includes overlapping people within real reach;
    # the calm fixture remains separated. This is UI/test data, not inference.
    for i, x in enumerate((405, 525) if active else (290, 670)):
        y = 195
        points = [(x, y, .99)] * 17
        for index, dx, dy in [(0,0,0),(1,-8,-3),(2,8,-3),(3,-15,2),(4,15,2),(5,-32,47),(6,32,47),(7,-48,94),(8,48,94),(9,-45,143),(10,45,143),(11,-21,145),(12,21,145),(13,-26,200),(14,26,200),(15,-32,250),(16,32,250)]:
            swing = np.sin(t*15+i)*85 if active and index in (9, 10) else 0
            points[index] = (float(x+dx+swing), float(y+dy-(55 if active and index in (9, 10) else 0)), .99)
        p = Person(i+1, (x-68, y-25, x+68, y+260), points, .6)
        people.append(p)
        for a, b in SKELETON:
            cv2.line(image, tuple(map(int, points[a][:2])), tuple(map(int, points[b][:2])), (89, 83, 73), 17, cv2.LINE_AA)
        cv2.circle(image, (x,y), 24, (105, 97, 85), -1)
    cv2.putText(image, "SYNTHETIC TEST SCENE / NO CAMERA CONNECTED", (28, 38), cv2.FONT_HERSHEY_SIMPLEX, .52, (160, 165, 170), 1)
    depth = np.tile(np.linspace(.1, .85, height)[:, None], (1, width))
    for p in people:
        x1,y1,x2,y2 = map(int,p.box)
        cv2.rectangle(depth, (x1,y1), (x2,y2), .6, -1)
    return image, people, depth, (.24 if active else .004), (.2 if scenario == "camera" else 0)
