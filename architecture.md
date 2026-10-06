# VDM Shield — current architecture and data flow

**Updated: 27 September 2026.** This document describes the current local working tree, including changes that have not yet been committed, and the explicitly labelled next-stage design. It covers required resources, components, models, data types, processing, storage and response flow. [HANDOFF.md](HANDOFF.md) records implementation checkpoints; this file explains how the components fit together. This documentation update changes no application code or runtime settings.

**Reading guide:** requirements and setup are in section 2; the full data path in section 3; tools/models and camera settings in sections 4–7; data/storage/response behavior in sections 8–14; training and connectivity in sections 15–16; the planned local System 1 layer in section 17; and the separate Vision Lab in section 18.

## October 4 addition: Gemini incident review and voice gateway

This trimmed workspace now includes an optional Vertex AI layer in `vmd/gemini.py`, configured through private `.env` settings and OAuth/ADC. It analyzes bounded incident frames, generates structured second-opinion reports and timelines, answers evidence questions, and follows fresh aftermath windows for at most two minutes using a separate rendered-frame buffer. Reports and observations are saved in additive SQLite tables and versioned against the incident evidence. Operator-approved briefings can enrich the existing Twilio voice/SMS text, while detection and dispatch continue independently.

`vmd/voice.py` is a separate loopback service on 8769 for signed Twilio callbacks and bidirectional Gemini Live audio. It exposes no dashboard/API routes, uses expiring one-use call-bound session tokens, and provides a read-only context tool. It requires separate HTTPS hosting and a Vertex-accessible Live model. See [Gemini setup](docs/gemini-setup.md) for configuration, limits and verification. Real cloud/phone commissioning remains pending; local pose, tracking, depth, specialist weapons and the standalone model labs are preserved. Older planned-only descriptions below refer to the separate local System 1 design and predate this optional cloud layer.

## 1. What the system currently is

VDM is a local video-monitoring application. A Python service reads cameras or recordings, runs vision models and temporal rules, displays processed feeds, stores incidents, and coordinates a response through alarms and configured contacts.

The feature called the **response agent** is an autonomous, rule-based workflow around AI detections: it observes incidents, maintains a deadline, requests configured actions, remembers attempts, and monitors provider status. Its decisions are implemented as Python conditions and SQLite state transitions. The current calling workflow does not use an LLM, GLiNER, a conversational agent framework, speech recognition, or an AI-generated narrative.

The machine-learning components currently perform **pose estimation, object detection and relative-depth estimation**. The fight decision combines those observations with motion and temporal rules; it is not a trained video violence classifier. Twilio supplies speech synthesis and telephone delivery after the response agent requests a call.

**Planned addition:** we will use a **local System 1–style classification model for incident hypotheses**, after the vision/motion/depth evidence has been assembled. The intended starting candidate is the previously discussed **GLiNER2.5-Decide**. It is an additional interpretation layer, separate from the existing response state machine. The classifier, evidence-summary adapter and hypothesis API described later are planned, not installed or connected to the main application yet.

### Current implementation status

| Area | Current status |
| --- | --- |
| Multiple cameras, tracking, fight/posture rules | Implemented; maximum four camera/analysis sessions, with finished unsaved recordings recyclable. |
| ZipDepth and depth-confirmed fights | Implemented; available alongside the earlier responsive mode and MiDaS/DPT alternatives. |
| Guarded punches, strike timer and standing snatching | Implemented; exact contact-depth checks, configurable fight escalation and one-incident stage upgrades. |
| Knife/gun observations and saved incidents | Implemented using the pinned Assalim YOLOv8n checkpoint. Reported gun false positives remain unresolved. |
| Eco scheduling and camera recovery | Implemented, with saved settings and explicit manual-stop behavior. |
| Centre account and authority/hospital contacts | Implemented for one centre per installation. |
| Alarm, ten-second escalation, manual dispatch | Implemented as a persistent response state machine. |
| Dynamic Twilio incident announcements | Implemented through a Twimlets Echo URL. Hosting/encoding checks passed; a real custom-announcement call on this account remains unverified. |
| SMS with location and an expiring clip link | Implemented as an optional, separately configured delivery path. |
| Public Cloudflare evidence tunnel | Implemented: managed Quick Tunnel, independent one-hour link generation/revocation and verified public sample playback. Messaging API delivery is separate. |
| Event details with a video link | Implemented: authenticated local POST returns saved incident metadata and a fresh one-hour clip URL; no dispatch or outgoing API submission. |
| Improved weapon model training | Colab notebook and evaluation workflow created; no replacement model trained on the full dataset or deployed yet. |
| Local System 1–style incident-hypothesis classification | Planned: evidence summary → local classifier → structured hypotheses → existing policy/review flow. GLiNER2.5-Decide is the intended starting candidate. |
| Small vision-language model experiment | Implemented only in the separate Vision Lab on port 8770; no main-app integration. |
| Temporal knowledge graph and learning from reviewed clips | Deferred; no automatic graph construction or retraining loop in the main app. |

The main dashboard binds to loopback port **8765**. The optional evidence gateway uses **8768** when explicitly started; the independent Vision Lab uses **8770**. Calling and messaging each need their own enabled/configured state. These addresses describe service roles, not proof that every optional service is running. No real contact numbers, credentials, camera URLs or recordings are reproduced here.

## 2. Required resources and setup

| Requirement | Needed for | Details |
| --- | --- | --- |
| Local computer with CPU, available RAM and writable disk | All main-app operation | The current development machine is an Apple Silicon M2 laptop with 16 GB unified memory. This is a tested machine, not a measured minimum specification. CPU inference is supported; model count, camera count and sampling rates determine actual capacity. |
| Python 3.10–3.13 and a virtual environment | Backend and inference | Declared in `pyproject.toml`; this checkout uses Python 3.11. |
| FastAPI, Uvicorn, Pydantic, NumPy and OpenCV | HTTP API, validation, decoding and image operations | Base dependencies install through the project package; Pydantic is used through FastAPI. The app uses FastAPI, not Flask. |
| Ultralytics, PyTorch, torchvision, `lap` | Pose, tracking and weapon models | Installed through the `vision` extra; versions and bounds are in `pyproject.toml`. |
| ONNX Runtime; local ZipDepth model and manifest | Preferred depth path | ONNX inference is local on CPU. `timm==0.6.13` and `einops` support the optional legacy depth paths. The `depth-export` extra is for initial ONNX preparation, not a cloud runtime. |
| Local model assets under `models/` | Neural inference | YOLO11n pose, ZipDepth, and, when weapons are enabled, the specialist and person-mask detector. Optional MiDaS/DPT assets are needed only when selected. |
| Webcam permission, reachable IP stream or supported recording | Input | IP cameras require the video endpoint, not their administration page. LAN client isolation can prevent access despite a shared Wi-Fi name. |
| Modern browser with JavaScript | Dashboard and media player | Native HTML/CSS/ES modules; no Node/frontend build step is needed to run the UI. Web Audio needs an operator gesture; geolocation needs browser/OS permission. |
| SQLite and writable `data/` | Persistence | SQLite is embedded through Python. Allow disk space for uploaded originals, incident AVI files and additional MP4 playback caches. No cloud database is required. |
| FFmpeg or `imageio-ffmpeg` fallback | Browser-playable evidence | Converts local media into H.264 MP4; the Python fallback package is a base dependency. |
| Centre registration and configured authority/hospital contacts | Authenticated operation and response destinations | One centre per installation. Credentials and contacts stay in private configuration/database storage. |
| Twilio account, voice-capable sender, credentials and allowed destinations | Optional telephone calls | Internet is required for submission, TwiML retrieval and status polling. The Python adapter uses REST; it does not require the Twilio SDK. |
| SMS configuration, public evidence URL and running gateway/tunnel | Optional clip-link messages | Separate from voice. A managed temporary Cloudflare tunnel provides clip hosting; the laptop must remain online. |
| Pinned local classifier package/checkpoint and additional measured memory budget | Planned System 1 layer only | Not a current installation requirement. Benchmark alongside the existing vision workers before enabling it. No extra cloud inference account is planned for classification. |

Initial installation needs internet for packages and model downloads. A fresh-machine setup, run from the repository directory, is:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[vision,depth-export]'
.venv/bin/python scripts/download_models.py --depth zipdepth --objects
.venv/bin/python -m vmd
```

Open `http://127.0.0.1:8765`, register/sign in to the centre, and add a camera or upload a recording. Once dependencies and models are installed, local detection does not require internet. The source still must be reachable; calls, remote clip access and external map tiles have their separate network requirements. Development tests additionally use the `test` extra and Node's built-in test runner. These commands document setup; this architecture-only update has not executed installation or downloaded a classifier.

## 3. End-to-end flow in text

### Complete data path at a glance

```text
Webcam / reachable IP video stream / uploaded recording
  → OpenCV decoding → frame + camera ID + sequence + source time
  → per-camera Engine
      ├─ live Eco scheduler (recordings bypass quiet-scene skipping)
      ├─ YOLO11n pose → ByteTrack → optical flow + pose stabilization
      ├─ sampled ZipDepth / optional MiDaS → original-frame depth matches
      └─ sampled YOLO weapon detector → fresh gun/knife observations
  → timestamped observations and short interaction histories
      ├─ CURRENT: anatomical/motion/depth/timer gates or weapon episode gate
      │    → typed incident + reasons + score + camera/location snapshot
      │    ├─ SQLite metadata + local evidence files → Incident Library/player
      │    ├─ real saved-data queries → analytics/map/heatmap
      │    └─ eligible live incident → AlertAgent countdown/operator action
      │         ├─ browser alarm
      │         ├─ voice worker → TwiML URL → Twilio → configured phone
      │         └─ optional delivery → MP4 + token → SMS video/map link
      │              → recipient → public tunnel → evidence gateway → local MP4
      └─ PLANNED: bounded evidence summary → local System 1 classifier
           → typed incident hypotheses + scores + attached evidence references
           → advisory inspector/review data; evaluate before any policy role

Browser controls → authenticated FastAPI → camera settings / review / dispatch
Operator review → saved review label; future curated training, not online learning
```

Pixel arrays stay inside local processes. API JSON carries state and metadata; previews use JPEG bytes/base64; evidence playback uses video bytes. Calls send announcement instructions, while SMS sends a link whose target serves the video. The planned classifier receives compact text derived from observations, not those video bytes.

### Live monitoring and incident creation

1. **Camera → capture:** a webcam index or network video URL is opened by OpenCV. Compressed video becomes an in-memory BGR pixel array.
2. **Capture → camera engine:** the newest frame is supplied with a sequence number and source timestamp. An old frame backlog is not retained.
3. **Engine → eco gate:** a small grayscale representation determines whether expensive inference should run at the configured rate or a reduced quiet-scene rate.
4. **Engine → pose/motion path:** YOLO11n pose produces person boxes and keypoints; ByteTrack assigns camera-local IDs; optical flow and pose stabilization produce motion features.
5. **Engine → depth worker:** selected frames go to a separate process. Relative depth is returned with the original frame identity and matched to that frame's saved pose observations.
6. **Engine → object worker:** selected frames go to a separate process. The specialist returns gun/knife boxes; a separate person detector supports head blurring of that preview.
7. **Observations → incident decisions:** fight/posture rules or the weapon episode gate produce typed incident events, scores, reasons and signal metadata.
8. **Engine → previews and evidence:** processed frames become JPEG bytes. A short evidence buffer and a bounded storage queue feed clip creation and SQLite persistence.
9. **Backend → browser:** authenticated HTTP polling supplies camera state, base64 JPEG previews, incidents, analytics and response status.

### Response and calls

1. An accepted incident storage job invokes `AlertAgent.register()` immediately, without waiting for clip encoding.
2. A new eligible live-camera confirmed fight, knife or confirmed snatching incident creates a persistent `pending` response with a ten-second deadline. Gun and provisional incidents remain manually dispatchable.
3. The browser reads that state and plays increasingly loud beeps after sound has been enabled by the operator.
4. The deadline or a manual slide changes the response to `requested`.
5. The calling worker reads the centre's saved contacts, reserves one attempt per incident/phone, and builds the incident announcement.
6. The announcement becomes TwiML XML, then a URL-encoded Twimlets Echo URL.
7. The worker submits `To`, `From` and `Url` to Twilio's Calls API.
8. Twilio retrieves the instructions and, if the call connects and the instructions are accepted, synthesizes the announcement to the recipient.
9. The worker polls the returned Call SID and persists the provider status. The browser displays a compact result and detailed history in Centre settings.

### Optional clip delivery

A requested response can independently enter the SMS worker when messaging is enabled and configured. That path requires saved location and evidence, converts the clip to browser-playable MP4, creates an expiring token, and sends an SMS containing a map link and video URL. The recipient's browser retrieves video bytes from the separate evidence gateway. Telephone speech does not transport the video.

## 4. Runtime components, tools and responsibilities

| Component / implementation | Tool or technique | Input data | Output / responsibility |
| --- | --- | --- | --- |
| [Application entry](vmd/__main__.py), [API](vmd/api.py) | Python, Uvicorn, FastAPI, Pydantic | HTTP requests and validated JSON | Starts shared services, exposes APIs and static UI, enforces sessions and local-origin rules. |
| [Dashboard](vmd/static/app.js) and browser modules | HTML, CSS, JavaScript ES modules | API JSON, JPEG previews, media URLs | Camera grid, inspector, evidence library, settings, review, dispatch and health views. |
| [Centre](vmd/centre.py) | Pydantic, scrypt, token hashing, SQLite | Centre name, password, contacts and toggles | One registered centre, authenticated sessions, saved authority/hospital destinations. |
| [CameraManager](vmd/cameras.py) | Python supervisor thread and atomic JSON writes | Validated camera settings | Creates/stops/restarts engines and persists connection settings. |
| [Capture](vmd/capture.py) | OpenCV `VideoCapture`, FFmpeg backend for network input | Device index, HTTP(S)/RTSP(S) URL or local video path | Latest live packet, or one acknowledged selected file packet at a time. |
| [Engine](vmd/engine.py) | One processing thread per camera | Latest capture packet and settings | Coordinates model sampling, rules, previews, evidence and telemetry. |
| [EcoGate](vmd/eco.py) | OpenCV/NumPy image differences | Downscaled grayscale frames | `active`/`quiet` state and per-model scheduling decisions. |
| [PoseModel / Motion](vmd/vision.py) | Ultralytics YOLO11n pose, PyTorch, Farneback optical flow | BGR frame | Person boxes, COCO keypoints and image-motion measurements. |
| [CameraTracker](vmd/tracking.py), [PoseStabilizer](vmd/stabilization.py) | ByteTrack and track-local filtering | Person detections across time | Camera-local track IDs, reliable poses and motion-supported limb speeds. |
| [DepthWorker](vmd/depth_worker.py), [ZipDepth](vmd/depth.py) | Spawned process, bounded queues, ONNX Runtime CPU | Sampled frame plus identity/timestamps | Normalized inverse-depth map, JPEG visualization, latency and status. |
| [BehaviorHeuristic](vmd/behavior.py), [FightHeuristic](vmd/heuristics.py), [FightConfirmation](vmd/confirmation.py), [spatial checks](vmd/spatial.py) | Geometry, motion and temporal rules | Stabilized tracks, flow, optional matched depth | Fight/posture assessments, typed events, scores, reasons and blockers. |
| [SnatchingHeuristic](vmd/snatching.py) | Neck reach, supported pull, same-actor departure and original grab depth | Short per-track movement histories | Possible snatching and confirmed snatching hypotheses from movement; not jewelry recognition. |
| [ObjectWorker / WeaponModel](vmd/objects.py) | Separate CPU process; two YOLO detection passes | Sampled frame | Knife/gun boxes and a preview whose detected heads are blurred. |
| [Store / EvidenceBuffer](vmd/storage.py) | JPEG deque, bounded queue, SQLite, OpenCV MJPEG encoder | Events, sampled frames, crowd telemetry | Persistent incident metadata and local AVI evidence. |
| [MediaLibrary](vmd/media.py), [player](vmd/static/player.mjs) | OpenCV validation, FFmpeg H.264 conversion, HTML video | Uploaded files or saved evidence | Local video catalogue, cached MP4 playback, seek/speed controls. |
| [AlertAgent](vmd/alerts.py) | SQLite-backed state machine and timer thread | Live incidents and operator actions | Durable countdowns, acknowledgements, cancellation and response requests. |
| [Response UI](vmd/static/response.mjs) | Web Audio and HTTP actions | Response state and user gestures | Beeps, countdown, gray dispatch lock and hospital dialog. |
| [Calling](vmd/calling.py) | Python `urllib` HTTPS REST, XML ElementTree, Twilio/Twimlets | Response request, contact, incident snapshot | One call attempt per recipient and saved call status. No Twilio SDK is required by this adapter. |
| [Delivery](vmd/delivery.py), [evidence gateway](vmd/share.py) | Twilio Messages REST API, random expiring tokens, FastAPI file serving | Response package and cached MP4 | Optional SMS with a remotely retrievable clip URL. |
| [Training notebook](notebooks/train_vdma_weapons_colab.ipynb) | Google Colab, Ultralytics/PyTorch, Open Images, review widgets | Labelled positives and reviewed hard negatives | Candidate checkpoint, thresholds, provenance and evaluation reports. Separate from the live app. |
| Planned hypothesis classifier — no production module yet | Local System 1–style text classifier; intended candidate GLiNER2.5-Decide | Bounded evidence summary and fixed label schema | Advisory incident hypotheses. No direct call, camera-control or database mutation tool access. |

## 5. Camera input, clocks and recovery

### Accepted inputs and frame representation

`StartRequest` accepts a source and settings such as device, depth model, detection mode, eco mode, object sampling rate, fight threshold, hold duration and camera location. Source validation accepts webcam indexes 0–9, supported network schemes, or existing supported local videos. An uploaded recording is resolved from its server-generated `recording_id` rather than trusting an arbitrary browser file path.

The inspector's Camera controls also exposes **Edit detection settings**. It submits only the existing tuning fields to `PATCH /api/cameras/{id}/settings`. CameraManager validates the merged settings before stopping anything, atomically saves physical-camera configuration, and restarts enabled cameras while keeping stopped cameras stopped. Source, identity, location and incidents remain intact. The editor and add-camera form share controls/limits. Worker freshness and pair-depth compatibility are displayed separately.

| Editable field | Limits / starting values | Effect |
| --- | --- | --- |
| `device` | `auto`, `cpu`, `mps`, `cuda:0`; default `auto` | Chooses pose/legacy-depth compute. ZipDepth and object workers remain CPU-only. |
| `depth` | `off`, `ZipDepth`, `MiDaS_small`, `DPT_Hybrid`, `DPT_Large` | Selects relative-depth estimator; assets must exist. UI starts with ZipDepth; bare API default is off for compatibility. |
| `detection_mode` | `depth_confirmed` or `responsive` | UI starts Depth-confirmed; the API's legacy default is `responsive` (Review only), which cannot confirm fights. |
| `depth_fps` | 0.1–2; UI 1, API 0.5 | Controls regular depth sampling; Depth-confirmed requires a model and at least 0.5 Hz. Contact priorities can add bounded samples. |
| `threshold` | 0.40–0.95; default 0.60 | Fight heuristic score cutoff. Lower is more sensitive; it cannot bypass geometry/depth gates or change weapon confidence. |
| `hold_seconds` | 0.3–5; default 0.7 | Provisional sustained-grappling observation window; explicit supported strikes use their contact checks. |
| `fight_confirmation_seconds` | 0.5–10; default 3 | Required supported fight span. Reciprocal/continuing evidence and fresh depth remain mandatory. Some legacy option copy still mentions three seconds; the saved numeric field is authoritative. |
| `target_fps` | Integer 2–30; default 8 | Desired pose-analysis rate, not guaranteed throughput. For files, controls selected source frames. |
| `eco_mode` | Boolean; default false | Reduces quiet live-scene inference. Recording analysis does not skip selected frames for Eco. |
| `object_detection` | Boolean; UI true, API false | Enables the independent gun/knife worker. |
| `object_fps` | 0.2–2; default 1 | Weapon-worker sample rate; independent of pose FPS. |

OpenCV decodes frames to `numpy.ndarray`, normally `uint8` with shape `[height, width, 3]` in BGR channel order. Live capture preserves aspect ratio, caps width at 960 pixels, and uses even dimensions for evidence encoding. Network capture has five-second open/read timeouts. The live capture thread owns the decoder and replaces its latest packet; inference can skip intermediate frames instead of lagging behind a growing queue. **File capture is different:** it samples the requested FPS, also caps height at 480, and waits for the engine to consume each selected packet before handing off the next. Model warmup does not silently skip selected recording frames.

Three time concepts must remain distinct:

| Field / clock | Meaning | Use |
| --- | --- | --- |
| `sequence` | Increasing frame number within the capture session | Match asynchronous outputs to their original input. |
| `source_time` / `signals.source_seconds` | Monotonic capture time for live feeds; elapsed recording seconds for files | Motion, temporal rules, evidence ordering and playback offsets. It is not a UTC date. |
| `created`, `submitted_at`, `updated`, deadlines | Unix wall-clock seconds | Incident detection/processing time, response deadlines, persistence and call announcements. |

The announced time is the saved incident's `created` time converted to the server's local timezone, including date/timezone. It is not the later dispatch time and is not guaranteed to equal a camera's hardware exposure timestamp. Asynchronous weapon inference can introduce delay between a sampled frame and the recorded incident time.

### Automatic recovery

Only live device/network sources are persisted for automatic reconnection. `data/cameras.json` contains the source, settings and enabled flag; it is written via an atomic replacement with owner-only permissions. It may contain private stream credentials and must stay out of Git.

The supervisor checks approximately every second. Failed/finished/idle/stopping sessions, or running sessions whose processed frames have stalled beyond 15 seconds, can be restarted. Initial retry is scheduled after five seconds; subsequent delays grow and are capped at 60 seconds. Recovery uses the existing settings object, so it retains the source, model choices, thresholds, sampling rates, eco setting and location. A manually stopped camera is persisted as disabled and remains stopped. Adding a new camera uses the setup form. Recordings finish at EOF and are not automatically looped or restored as cameras.

Recovery operates while the server is alive. It is not an OS service manager that restarts a dead Uvicorn process.

## 6. Vision inference, models and decision logic

### Model inventory

| Purpose | Current model/tool | Execution | Output meaning |
| --- | --- | --- | --- |
| Person pose | `yolo11n-pose.pt` | PyTorch; `auto` selects CUDA, then Apple MPS, then CPU | Person boxes and 17 COCO keypoints, each with coordinate/confidence data. |
| Preferred lightweight depth | `zipdepth-91f3fd2.onnx` plus verified manifest | ONNX Runtime CPU, one thread per runtime pool | Frame-normalized relative inverse depth in `[0,1]`; no metric distance. |
| Optional legacy depth | MiDaS Small, DPT Hybrid, DPT Large | Local PyTorch assets in a depth process | Relative-depth maps, not metres. Availability depends on installed assets. |
| Public weapon detections | Assalim `Normal_Compressed/best.pt`, stored as `assalim-normal-compressed-best.pt` | YOLOv8n on CPU in the object worker | `0: guns`, `1: knife`; public `guns` label becomes `gun`. |
| Object-preview privacy | `yolo26s.pt`, person class only | CPU in the object worker | Person boxes for approximate head blurring; other COCO labels are not public detections. |
| Replacement weapon candidate | Fresh pretrained YOLOv8s by default in the Colab notebook | Colab GPU during training | Same gun/knife class order; not deployed in VDM yet. |

The specialist and ZipDepth assets have integrity checks. The weapon loader validates class order and uses restricted `torch.load(..., weights_only=True)` with an explicit installed-class allowlist. The current specialist is enabled for both classes at a 0.90 inference cutoff. `weapon_detections()` and the live UI require confidence **strictly above 0.90**; exactly 90% is rejected. Raw confidence is preserved until display formatting. The same filter controls preview boxes, live actions and new weapon incidents. Changing the camera's **fight threshold** does not change these weapon cutoffs.

### Pose, tracking and motion

`PoseModel.infer()` predicts at image size 640 with a 0.35 detector confidence cutoff. ByteTrack associates detections over time. Track IDs belong to one camera/tracker lifetime; they are not person identities and are not cross-camera identification. Untracked boxes can support privacy masking, but negative track IDs cannot trigger tracked-person rules.

`Motion.infer()` computes Farneback optical flow on a 320×180 grayscale frame. It estimates background camera movement, subtracts it, and measures residual movement in person/joint regions. `PoseStabilizer` anchors to visible torso joints, applies smoothing/deadbands, rejects large discontinuities, and requires image-motion support before interpreting pose displacement as limb speed. Unreliable poses and gaps over the supported timing window reset motion history.

`FightHeuristic` considers proximity, limb motion, contact/strike evidence, repeated activity and camera motion. Its score is a rule score, not a calibrated probability of violence. `BehaviorHeuristic` also handles `possible_fall`, `person_down`, `person_down_after_fight` and `hands_up` with track-local temporal evidence. These observations require review; posture does not establish intent.

### Review warnings, confirmed fights and standing snatching

Every main-app fight goes through `FightConfirmation`. An accepted supported strike immediately produces `possible_fight` for review without automatic response. Anatomical head/torso regions include a bounded torso-length allowance for side views; empty person-box space does not count as contact. A supported reach can connect nonoverlapping boxes, but a review alert then requires compatible depth. Slow contacts also require compatible depth from their exact original contact frame before onset. Pending contact checks show `Checking interaction`. Fresh same-frame separation vetoes both stages. The saved `hold_seconds` (default 0.7) remains the provisional observation window for generic sustained grappling; it does not delay accepted explicit strikes. The legacy `responsive` mode cannot confirm fights; its separation veto remains active.

**Depth-confirmed** mode requires the same reliable pair, image-supported contact, a continuing strike exchange or sustained grappling, and the configured **fight_confirmation_seconds** interaction span (0.5–10 seconds, default 3). This backend source-time clock runs from accepted onset to the latest supported observation. During a withdrawal its displayed value freezes; support returning within 1.5 seconds continues the episode, while longer interruptions reset the timer. Silence cannot confirm. Continuous bilateral grappling integrates valid supported intervals separately. A nearby reliable pair can reuse its incident/handling flags for up to 12 seconds while collecting fresh evidence. Tracking/frame gaps, camera motion, missing poses and separation invalidate continuity.

Confirmation requires at least two distinct fresh compatible depth samples, with a current compatible result within the 2.5-second source-time freshness bound. There is no additional full-duration depth clock: `confirmation_seconds` reports the matched-depth span for diagnostics, while `fight_timer_seconds` controls escalation. Missing, uncertain, stale or separated depth cannot emit `fight`, regardless of motion strength. New candidates and directed contacts request bounded priority depth, with exact original sequence/time/shape joins; sampling and inference latency can delay acceptance.

Slower reciprocal punches have a separate evidence path in `FightHeuristic`: filtered wrist displacement relative to the actor's shoulders, measured motion on that wrist, and a torso-relative approach/extension toward the other person's confident torso/head. Directed contacts record actor, wrist, target person, body/guard surface, target joint and source timestamps. A bent raised defensive guard can intercept a supported punch before it reaches the body; a low hanging hand below the hips cannot use padded torso bounds. Exact depth compares the attacking wrist with the reached head/torso/guard, checking local patch noise and distal forearm coherence. Torso-only depth remains separate for snatching and ordinary fast/grappling confirmation; the latter cannot borrow an unrelated arm’s matching surface. Known torso separation resets fast evidence even when a directed slow contact can remain valid. Gross torso separation vetoes all fight contacts. Accepted strikes from both people across a continuing exchange can confirm before full withdrawals complete; there is no three-completed-cycle quota. Simultaneous two-arm contact alone, a hug, wave or unilateral slow movement cannot qualify from activity alone. The saved score threshold still applies, and slow motion does not inflate fast-strike or grappling counters. Exact compatible depth at each accepted slow contact is mandatory. A brief contact whose depth arrives after withdrawal is credited once at its source time, within motion grace. Supported movement with short uncertain contact depth can preserve an already-started timer while its value freezes, but cannot start or confirm a fight and cannot outlive depth freshness. Separation clears slow trajectories; expired/unavailable depth requires fresh evidence. Interior torso samples and bounded near-contact tolerance accommodate edge pixels and filter lag without bypassing separation. Incident identity and report flags prevent duplicate escalation. No new model, worker or external inference service is added.

Each live assessment publishes `fight_timer_seconds`, `fight_timer_required_seconds`, `fight_timer_started`, `fight_timer_state` and `fight_timer_reason`. States distinguish counting, paused, waiting for depth, waiting for continuing strikes, confirmed and idle. Before an accepted onset, `fight_timer_started=false` keeps the label at Checking interaction. Inspector, camera cards and full feed render this live state independently of saved incidents; they do not extrapolate browser time and hide progress for stopped/stale cameras. Confirmation is explicit backend state, never inferred from a full progress bar.

`SnatchingHeuristic` reuses stabilized person poses and measured wrist/ankle image motion (or a stricter observed upper-body departure path when ankles are cropped). For standing people it follows one actor's wrist approaching another person's neck/upper chest, then a sharp supported withdrawal within one second. That can create `possible_snatching`; known uncertain/separated depth at the original contact vetoes the warning, and a late compatible original-frame result may restore it. An upgrade to `snatching_detected` requires the same actor moving rapidly away, with several trajectory samples and supported leg motion (or the stricter cropped-leg torso departure evidence), within five seconds. A victim or unrelated person running cannot upgrade it. Matching depth must support the original contact within ±0.15 seconds, rather than a later pull or departure. The observed reach requests a bounded priority depth sample between ordinary samples; priority requests are limited to two per second and still use the single-item queue. A later result is joined to the exact original frame/poses. This is a movement hypothesis; the system does not detect jewelry or establish property removal. Motorcycle snatching remains deferred.

The engine preserves original raw poses for each submitted depth frame, including source time and shape. A bounded 32-frame cache is joined by sequence, exact source time and shape. Flat/uncertain maps cannot support escalation. Snatching keeps only a short depth history and clears it on discontinuities; old observations cannot attach to a reacquired identity.

A provisional event and its escalation share one UUID for that camera/pair episode. Storage upgrades the row, retains its first detection time, review decision and handling time, and appends stage metadata. The fuller escalation clip gets a separate filename; original evidence remains available on disk. A later weaker observation cannot downgrade a stronger saved event. Response registration starts its ten-second countdown at escalation, and preserves existing manual dispatch/cancellation/acknowledgement, recipients and deadlines, preventing a second call for the same incident.

### Interaction processing budget

New analysis sessions default to **8 FPS**, and the grid/detail preview polls at up to **8 updates/second**. Saved camera rates remain explicit settings and are preserved. Depth retains its independent configured rate. Latest-frame capture and bounded workers avoid a backlog. Detailed limb checks skip distant pairs, pose/motion evidence is shared with snatching, and pending grabs keep Eco active. `stage_ms` exposes the latest pose, motion/stabilization, rules and rendering/encoding durations; async depth/object latency remains separate. These are timings, not accuracy or energy measurements.

### Weapon episodes

Weapon detections do not require two people, a person count, or fight confirmation. A fresh sample can update the live action immediately. `WeaponIncidents` retains the strongest observation per class, prevents repeated processing of the same sample, and saves at most one incident per sustained visible episode, with a disappearance gap and ten-second saved-event cooldown before rearming. This is episode suppression, not a requirement for several consecutive positive weapon frames.

A single confident false detection can therefore still become a weapon incident. The notebook is intended to address that model/data weakness; no accuracy improvement is claimed until a new model is evaluated. Stale, failed or stopped object results are removed from the public observation/action state.

The object worker budgets one OpenCV and one PyTorch CPU thread. Both person and weapon models register an `on_predict_start` callback to restore the PyTorch limit after Ultralytics' lazy device setup, before warmup/inference. This avoids that setup silently expanding the background worker to the machine-wide thread count; it does not alter model weights, input size, classes or confidence. The main pose model and its MPS serialization guard are independent of this worker process.

## 7. Eco mode and process boundaries

Eco mode is a scheduler, not a separate neural network. It resizes frames to at most 160×90 grayscale pixels, blurs them, and compares both the previous image and a slowly updated background. Motion or active/pending incident evidence keeps analysis active for another ten seconds.

| Work | Active scene | Quiet scene target |
| --- | --- | --- |
| Capture/decode | Remains connected | Remains connected |
| Pose/motion pipeline | Configured target rate | At most about 2 analyses/second, subject to the configured limit |
| Depth submissions | Configured depth rate | About 0.1/second, capped by configuration |
| Object submissions | Configured object rate | About 0.5/second, capped by configuration |

Actual rates depend on processing capacity. Eco counters measure skipped work/submissions; they are not measured battery savings. The skip happens before the evidence-buffer append, so skipped frames are not retained as full-rate footage. This matters for future clip-description work and prevents describing the existing buffer as a continuous DVR.

The main process hosts FastAPI, per-camera engine/capture threads, the camera recovery supervisor, evidence writer, alert timer, call worker and optional SMS worker. Each enabled camera can also own a spawned depth process and a spawned object process. Their request/response queues each hold at most one pending item and replace stale work for live processing. Recording analysis waits for scheduled/priority depth results with bounded timeouts; its object worker remains asynchronous. Model failure in a child is reported independently of the pose path.

MPS pose prediction, CPU result transfer and GPU synchronization share a process-wide lock. This was added after concurrent Metal inference caused native process crashes. CPU tracking has its own small lock around ByteTrack's global ID counter. Worker processes watch the parent's process sentinel and exit after parent death, avoiding orphaned inference workers. `faulthandler` emits native-crash thread diagnostics. These changes improve observed stability; they do not guarantee indefinite uptime.

## 8. Data contracts between components

| Data | Representation | Producer → consumer | Stored? |
| --- | --- | --- | --- |
| Camera configuration | Validated Pydantic model / JSON | Setup API → CameraManager → Engine | Private `cameras.json` for live sources. |
| Capture packet | `(int sequence, float source_time, uint8 BGR ndarray)` | Capture → Engine | Latest live packet or acknowledged file packet in memory. |
| Person observation | `Person` dataclass: track ID, pixel `xyxy` box, keypoint `(x,y,confidence)` tuples, reliability/motion values | Pose/tracker/stabilizer → heuristics | Transient; selected signals/track pairs enter incidents. |
| Worker request | `(sequence, source_time, submitted_at, copied frame)` | Engine → depth/object process | Bounded multiprocessing queue. |
| Depth result | Status, IDs/times, `float16[H,W]` map, JPEG bytes, latency | Depth process → Engine | Latest result/history in memory. |
| Weapon result | List of label, confidence, pixel `xyxy` box, detector name; sampled JPEG and identity | Object process → Engine | Latest result plus selected incident metadata. |
| Assessment | Rule score, state, reasons, signals, candidate track pair and event list | Behavior rules → Engine | Event subset is saved. |
| Planned hypothesis data | Bounded evidence-summary text in; typed labels/scores, episode/window/model revision and evidence references out | Future summary adapter → local classifier → advisory review | Proposed episode metadata; no current persisted classifier schema. |
| Evidence buffer | List/deque of `(source_time, JPEG bytes)` | Engine → Store | Memory, maximum ten seconds and 32 MiB per engine. |
| Incident | ID, wall-clock creation time, camera, mode, type, score, reasons, signals | Engine → Store / AlertAgent | SQLite; clip reference points to a file. |
| Browser frame bundle | JSON metadata plus base64 JPEG strings | API → browser | Browser memory; not a direct camera stream. |
| Response request | Incident ID, action enum, optional hospital boolean | Browser or timer → AlertAgent | SQLite response row. |
| Call instructions | Escaped TwiML XML, embedded in URL query | Calling → Twilio/Twimlets | Generated for submission; not put in ordinary UI logs. |
| Call result | Call SID, status, sanitized error, timestamps, recipient snapshot | Twilio/Calling → SQLite → browser | `response_calls`. |
| Response package | JSON incident/location/recipient categories plus clip reference | AlertAgent → optional Delivery or ZIP endpoint | Generated from the saved snapshot. |
| Evidence download | AVI original or cached H.264 MP4 bytes | Store/MediaLibrary/gateway → browser | Files under the private data directory. |
| Training labels | YOLO text: `class cx cy width height`, normalized to `[0,1]` | Reviewed dataset → Colab trainer | Training workspace/Drive; no automatic live feedback loop. |

Bounding boxes in runtime incidents are pixels in the processed frame. Training labels are normalized coordinates. Do not interchange them without image dimensions and conversion. A weapon model confidence and a heuristic fight score have different meanings even though both occupy the incident's numeric `score` field.

Example incident structure, with fictional values and a shortened ID for illustration:

```json
{
  "id": "example-incident",
  "created": 1790400000.0,
  "mode": "live",
  "camera_id": "cam-example",
  "camera_name": "Example gate camera",
  "event_type": "knife_detected",
  "score": 0.97,
  "reasons": ["Knife detected; review required"],
  "signals": {
    "live_camera": true,
    "location": {"place": "Example gate", "latitude": 0.0, "longitude": 0.0},
    "object_detection": true,
    "object_label": "knife",
    "box": [110.0, 120.0, 160.0, 240.0],
    "source_seconds": 12345.6,
    "sequence": 420,
    "buffer_seconds": 8.2
  }
}
```

Real incident IDs are generated UUID hex strings. The browser acquires laptop coordinates through one shared `MapLocation` instance. Camera setup uses that fix for physical webcams/IP streams; fresh fixes also initialize existing physical cameras without a location. An optional place name defaults to `Laptop location`. This is the browser device’s position, not independent GPS from the phone/IP camera; a remotely mounted camera is not automatically geolocated by knowing its stream URL. `CameraLocation` stores coordinates, source (`browser` or legacy `manual`), accuracy in metres and Unix capture time. The camera-location dialog can explicitly replace a saved position. Permission errors never create fallback coordinates, and recordings are excluded. Engine snapshots expose `live_camera` without leaking source URLs. Each incident copies the camera location at detection time; later camera changes never move historical evidence. An explicit user-requested historical correction retains the previous location, reason and timestamp in `signals.location_correction`; matching response snapshots are corrected without changing dispatch state or retrying calls.

Map View uses located physical cameras, including stopped cameras. The heatmap groups original `signals.location` coordinates from the latest 100 API incidents, requiring `signals.live_camera=true` and excluding false positives and synthetic events. Missing coordinates remain unplotted. `map-data.mjs` produces real counts; `map.mjs` draws exact-location camera groups and incident-density circles, with no simulated safety scores, sectors or response units. Browser position only controls the current laptop marker and map view; it is never used to fill in past incidents.

Analytics uses `/api/analytics/summary?days=7|30`, with server-local calendar boundaries through the query time. Indexed SQLite queries in one read transaction aggregate every stored eligible incident in the window, bypassing the library’s 100-record limit. The query requires `mode=live` and a true `signals.live_camera` flag, excludes false-positive reviews from charts and reports their count separately. Daily/type/camera totals agree; reviewed/unreviewed and gun/knife counts are measured from the same snapshot. Camera names come from stored events, not presentation sectors. Unknown event types remain counted. No human-response-time claim is derived from Twilio status.

New telemetry tuples contain a sixth `live_camera` field, stored in a nullable SQLite column. Four/five-field legacy writes remain accepted with unknown provenance; the real analytics page excludes those rows and recordings from its last-24-hour people-count table. No historical row is relabelled as physical-camera data. The frontend polls the summary every five seconds only on Analytics, discards obsolete period responses, preserves failed-refresh data with a stale label, and distinguishes an unavailable query from an empty database.

### Which dashboard surfaces use real data

| Surface | Current source and boundary |
| --- | --- |
| Connected camera grid, inspector and local Incident Library | Actual camera/recording state and saved evidence; source provenance distinguishes physical cameras from files/tests. |
| Map View and Safety Heatmap | Located physical cameras and eligible saved incident coordinates; missing positions remain unplotted. Heatmap uses the latest 100 incident list. |
| Analytics | Full selected 7/30-day eligible database history; recordings, tests and unknown physical-camera provenance are excluded. |
| Home Overview | Still uses presentation data. Do not interpret its overview cards as verified live analytics. |
| Demo network / presentation incident library | Fictional browser presentation records, kept separate from stored incidents. |
| System Health | Explicit local-observation mode and separate sample-infrastructure mode. |

## 9. Evidence, recordings and playback

The engine renders pose overlays with best-effort head blurring, encodes JPEGs, and appends processed frames to `EvidenceBuffer`. When an incident is generated, the buffer snapshot is queued. Weapon evidence also includes the original object worker's sampled/annotated frame and available newer buffered frames that arrived while inference ran. There is no guaranteed fixed post-event recording window.

`Store.jobs` is a bounded 32-item queue shared by incident and telemetry writes. A successful incident enqueue synchronously attempts response registration; the background writer separately creates the evidence file and incident row. A full queue drops the incoming record and reports a storage error. These enqueue/registration/file operations are not one atomic end-to-end transaction: a response can be registered before its clip is ready, or an error can leave one part incomplete.

With at least two decodable buffered frames, the writer resamples their timestamps to a 10 FPS MJPEG AVI, preserving approximate elapsed duration by repeating the appropriate sampled frame. It does not reconstruct visual detail from skipped frames. If encoding fails or insufficient frames exist, the incident records `clip_error` and can have no clip. Voice can still proceed from the alert snapshot; SMS clip delivery waits for evidence.

The upload endpoint accepts supported videos up to 1 GiB. OpenCV verifies that the first frame and duration are readable; files receive server-generated IDs. Originals live in `data/recordings/`. On-demand playback uses FFmpeg, or the bundled `imageio-ffmpeg` executable, to create H.264/yuv420p MP4 with fast-start metadata and no audio. Conversions cap playback width at 1280 pixels, are serialized, use two encoder threads, and have a 180-second timeout. This playback limit is separate from the recording-analysis height cap. Cache keys include source path, size and modification time.

The web player supports seeking, ±10 seconds, start/end, playback speeds from 0.25× to 4×, fullscreen and original download. Evidence also supports jumping to the event. This library handles stored files; it is not continuous multi-camera DVR recording. Uploaded originals may retain audio, while generated playback copies omit it.

### Recorded-video analysis continuity

Uploads are stored by `MediaLibrary`; the browser then starts a file session and navigates to its analysis feed. At capacity, only finished or manually stopped unsaved file sessions can be replaced; configured physical cameras and active analyses are retained. Recordings sample to the requested analysis FPS (8 by default) and at most 480 pixels high, preserving original video timestamps. A one-packet capture handoff prevents model warmup or slow inference from skipping selected frames. File analysis waits for exact source-frame depth through `DepthWorker.analyze_file_frame`; regular cadence uses source time and grab priorities cannot be lost to live wall-clock throttling. Progress and completion are visible in the feed. Every resulting event carries `live_camera=false`, preventing automatic response calls. Originals remain available for playback and seeking.

## 10. Where the response agent is implemented

The agent spans [alerts.py](vmd/alerts.py), [response.mjs](vmd/static/response.mjs), [calling.py](vmd/calling.py) and optionally [delivery.py](vmd/delivery.py). [api.py](vmd/api.py) connects those services to the camera/store lifecycle and browser actions.

| Agent capability | Actual implementation |
| --- | --- |
| Observe | `Store.enqueue()` invokes `AlertAgent.register()` for a generated incident. |
| Remember | SQLite retains the incident snapshot, deadline, response state and contact/attempt records. |
| Decide | Python rules check event eligibility, elapsed deadline, operator actions, service settings and prior attempts. |
| Act | The browser sounds an alarm; server workers submit configured calls/SMS. |
| Monitor | Calling polls Twilio status; the browser reads persistent status and displays failures/uncertainty. |
| Stop/escalate under operator control | Acknowledge/cancel/review transitions and hospital selection change remaining actions. |

### Eligibility and timing

`AUTO_EVENTS` currently contains **`fight`, `knife_detected` and `snatching_detected`**. Automatic registration also requires `mode="live"` and `signals.live_camera=true`. Gun incidents and other eligible live-camera event types can be dispatched manually; guns do not currently start the automatic ten-second countdown. Saved recording-analysis incidents also support **manual dispatch**: the real processing mode is `live`, with an explicit boolean `live_camera=false`. Uploading, analysing or finishing a recording never creates an automatic countdown. Synthetic/demo input and unknown or non-boolean source provenance cannot request a real response. Presentation examples only simulate UI actions.

For manual recording dispatch, the operator reviews the selected saved incident and slides explicitly. The backend creates a requested response with a fresh request timestamp even when the recording was analysed earlier. Existing false-positive/cancel checks, centre enablement, frozen contacts and per-recipient deduplication still apply. Later Possible-to-confirmed incident upgrades refresh an existing response snapshot without starting a countdown or repeating calls. Recording events retain their file provenance and remain excluded from physical-camera analytics and automatic geolocation.

For an eligible new event, the response has `created=event.created`, `deadline=event.created+10`, and `state="pending"`. The timer checks about every 0.2 seconds. At the original deadline it sets `state="requested"`, `trigger="automatic"` and `voice_requested_at=deadline`. Restarts/page refreshes do not reset that timestamp. Model latency, thread scheduling, connectivity and provider response mean ten seconds is a **request deadline**, not a guaranteed handset ring or clip-delivery time.

### Persistent state and operator controls

| State / action | Meaning / transition |
| --- | --- |
| `pending` | Countdown active for an automatically eligible event. |
| Deadline expires | Changes `pending` to `requested`. |
| `dispatch` | Requests authority response immediately and records a fresh voice-request time. |
| `call_hospital` | Requests response and sets hospital inclusion to true. |
| `acknowledge` | Changes a pending/requested alert to `acknowledged`; stops further unsubmitted actions. |
| `cancel` / false-positive review | Changes response to `cancelled`; prevents further dispatch and invalidates associated public evidence links. |
| Confirm review | Marks the incident confirmed and acknowledges an active pending countdown. |
| `requested` | Work has been requested; it does not mean calls succeeded or a human responded. Recipient outcomes are separate records. |

The browser needs an **Enable alarm sound** gesture before Web Audio can play. While response data is fresh and an alert remains active, it emits short 880 Hz tones roughly every 0.9 seconds. Gain rises over the initial ten seconds and then stays capped. Closed/suspended browser tabs cannot beep; the server's countdown continues. A stale API response suppresses the browser alarm rather than pretending the displayed status is current.

A single camera click selects its red-bordered card; double-click/Enter opens the feed. A newly observed unreviewed live incident can select its camera once. A recording incident is manually actionable from its finished feed or **Incident Library → Review dispatch**, which stays bound to the saved incident even if its analysis camera has been removed or replaced. Sliding to dispatch locks/grays the control after the API accepts the request. A centered hospital dialog follows; selecting Yes queues the saved hospital without redialing already attempted authority numbers. The dialog stays bound to that incident even if another camera is selected. Detailed recipient errors/status live in **Centre settings → Recent response details**. Downloads remain in the incident library.

The first manual/automatic dispatch or false-positive review persists Unix `handled_at` in the incident and, when present, its response row. Both tables retain the original time across subsequent actions and restarts. Automatic requests can precede the evidence writer: the timer copies handling time to an existing incident, and a later insert copies it from the response in the insert transaction. Response snapshots expose `incident_saved`, separately from `clip_ready`; a failed clip still has its incident/error record.

At `handled_at + 30` seconds the browser removes that saved incident from the physical camera's operational selection and stops its alarm. The inspector returns to Monitoring only for a running, fresh physical camera; stopped/stale cameras remain labelled offline. It never falls back to an older incident for that camera. Evidence, review, response outcomes and per-recipient deduplication are retained; this timeout does not declare response success or cancel a submitted call. A newer incident ID is independently actionable even during the older countdown. Model workers remain running, and their clear-interval/episode gates remain unchanged, avoiding repeated reports of one continuous event. Recording reviews remain available after handling; a finished file never claims that live monitoring resumed. Unhandled events do not expire merely because 30 seconds elapsed. Legacy handled records use their existing dispatch/cancellation times; historical false positives without a response/review timestamp use detection time to retire the old selection rather than start a fresh countdown.

## 11. Dynamic Twilio calling flow

### Message and destination selection

The calling worker wakes about every 0.5 seconds. It requires enabled centre calling and valid local Twilio configuration. It considers `requested` responses whose `voice_requested_at` is within the last five minutes. Old requests do not start dialing merely because credentials or connectivity later become available.

At preparation, authority and hospital contacts are frozen into `voice_contacts`. Authorities come from the saved centre account; the hospital is included only when requested. Calls are made to those configured phone numbers, not a built-in emergency dispatch network. Each attempt reserves a row under `UNIQUE(incident_id, phone)`, so repeated polling, double-clicks or a shared authority/hospital number do not create duplicate attempts. Failed and uncertain rows also prevent automatic retries.

`call_instructions()` builds this message from the centre and the incident snapshot:

> Hello, we are calling from {centre}. {incident} was detected at {place} at {original incident time/date/timezone}. Camera {camera name}. Latitude {latitude}, longitude {longitude}. Please verify the incident and coordinate a response.

For recording incidents the announcement explicitly says **recorded footage**, names its analysis session, gives the analysis time and a valid source-video offset, and describes any location as configured and unverified. It never presents analysis time as the original occurrence time. Missing location is stated explicitly instead of invented. The XML is constructed with ElementTree escaping. `<Say voice="Polly.Joanna" language="en-US" loop="2">` speaks the message twice, followed by `<Hangup/>`. This is a generated template, not free-form LLM output.

### HTTP data exchange

| Hop | Protocol/data | Responsibility |
| --- | --- | --- |
| Calling → Twilio Calls API | HTTPS POST, Basic authentication, form fields `To`, `From`, `Url` | Request one outbound call. |
| `Url` value | `https://twimlets.com/echo?Twiml=<URL-encoded XML>` | Carry the dynamic announcement as retrievable TwiML. |
| Twilio → Twimlets Echo | HTTP request to the instruction URL; XML response | Retrieve the instructions when processing the call. |
| Twilio → recipient | Telephone audio synthesized from `<Say>` | Speak the configured message if the account/call/instructions succeed. |
| Calling → Twilio call resource | HTTPS GET using the returned Call SID | Refresh provider status. |
| API → dashboard | Authenticated JSON from stored results | Show accepted, active, failed or uncertain outcomes. |

Turning calling off stops new submissions; it does not cancel calls already sent or stop their status polling. The REST adapter uses Python's standard library, not the sample Twilio helper SDK. Credentials go in authentication to Twilio, not into the Echo URL. The Echo URL does contain announcement/location text; anyone holding it can retrieve that text, so it must not be logged or shared casually.

`incident` mode sends only `To`, `From` and `Url`; it does not currently send custom ring timeout or duration parameters. Provider defaults apply. The optional `trial_template` mode selects Twilio's fixed hosted TTS demo and does not transmit the incident announcement. There is no automatic fallback from incident speech to a demo.

### Results and failure semantics

Twilio acceptance yields a Call SID and a status such as `queued`, `ringing`, `in-progress` or `completed`. Terminal outcomes can also be `busy`, `failed`, `no-answer` or `canceled`. Known active calls are polled at approximately five-second intervals for up to ten minutes. Missing final status becomes `uncertain`.

A rejected request can fail before any Call SID exists. A timeout, interrupted submission or server error can leave an unknown result; the adapter records uncertainty and does not automatically retry. On startup, interrupted `submitting` records become uncertain. Acknowledgement/cancellation stops work not yet submitted; the app does not recall a call already submitted to Twilio. `completed` is provider call status, not evidence of a human acknowledgement or response team deployment.

The current account previously rejected inline TwiML parameters. The Echo URL route was implemented from the earlier working caller pattern and verified with fictional GET/POST requests. That establishes correct instruction hosting/encoding, not successful custom speech over the carrier on this account. The earlier fixed demo call is the one with user-confirmed ringing/audio.

### Configuration boundaries

Voice configuration comes from private `data/twilio.json` or `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM_NUMBER` and `TWILIO_VOICE_MODE`. Environment values override the private file. `.env.example` documents names but is not automatically loaded. Credentials never go to the browser or architecture file.

## 12. Independent evidence links, Cloudflare hosting and optional SMS

### Prepare and revoke without dispatch

`EvidenceSharing` in [evidence.py](vmd/evidence.py) is independent of voice and SMS. **Incident Library → Video link** reads sharing status; an explicit Generate action posts only the saved incident ID. The authenticated API verifies real-source provenance, review/cancellation state and a safe saved clip path, calls `MediaLibrary.playback()` for an H.264 MP4, then rechecks eligibility/current clip under a SQLite write transaction before issuing a one-hour 32-byte random bearer token. Only its SHA-256 hash, incident ID, cache filename and expiry are stored. The JSON response is `{url, expires_at, incident_id}`. This operation never changes `response_alerts`, `handled_at`, contacts, calls or deliveries.

The dialog displays Copy/Open and local expiry; a 195-second request timeout accommodates the existing 180-second conversion limit. Other API requests retain their 12-second default. Closing discards the raw URL; a later GET returns only active-link count and latest expiry. DELETE removes all link records for that incident, including previously issued SMS links. False-positive/cancelled incidents cannot create links and their current links are denied at the gateway. Original recordings need no invented location to share their evidence. The authenticated `POST /api/incidents/{id}/event-package` now returns saved event details with a fresh video link in one JSON response. A later integration can transmit that package; this endpoint itself makes no outgoing submission.

### Return event details and a clip link

The [event-package API](docs/event-api.md) reuses `EvidenceSharing` and the metadata allowlist used by the read-only mobile API. A local centre-authenticated POST supplies only an incident ID. After preparing the clip, the service reads the latest saved incident within the same transaction that issues the link, then returns event/camera identifiers, UTC detection time, unknown occurrence time, source kind/recording offset, nullable saved place/location, review status and `video_url`/`video_expires_at`. Location provenance stays explicit and unverified; absent coordinates are not replaced with the laptop’s current position. No raw signals, source URLs, private paths or contacts enter the response.

```text
Local client + centre session + incident ID
  → POST /api/incidents/{id}/event-package on 8765
  → eligible saved incident + cached/prepared MP4
  → final metadata snapshot + new one-hour token record
  → JSON event details and HTTPS video URL returned to client
```

Each successful POST issues a new link; it does not dispatch, call, message or change reviews/handling state. Existing share revocation also revokes package-issued links. The main endpoint remains local; the public Cloudflare gateway only serves token-scoped video, and mobile gateway 8766 remains read-only metadata without link issuance.

### Hosting and public playback

[scripts/evidence_tunnel.py](scripts/evidence_tunnel.py) provides start/status/stop on POSIX. It starts the evidence gateway and installed `cloudflared` as detached processes, targets **127.0.0.1:8768 only**, isolates tunnel configuration, binds metrics to loopback and uses HTTP/2. Port conflicts stop startup. The helper verifies public health before atomically saving its origin and process identities in owner-only `data/evidence-host.json`; private logs stay under `data/evidence-runtime/`. Stop clears configuration first and checks process birth identity before terminating owned children. Partial services are explicitly reported for stop/start recovery; this is not an automatic OS startup service.

The main API reads managed hosting configuration dynamically and checks process liveness/available birth identity. An explicit `VMD_EVIDENCE_BASE_URL` HTTPS origin overrides this file for independently managed deployments. Configuration/liveness is not an Internet uptime guarantee. Quick Tunnel restarts change the origin, requiring newly generated links; one-hour token expiry does not keep an offline laptop or tunnel available. The helper was verified against Cloudflare using generated sample media in isolated storage.

This path is independent of voice. [delivery.py](vmd/delivery.py) reads its credentials and `VMD_EVIDENCE_BASE_URL` from the process environment; it does **not** reuse the voice adapter's private-file fallback. It also requires the centre's separate `messaging_enabled` toggle. Successful voice configuration therefore does not imply working SMS/video delivery.

For a recent requested response, the worker freezes contact destinations, requires the saved camera location and encoded evidence, and asks `MediaLibrary.playback()` for an H.264 MP4. It creates a random URL-safe bearer token, stores only its SHA-256 hash with the incident/cache filename and one-hour expiry, and submits an SMS containing the centre, event, place, coordinates, a map link and `/e/{token}` video link.

The SMS uses Twilio's Messages API fields `To`, `From` and `Body`. It sends a **link**, not an MP4 attachment. Recording messages identify a recorded-footage review and qualify the configured location and unknown capture time. The package includes `source_kind` and `source_seconds`; for recordings, `detected_at` is the analysis time. A recording with no saved location carries `location=null` and its SMS explicitly says location unknown, with no map or invented coordinates. Physical-camera SMS still requires its saved location. Accepted means the API accepted it; delivery receipts/callbacks are not implemented. One attempt per incident/phone is recorded. Uncertain outcomes are not automatically retried. Failed clip preparation can be retried by a new manual request; repeated polling does not repeatedly perform failed preparation.

The evidence gateway is a separate FastAPI application exposing `/health` (service identity only) and `/e/{token}` GET/HEAD, with no dashboard or public API documentation. It validates token shape/hash/expiry, a real eligible saved incident, false-positive/cancellation state and cache-file containment, then serves only the matching MP4 with range support and no-store/no-referrer headers. A left join permits independently shared incidents without response rows. The recipient does not need a dashboard session; possessing a valid token grants temporary access. This gateway is not object storage: the file stays on the laptop.

Playback data flow:

```text
Recipient opens HTTPS evidence link
  → Cloudflare tunnel
  → local evidence gateway on 127.0.0.1:8768
  → token validation against the local SQLite database
  → matching MP4 in data/playback/
```

Only port 8768 is exposed by the production evidence helper. The dashboard on 8765 remains local. The laptop, gateway, tunnel and network must remain running for remote playback. Actual HTTPS clip playback and revocation were verified; end-to-end clip messaging is still a separate phase. Voice through Twimlets Echo does not need Cloudflare.

## 13. Persistent storage and authentication

The shared database is `data/telemetry.sqlite3`, configured for WAL mode. Components use their own short SQLite connections with a ten-second busy timeout; there is no separate PostgreSQL server, Redis queue or cloud database.

| Table / file | Data stored | Owner / use |
| --- | --- | --- |
| `incidents` | Event ID/time/camera/type, score, JSON reasons/signals, review label, clip filename/error | Evidence library and incident review. |
| `telemetry` | Time, camera, mode, people count, rule score | Approximately one sample/second from processed frames; local hourly crowd aggregates. |
| `recordings` | Recording ID/name/filename/time/size/duration | Uploaded-video library. |
| `centre` | One centre's settings/contacts as JSON, salt and password hash | Registration, preferences and contact selection. |
| `centre_sessions` | SHA-256 session-token hash and expiry | Authentication without storing raw session tokens. |
| `response_alerts` | Frozen event, deadline/state, trigger, hospital flag, request time, frozen contacts and preparation errors | Persistent response workflow. |
| `response_calls` | Recipient snapshot, attempt status, Call SID, sanitized error, timestamps | Voice deduplication and provider monitoring. |
| `response_deliveries` | Recipient snapshot, attempt status, Message SID/error | SMS deduplication and outcome reporting. |
| `evidence_shares` | Token hash, incident, MP4 cache filename, expiry | Independently prepared or SMS-issued external evidence access. |
| `data/evidence-host.json` | HTTPS origin, owned-process IDs/birth identities and port | Private managed Quick Tunnel configuration, read dynamically. |
| `data/cameras.json` | Live-source connection/settings/enabled state | Camera recovery and restore; private. |
| `data/clips/` | Generated MJPEG AVI evidence | Original incident recordings. |
| `data/recordings/` | Uploaded originals | Local source media. |
| `data/playback/` | Cached H.264 MP4s | Browser playback and optional sharing. |
| `data/twilio.json` | Local voice credentials/sender/mode | Private configuration, Git-ignored. |
| `models/` | Downloaded verified model assets and runtime caches | Local inference, Git-ignored. |

A centre has one password, at least one and at most twenty authority contacts, and one hospital contact. These authority/hospital entries are destinations, not separate user accounts or authority/hospital portals. Phone numbers are validated in international format. Passwords are stored with a random salt and scrypt hash. Sessions use random tokens, a 24-hour expiry and an HttpOnly/SameSite=Strict cookie. The public evidence gateway is a token-scoped download route, not an authenticated recipient portal. The current loopback HTTP cookie is not a public HTTPS authentication design. There is no multi-centre tenancy, per-officer role system, password reset or recipient identity verification in this version.

FastAPI protects non-public `/api/` routes with the session. Host/origin checks restrict the local surface; mutations require `X-VMD-Client: dashboard`. Media paths must remain inside their designated directories. Presentation settings/reviews can use browser local storage, but those are distinct from authoritative server incidents and response records.

Raw frames temporarily exist in memory. Published annotated previews and generated evidence use best-effort head blurring; missed people can leave identifying detail visible. Uploaded originals are not automatically anonymized. There is no automatic evidence-retention purge or application-level encryption of all stored media/contacts.

## 14. Browser/API interaction

The browser uses ordinary HTTP polling and file playback, not a WebSocket media bus or WebRTC session. State and frames have separate identities/age metadata so stale depth/object previews can be hidden instead of drawn over newer footage. API frame bundles include base64 JPEG data; media playback returns video files. Browser map tiles are an external UI dependency and do not determine an incident's saved coordinates.

Grid/detail preview polling targets up to 8 updates per second, including request time in each interval. Only one frame request can be in flight; slow requests reduce the achieved rate rather than building a queue. Other dashboard polls keep their existing delays. Hidden tabs and pages outside Operations make no frame requests. An image element receives a new source only when its encoded JPEG changes, while unavailable/stale images are still hidden. Display cadence remains limited by the camera and processed-frame production rate; this is not a separate unblurred video stream or an increase in detector sampling rates.

| API group | Representative routes | Request/response purpose |
| --- | --- | --- |
| Authentication | `GET /api/auth/status`, `POST /api/auth/signup`, `/login`, `/logout` under `/api/auth` | Registration/session state and password login. |
| Centre settings | `GET/POST /api/centre` | Contact settings and configured/enabled response services. |
| Cameras | `GET/POST /api/cameras`; `PATCH /api/cameras/{id}/settings`; `POST /api/cameras/{id}/stop`, `/restart`, `/remove`, `/eco`, `/location` | Lifecycle/configuration actions and camera state JSON. |
| Frames | `GET /api/camera-frames`, `/api/cameras/{id}/frames` | Grid/detail JPEG bundles and original sample metadata. |
| Incidents/review | `GET /api/incidents`; `POST /api/incidents/{id}/review` | Saved event list and explicit review labels. |
| Response | `GET /api/response-alerts`; `GET/POST /api/incidents/{id}/response` | Countdown/action state and operator commands. |
| Evidence | `GET /api/incidents/{id}/clip`, `/play`, `/response-package` | Original evidence, playable MP4 or incident JSON plus clip ZIP. |
| Evidence sharing | `GET /api/evidence/status`; `GET/POST/DELETE /api/incidents/{id}/share` | Check hosting/counts, explicitly prepare a one-hour link, or revoke all incident links; no dispatch side effects. |
| Event package | `POST /api/incidents/{id}/event-package` | Return allowlisted event/place/location/time metadata and a fresh one-hour video link to the authenticated local caller; no outgoing submission. |
| Recordings | `GET/POST /api/recordings`; `GET /api/recordings/{id}/play` or `/download` | Upload/catalogue/playback of stored videos. |
| Analytics | `GET /api/analytics/summary?days=7` or `30`; legacy `GET /api/analytics` | Full-period, source-qualified summary for the real Analytics page. The legacy endpoint remains compatible but filters only `mode=live` and is not the page’s physical-camera data source. |
| External evidence gateway | `GET /e/{token}` on the separate service | Token-scoped MP4 bytes only. |

## 15. Training architecture and the path back into VDM

The [Colab notebook](notebooks/train_vdma_weapons_colab.ipynb) is an offline development workflow. It is not executed by the response agent, and marking an incident false positive does not start training.

1. **Acquire:** download a bounded Open Images subset with gun/knife boxes and candidate confuser images. Optionally add separately labelled datasets and actual camera sessions.
2. **Curate:** review negative candidates explicitly; use empty label files only for confirmed weapon-free images. Audit positives and require complete, valid target boxes. Preserve attribution metadata.
3. **Split:** retain public splits, separate local recordings by session, and filter exact/near duplicates across splits. Do not distribute neighboring frames from one recording across train/validation/test.
4. **Train:** initialize a fresh pretrained YOLOv8s by default; optionally use YOLOv8n or random initialization. Fine-tune on two classes and reviewed negatives. Save checkpoints/configuration to Drive with dataset fingerprint checks for resume.
5. **Calibrate:** use validation only to assess gun false alarms and recall, including separate handgun recall. Return no threshold recommendation when the experimental targets are not met.
6. **Evaluate:** freeze the checkpoint/cutoff, measure held-out results, check knife regressions, and optionally compare the exact current specialist. Frame false-positive rates do not establish real false alerts per hour.
7. **Export:** produce `best.pt`, SHA-256/class-map manifest, validation/test reports and provenance in `vdma-weapon-model.zip`.
8. **Integrate later:** evaluate actual camera footage and CPU latency, then deliberately update the pinned model and both weapon threshold gates while preserving safe loading and class-map validation.

Full Colab training and accuracy evaluation have not run in this workspace. Notebook schema/syntax, data/metric regression checks, a small real public download and a synthetic one-epoch CPU training/checkpoint smoke test have passed. Those checks establish workflow plumbing, not a solution to the phone/laptop gun false positives. The checked public validation pool has only 17 usable handgun images before duplicate filtering, so a stronger independent handgun evaluation set remains necessary.

## 16. Network requirements and known boundaries

| Operation | Connection required |
| --- | --- |
| Dashboard, login, local database, local media playback | No internet after installation; local service must be running. |
| Webcam or stored-file analysis | No internet after model/dependency installation. |
| Phone/IP-camera stream | A reachable LAN/route to the source. Being on the same Wi-Fi name alone does not guarantee reachability if clients are isolated. |
| Local pose, depth, weapon inference and eco scheduling | No internet once assets are installed. |
| Twilio call submission, Echo instructions and call-status polling | Internet access and permitted account/destination settings. |
| SMS with remotely playable clips | Internet, configured messaging, public evidence URL and running gateway/tunnel. |
| Remote map tiles | Internet for the map display; independent of local inference. |
| Initial model/data downloads and Colab training | Internet; Colab compute and optional Drive storage. |

Important limits for implementation planning:

- AI observations are fallible. Gun false positives, missed knives, short fights and occlusion remain evaluation concerns.
- The ten-second automation currently covers confirmed fight, knife and confirmed snatching. Adding automatic gun escalation would be a deliberate behavior change.
- The server must stay alive for automatic response/recovery; no always-on remote dispatcher or installed OS watchdog is part of this architecture.
- A browser alarm is local audio, and Twilio `completed` is not an acknowledgement. No live call transcription, conversation or emergency-service integration is implemented.
- Voice works independently of clip encoding. Clip delivery depends on evidence, location, conversion, messaging configuration and public reachability.
- The ten-second evidence buffer is processed, sampled footage with no guaranteed post-event window. It cannot yet support a continuous semantic history of everyone in the scene.
- A local System 1–style incident-hypothesis classifier is planned below. There is currently no such runtime, integrated video-description model, temporal knowledge graph or automatic learning loop in the main app. The separate Vision Lab already tests single-frame descriptions.

## 17. Planned local System 1 incident-hypothesis layer

### Purpose, model choice and current status

We will add a small local classifier that maps a recent, grounded evidence summary to a controlled set of **incident hypotheses**. Here, “System 1” means a quick classification pass with a fixed output schema, rather than an open-ended reasoning or conversation loop. It is a design role, not a claim that a specific proprietary System 1 model has already been integrated.

The intended starting candidate is **Fastino GLiNER2.5-Decide**, which was previously proposed for this project. The publisher describes it as a 340M-parameter English classification model with runtime-specified labels and local execution; it is a text model, not an image/video encoder. Its outputs can classify evidence summaries, but do not themselves verify a fight, weapon or physical contact. See the [publisher model card](https://huggingface.co/fastino/GLiNER2.5-Decide) and [release description](https://fastino.ai/blog/gliner-2-5-decide-open-weight-decision-model).

**Documentation/design only:** there is no classifier import, downloaded checkpoint, worker, database migration or new API in main VDMA yet. The exact package/checkpoint revision, label schema, thresholds and resource budget must be pinned and evaluated during implementation. No laptop latency, RAM figure or incident accuracy is claimed from the publisher's hardware benchmarks.

### Where it fits

```text
EXISTING local vision and temporal evidence
  ├─ track pair, reliable pose and motion-supported arm/leg observations
  ├─ contact surfaces, original-frame depth status, ages and uncertainty
  ├─ supported interaction duration, strike/pull/departure sequence
  └─ fresh weapon label/confidence or missing-worker state
          ↓
PLANNED bounded observation window, grouped by camera and episode
          ↓
PLANNED evidence-summary adapter
  → deterministic short English text + fixed candidate-label schema
          ↓
PLANNED single shared local classifier worker
  → label scores / selected incident hypotheses / abstention
          ↓
PLANNED application validation and attachment to the same episode
  → Inspector explanation and review data; shadow evaluation first
          ↓
EXISTING detection gates, operator controls and response state machine
  → remain authoritative for confirmed incidents and calls
```

The classifier should consume measured facts, including missing or contradictory evidence, rather than simply being told “fight detected” and asked to repeat that label. The first integration will run in **shadow/advisory mode** alongside current decisions. Any later role in alert policy must be separately evaluated; a hypothesis alone will not bypass the strict weapon cutoff, mandatory fight depth, configured timer, live-camera provenance, cancellation or response deduplication.

### Input: a compact evidence window, not raw footage

The adapter will gather a short bounded window for a **camera and episode**, with zero, one or multiple track IDs, and serialize the observed changes. Pair IDs apply to fight/snatching interactions; a fall may involve one track and a weapon observation may involve none. It will include source start/end times and observation IDs so a late result can be matched to its episode. It will explicitly report stale depth, occluded joints, uncertain contact and tracking breaks. Operator notes, if supported later, will be data to classify; they will not redefine allowed labels or action policy.

Example **proposed adapter input**, with fictional values (not an existing endpoint):

```json
{
  "camera_id": "cam-example",
  "episode_id": "episode-example",
  "window_revision": 4,
  "source_window_seconds": [20.0, 23.4],
  "track_ids": [1, 2],
  "observations": [
    {"id": "o1", "kind": "directed_strike", "actor": 1, "target": 2, "surface": "raised_guard", "depth_status": "compatible"},
    {"id": "o2", "kind": "directed_strike", "actor": 2, "target": 1, "surface": "torso", "depth_status": "compatible"}
  ],
  "facts": {
    "both_tracks_reliable": true,
    "supported_interaction_seconds": 3.2,
    "depth_fresh": true,
    "camera_moving": false,
    "weapon_observation": "none_in_latest_sample"
  }
}
```

The model receives an adapter-produced summary such as: “Two reliable tracks exchanged image-supported strikes over 3.2 seconds. One strike reached a raised guard. Matching contact-depth observations are fresh. No weapon was reported in the latest object sample.” The original numerical facts remain available to the policy layer. “No weapon reported” does not mean proof that no weapon exists.

A separate vision-language model is **not required** for this first design: current pose, motion, depth and object outputs can produce the summary deterministically. If a vision description is added later, its capture time and uncertain/model-generated origin must remain distinct from measured observations.

### Output: typed hypotheses with provenance

The proposed label schema will cover incident hypotheses such as `fight`, `snatching`, `weapon_visible`, `fall`, `routine_activity` and `uncertain`. Multi-label output can represent simultaneous observations. Labels describe hypotheses, not confirmed intent or proof of a crime. Low confidence, conflicting scores, missing evidence or a stale window should produce abstention rather than a forced incident label. Cutoffs must be calibrated on reviewed, held-out examples.

Example **proposed application output envelope**, not a claim about the exact library return format:

```json
{
  "camera_id": "cam-example",
  "episode_id": "episode-example",
  "window_revision": 4,
  "model": "fastino/GLiNER2.5-Decide",
  "model_revision": "PIN_DURING_IMPLEMENTATION",
  "schema_version": "incident-hypotheses-v1",
  "hypotheses": [{"label": "fight", "score": 0.86}],
  "abstained": false,
  "evidence_refs": ["o1", "o2"],
  "mode": "advisory"
}
```

The application copies camera/episode IDs, evidence references and timing from its own input record; the classifier does not invent them. The model supplies classification outputs. A score is not a calibrated probability that a real crime occurred. Explanation text can be built from templates and the referenced facts; a generative explanation model is not needed.

### Local scheduling, storage and failure behavior

- Load one pinned classifier instance for the application, shared through a separate bounded worker rather than one copy per camera. CPU-first feasibility should be measured on the actual laptop alongside pose, depth and weapon workers.
- Submit on meaningful candidate/evidence changes, with per-camera rate limits and deduplication. Do not classify every camera frame. Keep inputs short, bound the window, and replace superseded queued summaries instead of building a backlog.
- Join results by camera, episode, source window and revision. A result for a stopped camera, replaced episode or older revision cannot overwrite the current hypothesis. Keep model version, input evidence references, inference latency and abstention reason for review.
- Initially retain hypotheses as advisory episode metadata, keeping one existing incident ID across upgrades. Any durable field/schema changes are implementation work still to do. Do not create duplicate incidents or provider attempts merely because classification is rerun.
- If the classifier is unavailable, slow, times out or returns invalid labels, publish its unavailable/abstained status and continue the current detection pipeline. The existing response timer must not block waiting for classification, and a classifier failure must not trigger a call.
- Keep inference local after initial installation. Checkpoint disk size and parameter count are not runtime RAM measurements; measure worker RSS, peak memory and actual latency before choosing a memory/rate budget. There is no claimed “very low RAM” guarantee yet.

### What implementation will require

1. Pin and install the model/library in the main project's environment only after compatibility/resource evaluation; cache assets for offline use.
2. Add the bounded observation/summary adapter and versioned label schema without replacing existing sensor gates.
3. Add one isolated classifier worker with timeout, shutdown, stale-result rejection and health telemetry.
4. Add advisory hypothesis fields and a compact inspector display, tied to the existing episode and evidence.
5. Evaluate on separate recordings: actual reciprocal fights, non-contact gestures at different depths, hugs/handshakes, standing snatching, ordinary walking, weapons/confusers and missing-input cases. Split by recording/session so adjacent frames cannot leak into evaluation.
6. Measure label confusion, abstention, false alarms, missed incidents, latency and memory. Compare against the existing heuristic-only system before assigning any operational policy role.

Learning from reviewed clips remains a later **offline curation and training** task. A review label does not immediately change model weights. A temporal knowledge graph is also separate: structured hypothesis records can become future graph inputs, but this design does not claim that a graph database or cross-camera identity graph already exists.

## 18. Separate Vision Lab and its boundary

The existing [Vision Lab](vision-lab/README.md) lives in `vision-lab/`, uses its own environment/assets and serves a local UI on **8770**. It runs **LiquidAI LFM2.5-VL-450M in 4-bit MLX** on Apple Silicon to describe one fresh image at a time. Input is capped at 512×512; output is capped at 100 tokens. By default it waits five seconds after inference before taking the next sample, with a 2–60-second setting and a manual Analyse action.

```text
IP camera → isolated capture → latest JPEG
                               ├─ local preview
                               └─ single image-description worker
                                    → short text + original capture time
                                    → up to 30 in-memory observations
```

It has no integration with the main database, incident confirmation, response calls, evidence storage or planned classifier. It is a single-frame description experiment, not a trained temporal fight detector. Process isolation does not isolate hardware: it still shares laptop RAM/compute with VDMA. A future integration could feed explicitly labelled descriptions into the hypothesis adapter, after separate accuracy/resource checks. This architecture update does not connect the two services.

## 19. Where to look when changing a flow

- Camera connection/recovery: [capture.py](vmd/capture.py), [cameras.py](vmd/cameras.py), [api.py](vmd/api.py).
- Detection and scheduling: [engine.py](vmd/engine.py), [vision.py](vmd/vision.py), [eco.py](vmd/eco.py), [objects.py](vmd/objects.py), [confirmation.py](vmd/confirmation.py).
- Incident/evidence lifecycle: [storage.py](vmd/storage.py), [media.py](vmd/media.py), [alerts.py](vmd/alerts.py).
- Response behavior: [alerts.py](vmd/alerts.py), [response.mjs](vmd/static/response.mjs), [calling.py](vmd/calling.py), [delivery.py](vmd/delivery.py), [share.py](vmd/share.py).
- Account/contact behavior: [centre.py](vmd/centre.py), [centre.mjs](vmd/static/centre.mjs), [login.mjs](vmd/static/login.mjs).
- Setup and evidence: [response setup](docs/response-setup.md), [runtime stability investigation](docs/runtime-stability.md), [weapon training guide](docs/weapon-training.md), [detection update](docs/detection-update.md).

Update this document when a model, event eligibility rule, dispatch action, external delivery path or stored data contract changes. Keep operational secrets and real camera/contact data in their private configuration, not this document.
