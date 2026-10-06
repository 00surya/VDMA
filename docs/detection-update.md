# Detection update

**Current main-app cutoff (2026-09-26):** gun/knife confidence must be **strictly above 90%** for live labels, preview boxes and new incidents. Both inference and public-output gates were updated; historical 0.50 model-comparison results below are unchanged. This can suppress real weapons too and does not guarantee accuracy.

Reference reviewed: [00surya/vdm-shield at 77e9d7f](https://github.com/00surya/vdm-shield/tree/77e9d7f85d34e756353e96bf19455ffc0dbd41fc). The depth adapter/exporter, frame-matched spatial evidence, confirmation algorithm, and their regression fixtures are adapted from that revision. Its YOLO11n pose, optical flow, and core fight rules largely match this application's baseline. The integration retains FastAPI, the existing dashboard, head blurring, local evidence, and independent camera controls.

The reference also contains weapon detection, licensing/cloud features, and a ResNet18/GRU clip learner. The initial update imported general-object detection; subsequent updates below replace the specialist and limit displayed predictions to knives and guns. Saved weapon alarms, licensing/cloud features, and the clip learner are not enabled. The Git repository contains model adapters, downloaders, training code, and tests, but no `.pt`, `.pth`, `.onnx`, or trained `.npz` assets. It downloads pretrained pose/depth/object models separately. Its clip learner needs a trained GRU head for the installation's reviewed video examples; an ImageNet-pretrained encoder alone is not a trained fight classifier. Its weapon training candidate is not its live detector. No measured fight-accuracy improvement is established by this integration.

## Models and evidence

- **ZipDepth:** the [official upstream model](https://github.com/fabiotosi92/ZipDepth) at revision `91f3fd21e131641f51e8d35736d1958350180e3a`, exported locally to ONNX. The installer pins and verifies source, license, and checkpoint SHA-256 hashes. It retains the learned global-context attention and checks export parity on four shapes. The upstream MIT notice is saved beside the installed assets. Generated weights are excluded from Git.
- **Depth-confirmed mode:** the current interaction upgrade uses a three-second minimum plus reliable matching depth for the same pair. Brief strikes remain review-only; repeated strikes or sustained grappling can confirm. Missing/stale/uncertain depth never confirms a fight. See [interaction rules](interaction-heuristics.md) for current sequences, limitations and validation.
- **Frame identity:** each camera session owns its worker and bounded pose cache. Only successfully submitted frames are cached, with raw pose coordinates, source time, and image shape. Results must match all three before contributing depth evidence. No old depth is attached to a new pose. A restarted session gets an empty cache and a new worker.
- **Review-only mode:** legacy API value `responsive` now saves possible-fight warnings without confirmed fight events or automatic fight dispatch. Depth is mandatory for stronger fight and snatching labels. Hands-up, person-down and fall rules retain their original behavior.
- **Objects:** official pretrained YOLO26s at the pinned v8.4.0 asset URL, with SHA-256 verification and an exact 80-class mapping check. A separate CPU process keeps object inference off the pose path. The per-camera rate defaults to 1 FPS; errors are exposed independently. Boxes and labels are rendered on their original sampled frame with best-effort head blurring, and stale previews/results are hidden. General labels, including knife/scissors, never produce threat alarms or affect the fight score.

## Eco mode

The scheduler uses small grayscale frame differences plus a slowly adapting background. After 10 seconds without activity, it keeps pose checks at 2 FPS, depth at 0.1 FPS, and general-object checks at 0.5 FPS (never exceeding each configured rate). The reference uses 1 FPS quiet pose scans; this integration uses 2 FPS to stay inside the existing behavior filters' 0.65-second continuity limit. Pending interactions, active alerts, hands-up checks, and prone-posture checks keep full sampling. Timestamp discontinuities, frame-size changes, and motion wake full analysis. Each camera has its own scheduler, counters, workers, and live toggle; reconnect starts fresh.

Capture and video decoding continue, so this reduces inference work rather than shutting the camera off. Quiet previews and evidence can have a lower frame rate. Depth preview freshness can extend to 20 seconds in quiet mode, but fight confirmation still rejects samples older than 2.5 seconds. Counters expose processed pose frames, skipped eco checks, model time, and worker submissions. They do not estimate energy, thermal behavior, or system-wide CPU savings.

## Validation commands

```sh
.venv/bin/python -m pytest -q
node --test tests/presentation.test.mjs tests/map.test.mjs
.venv/bin/python scripts/benchmark_depth.py --compare-midas --runs 10
.venv/bin/python scripts/benchmark_eco.py
```

Tests include delayed depth arriving after newer poses, timestamp/shape mismatches, separated and flat depth, reacquired tracks, camera motion, interrupted evidence, repeated polling, independent pairs, API validation, and the preserved responsive pipeline. These deterministic fixtures verify behavior and integration; they do not measure real-world accuracy. The depth benchmark uses the same bundled public image for both models and reports latency separately from accuracy.

## Measured validation on this laptop

The complete suite passes **98 Python tests and 12 frontend tests**. Real-model inference found a bus and people in the bundled public image, and both depth and objects ran beside pose in an isolated recording pipeline. The browser check exercised new-camera settings, depth and object previews, and live eco switching without a camera restart.

With CPU pose limited to one thread and both background models enabled, the same eight-second quiet interval produced:

| Measurement | Continuous | Eco |
| --- | ---: | ---: |
| Pose frames processed | 91 | 16 |
| Time inside pose inference | 3,585.8 ms | 881.4 ms |
| Depth submissions | 8 | 1 |
| Object submissions | 8 | 4 |

Eco resumed full analysis when the recording started moving. This is about 82% fewer pose inferences in that quiet interval, not an 82% battery or total-system compute claim. Real-stage precision/recall, four-camera throughput, and detection quality under lighting changes still need representative labelled recordings.

## Knife detection follow-up

The user's phone session had object detection enabled, but reported no knife label. The original integration used only general COCO detection at confidence 0.40. A lone person never blocked that detector, but the reference's separate pretrained knife detector was missing. The missed phone frame was not retained, so that specific miss has not been reproduced.

The object worker now also runs the knife class of [Subh775/Threat-Detection-YOLOv8n](https://huggingface.co/Subh775/Threat-Detection-YOLOv8n), pinned to revision `c6d6fa4e6c9bfd4c4fccb46478db23609e5468fb` and SHA-256 `86c43444ae8319d2276dd300edc3e7f7a1137fe7566737f994ca579ad770f6ce`. This is the reference's deployed pretrained checkpoint, distinct from its untrained local training candidate. Only the knife class is enabled, at confidence 0.25; general objects retain 0.40. Scores are model outputs, not calibrated probabilities. The checkpoint is read with `weights_only=True` and an explicit allowlist of installed Torch/Ultralytics network classes. Class order and file integrity are verified before inference. No remote model code is imported.

Both models inspect the same sampled frame, independently of person count, depth, or fight confirmation. Overlapping knife boxes are merged. Missing/corrupt specialist weights or inference failure are displayed separately, while general objects continue. Eco still schedules the combined object worker at 0.5 FPS in quiet scenes. Object labels now appear directly on camera cards and in the inspector; Scene objects opens expanded. The preview bundle includes its own labels atomically, so it does not borrow labels from a newer status poll. A knife prediction does not create a fight or weapon incident.

Run `python scripts/download_object_model.py` to install both pinned checkpoints. PyTorch 2.6+ is required for the restricted checkpoint loader.

Validation: **103 Python tests and 13 frontend tests pass**, covering low-confidence knife handling, one-person/zero-person independence, duplicate merging, original frame identity, stale-label hiding, and specialist failure without disabling general objects.

A bounded comparison used the reference's pinned public [fcakyon/gun-object-detection](https://huggingface.co/datasets/fcakyon/gun-object-detection) validation archive, revision `0c8e46cbfe8edf71e592f495face94ba22155b46`, archive SHA-256 `d676264a3e040a71eb58ea71a4cd16391537c9eef84531df3d198311a1d86723`. Python random seed 17 selected 40 knife-labelled images and 20 images without knife annotations. At bounding-box IoU ≥ 0.5, the general detector matched **5 of 43** annotated knives and the combined detectors matched **34 of 43**. The combined models produced **6 knife predictions on the 20 images without knife annotations**, versus zero from general detection. Specialist inference added a median **49.9 ms** per sampled image on this laptop's CPU, with one Torch thread.

These are limited local comparison results, not deployment accuracy: labels may be incomplete, near-duplicates may exist, and overlap with the specialist's training data is unknown. Extra predictions and misses remain. An initial check of the author's annotated mosaic also missed small/occluded handheld knives; annotated mosaics were excluded from the comparison above. The user's actual phone footage still needs validation. No general claim about improved fight accuracy or total compute savings follows from these numbers.

## Assalim model replacement

At the user's request, the active knife specialist now uses [JoaoAssalim/Weapons-and-Knives-Detector-with-YOLOv8](https://github.com/JoaoAssalim/Weapons-and-Knives-Detector-with-YOLOv8), revision `3d641e6001abdaa1afa3cd45d0fe02a9554f3d0e`, specifically **`runs/detect/Normal_Compressed/weights/best.pt`**. It replaces the Subh775 model described in the historical section above. The installed filename is `assalim-normal-compressed-best.pt`, SHA-256 `21d61ad8068caca3062d33fd8d05da445ef9ae81a2bb817249e085b0f1307cf0`. The downloader uses an immutable Git blob and verifies SHA-256; the source's actual GPL-3.0 LICENSE is also pinned and saved as `models/Assalim-GPL-3.0.txt` (the source README's MIT statement conflicts with that file). Model binaries remain excluded from Git.

All six checkpoints' embedded training metrics were checked against their validation CSVs. They are YOLOv8 nano checkpoints, not differently sized models. Their best recorded **overall** mAP50–95 scores are:

| Variant | Overall validation mAP50–95 |
| --- | ---: |
| Normal | 0.70712 |
| Normal_Compressed | 0.67493 |
| Db | 0.58811 |
| Haar | 0.58811 |
| Haar_Compressed | 0.57990 |
| Symlet | 0.56521 |

Normal leads on the author's overall metric, which combines firearms and knives. The two leading RGB checkpoints were then compared with the previous specialist on the same pinned public sample described above: seed 17, 40 knife-labelled images containing 43 annotated knives, plus 20 images without knife annotations. Matching here is one-to-one at IoU ≥ 0.5. Counts below are specialist outputs, without merging COCO detections; unmatched boxes can include incomplete annotations as well as false predictions.

| Specialist | Confidence cutoff | Annotated knives matched / 43 | Total knife predictions | Predictions on 20 images without knife annotations |
| --- | ---: | ---: | ---: | ---: |
| Previous Subh775 | 0.25 | 34 | 56 | 8 |
| Previous Subh775 | 0.50 | 29 | 38 | 2 |
| Assalim Normal | 0.25 | 25 | 45 | 2 |
| Assalim Normal | 0.50 | 20 | 24 | 1 |
| Assalim Normal_Compressed | 0.25 | 29 | 50 | 0 |
| **Assalim Normal_Compressed (selected)** | **0.50** | **29** | **38** | **0** |

Normal_Compressed at 0.50 gives the best observed balance of matches and extra predictions in this comparison. It is also the checkpoint and display threshold used by the repository's own `detecting-images.py`. Inputs are ordinary camera frames with standard YOLO letterboxing; no wavelet transform or artificial JPEG degradation is applied. Wavelet models were ranked by their published logs, not incorrectly tested on raw RGB as if their preprocessing were interchangeable. This is a bounded model-selection comparison, not an independent accuracy benchmark: dataset overlap with training is unknown, the sample is small, and it is not the user's camera footage. Zero predictions on 20 negative-labelled images does not establish a zero false-positive rate. The retained phone sample still yielded no knife prediction with either RGB candidate.

The exact class order is verified as `guns`, `knife`; only knife (index 1) is enabled for this request. The restricted checkpoint loader now also allows the installed Ultralytics loss/config classes present in this older checkpoint. It never accepts arbitrary pickle globals or remote model code. The general YOLO26s model continues to handle other objects; its knife guesses are removed even if the specialist fails, so they cannot bypass the new threshold. Specialist failure remains visible and other scene objects continue. Pose, depth, fight rules, eco scheduling, and local incident behavior are unchanged.

Validation: **104 Python tests and 13 frontend tests pass**. A real spawned object worker returned the selected checkpoint's knife result at the correct frame identity and confidence cutoff. Median standalone CPU inference in the comparison was about **39.5 ms** for the selected specialist (one Torch thread); full-pipeline throughput is not inferred from this timing. Re-run `python scripts/download_object_model.py` on other machines to install the new checkpoint and license. Reconnecting a camera starts a fresh object worker with the new model.


## Knife/gun display and Action correction

Both Assalim classes are now enabled (`guns` index 0 and `knife` index 1), at the existing 0.50 threshold. The public label for `guns` is `gun`. The general COCO pass is restricted to people for head blurring on the same sampled frame; none of its object labels or boxes are published. Only specialist knife/gun predictions are returned in `scene_objects` and annotated in the preview. A specialist failure leaves no weapon labels and reports its error; COCO guesses cannot substitute for it. The legacy `knife_status`/`knife_error` API keys now cover both specialist classes.

Fresh weapon observations update camera cards, the expanded viewer, and the inspector Action. A new observation also takes precedence over the inspector's historical incident selection. One person or no people is sufficient. A latest sample without weapons, a stale sample, worker failure, or stopping the camera clears the weapon action. The existing fight/pose/depth pipeline, eco schedule, saved incident rules, and dispatch behavior are unchanged.

Validation: **104 Python tests and 14 frontend tests pass**. A real-model browser check displayed **Gun detected** and **Knife detected** with the correct Action fields; the empty-frame control remained **Monitoring**, and stopping the knife camera cleared its detection. Tests also cover both classes together, threshold enforcement, unrelated-label suppression, stale/error/stopped results, original-frame privacy blur, and specialist failure without falling back to COCO knife guesses. Public fixture checks verify wiring, not real-world accuracy.


## Local weapon incidents, camera recovery and recorded playback

Knife/gun predictions now use the normal SQLite incident and evidence queue. Each fresh specialist class produces a reviewable incident at the existing 0.50 detection cutoff, independently of person count or the fight threshold. Repeated polling and continuous presence do not duplicate events; a clear interval and ten-second cooldown re-arm that class. Failed, stale, future and repeated samples cannot create a new incident. Preview annotations retain their original frame timestamp. Buffered frames captured after the source frame, while the worker was processing, are retained in the clip so a first-frame detection has usable footage.

The camera manager atomically saves validated live-camera settings with mode 0600, restores enabled/stopped state on startup and retries disconnects/stalls with bounded exponential backoff. Stop disables retry; Reconnect enables it with the saved settings. Removal preserves evidence and removes the saved camera configuration. Video files and demos are never automatically recovered or persisted as live cameras. A corrupt configuration is reported and preserved instead of overwritten. Settings snapshots exclude connection URLs and credentials.

The existing incident library now includes uploads and a shared recorded-video player. Sources are kept under opaque local IDs, uploads are streamed with a 1 GB limit and decoded before registration, and original downloads remain available. Browser playback uses a cached H.264 conversion with range support for seeking. The player includes a timeline, ±10-second jumps, start/end, event seeking, speed selection, fullscreen, and recorded-time labels. Upload analysis always opens the settings form. No local learner or review-based training was added.

Validation for this update: **113 Python tests and 15 frontend tests pass**. Browser checks covered first-frame weapon evidence playback, upload, seeking, speed selection and the new-recording setup form. A real local MJPEG stream was disconnected and restored; the camera recovered with the same ID and every non-default setting. Restarting the test server also restored its enabled camera. Production was restarted on port 8765 with the existing stopped phone camera retained from its last verified session configuration.
