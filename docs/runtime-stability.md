# Runtime crash investigation — September 26, 2026

The repeated loss of the localhost dashboard came from native Apple GPU crashes inside the Python server. Internet loss was not the cause of the two captured process exits.

## Evidence and reproduction

macOS recorded two server crashes: PID 51176 at 14:52:50 (SIGABRT, Metal command-buffer assertion), and PID 51690 at 15:02:14 (SIGSEGV in the Metal GPU stream). Both stacks included PyTorch's `arange_range_fill_mps`; another thread was also doing MPS tensor work. The second server was detached and parented by launchd, so detaching the process had not prevented the failure.

The installation uses PyTorch 2.14.0 and Ultralytics 8.4.148. Two enabled camera pipelines selected MPS. Each camera had an independent pose model and inference thread, but those models used the same process-wide Metal stream. The existing model-load lock protected construction only; per-model prediction locks did not serialize different cameras.

An isolated two-camera workload using the real pose weights and alternating synthetic frame sizes reproduced a segmentation fault in **1.45 seconds**. It used no camera network, Twilio request or private footage. This establishes the concurrent MPS path as a trigger in the installed stack; no claim is made that every possible native crash has the same cause.

## Fixes

- A shared MPS lock now covers pose prediction, conversion of all results to CPU, and GPU synchronization, including the exception path. Tracking and other CPU work remain outside it. CPU/CUDA inference does not acquire this lock. Models, confidence thresholds and saved camera settings are unchanged. Cameras share the GPU execution slot, which can affect aggregate throughput under load.
- Depth and object workers now watch their parent's process sentinel and exit when it dies, even during blocked inference. Previously, daemon workers survived native server crashes; eight orphaned workers were found and cleaned up, along with their resource trackers.
- The normal entry point enables Python's fault handler so future native failures write thread stacks to the private runtime log. Python exception handling alone cannot catch a native abort/segmentation fault.

## Verification

- Identical two-camera workload after the fix: **200 frames completed**, exit 0, 6.55 seconds inside the workload.
- Four simultaneous camera threads: **2,000 frames completed**, exit 0, 36.54 seconds inside the workload.
- **132 Python tests passed**, including concurrent-MPS serialization through result transfer, cleanup after an inference exception, and killing an isolated parent while its worker is blocked. Two existing httpx/starlette deprecation warnings remain.
- Existing camera recovery tests preserve all model, rate, threshold and enabled/stopped settings.
- An isolated **78.19-second** integration run used two loopback MJPEG cameras with real MPS pose, ZipDepth and object/weapon workers: **154/154 HTTP checks passed**, at least 1,041 sampled pose frames processed, and camera A recovered about 6.6 seconds after its stream returned while camera B kept running. Registry bytes and settings were unchanged; test server and all workers stopped cleanly.
- The restored main app passed **60/60 HTTP checks over 30.5 seconds** (maximum 18.6 ms), retained its camera-registry hash and paused calling state, and created no new call submissions. Its saved camera endpoints separately timed out during TCP reachability checks; that is a camera-network problem while the dashboard remains reachable.

These bounded runs verify the reproduced crash and recovery paths, not long-duration uptime or detection accuracy.

Reproduction scripts and before/after outputs are retained under ignored `tmp/reproduce_mps_cameras.py`, `tmp/mps-before*`, `tmp/mps-after*` and `tmp/mps-four-camera-after*`. They are optional diagnostic artifacts, not runtime dependencies. See `HANDOFF.md` for the latest integration checks and live process checkpoint.

## Connectivity requirements

| Operation | Requirement |
| --- | --- |
| Local dashboard, sign-in, detection, eco mode, evidence saving and local playback | No internet once dependencies and model assets are installed |
| Phone/IP camera | Reachable camera network; same Wi-Fi can still have client isolation |
| USB camera or local recording | No camera network required |
| Twilio calls/SMS | Internet and configured provider capability |
| Someone elsewhere opening a shared clip | Internet, evidence gateway/tunnel, and this laptop remaining available |
| OpenStreetMap tiles and initial model/dependency installation | Internet |

The app loads pose, depth and weapon models locally with automatic downloads/installs disabled. Camera capture has bounded network timeouts; disconnect errors trigger per-camera recovery. Twilio networking runs in separate background threads with bounded timeouts. Local video saving and playback do not depend on the public tunnel. No tunnel or outbound test call was created during this investigation.
