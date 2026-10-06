# Vision Lab

A separate local experiment for describing an IP camera with **LiquidAI LFM2.5-VL-450M, 4-bit MLX**. It lives entirely in this folder and does not import, edit or control VDMA. VDMA remains on port 8765; this lab uses **http://127.0.0.1:8770**.

## Use

The environment and model are installed on this Mac. Open the lab, paste your camera address, then press **Connect camera**. An Android IP Webcam base address such as `http://192.0.2.10:8080` automatically uses `/video`. Other cameras need their actual HTTP(S) or RTSP(S) video stream, not their administration webpage. You can include camera credentials in the URL if needed; the app does not save the URL to disk or return it in API responses.

The camera preview runs independently of inference. By default, the model describes a fresh frame, then waits five seconds before sampling again. Change the interval to 2–60 seconds or press **Analyze now**. The first model load takes longer. **Stop camera** stops capture and unloads the model; reconnecting begins a new observation history. Closing the browser does not stop the lab—use Stop before leaving if you want to release camera/GPU resources.

Start/stop the lab server from this folder:

```sh
.venv/bin/python manage.py start
.venv/bin/python manage.py status
.venv/bin/python manage.py stop
```

Or run `./run.sh` in a terminal and keep that terminal open. `manage.py` verifies its own PID, start time, working directory and command before stopping anything. It never stops the VDMA server.

For a fresh Apple Silicon Mac with Python 3.11, run `./setup.sh`. All packages, download caches and model files stay under this folder. Initial installation needs internet; inference uses the local checkpoint with Hugging Face offline mode. The camera itself must be reachable. College/guest Wi-Fi client isolation can block access even when both devices share the same Wi-Fi name.

## What the test does

```text
IP camera → isolated OpenCV capture process → latest JPEG in memory
                                        ├→ browser preview
                                        └→ one model worker → short description + capture timestamp
```

- One camera at a time; Replace camera disconnects the previous test camera.
- Capture decodes continuously, publishes at most five preview frames/second and caps width at 960 px.
- Inference uses one image capped at 512×512, at most 100 generated tokens, and one request at a time. There is no accumulating inference backlog.
- Camera failures retry with increasing delay, capped at 30 seconds. Stale video is marked visibly; no new inference is scheduled from stale frames.
- Capture and inference are child processes. Stop terminates/reaps them; parent-death monitoring prevents orphan workers. Model loading/inference has a two-minute timeout.
- Up to 30 descriptions remain in memory. Each is tied to its original capture time, including when inference finishes after a disconnect.
- No video/clip storage, cloud vision API, calls, alerts, VDMA database access, tracking, or knowledge graph is implemented here.

This is **single-frame scene description**, not a temporal incident classifier. Descriptions can be incorrect and do not establish that a weapon/fight occurred. Use the lab to evaluate what it sees and how much compute it uses before choosing any production integration.

## Memory numbers

**Process RAM** is the model worker's resident memory measured after inference. **Peak MLX** is MLX's peak allocation for that request; the two values overlap and must not be added together. Neither number is total application/system memory. The model's ~376 MB checkpoint size is not its running RAM use. All local applications still share the same Mac hardware, despite process/package isolation.

## Files and boundaries

- `app.py`: local FastAPI UI/API, scheduling and worker lifecycle.
- `workers.py`: camera capture and local model inference.
- `static/index.html`: self-contained camera UI and observation history.
- `setup.sh`, `requirements.txt`, `requirements.lock`, `download_model.py`: reproducible local installation and pinned checkpoint download.
- `run.sh`, `manage.py`: foreground/detached startup and scoped shutdown.
- `test_lab.py`, `test_camera.py`, `smoke_model.py`: offline checks and synthetic end-to-end fixtures.
- `.venv/`, `.cache/`, `models/`, `.runtime/`: experiment-only, ignored by the nested `.gitignore`.
- `HANDOFF.md`: this experiment's checkpoint. The original project's handoff is intentionally unchanged.

The server binds only to loopback, checks Host/Origin, and requires a custom header for mutations. Keep it local; it has no remote-user authentication. URLs and observations are not persisted. Runtime logs contain service/model diagnostics, and capture decoder stderr is suppressed to avoid printing credential-bearing stream URLs.

## Verification

Offline checks (no camera, model or network access):

```sh
.venv/bin/python -m unittest -v test_lab.py
```

Real model smoke test on generated shapes (requires Metal GPU access):

```sh
.venv/bin/python smoke_model.py
```

Optional synthetic IP camera (does not open a physical camera):

```sh
.venv/bin/python test_camera.py
```

Then connect `http://127.0.0.1:8771` in the lab UI. Stop it with Ctrl+C after testing. Passing these checks verifies the pipeline; it does not measure CCTV description accuracy.

Model: [publisher model card](https://huggingface.co/LiquidAI/LFM2.5-VL-450M-MLX-4bit), [MLX-VLM](https://github.com/Blaizzy/mlx-vlm). The checkpoint is pinned to `f19926f17a25164d4cbcdc16d9eaf4714b807cfb`; its LFM 1.0 license is downloaded beside the weights. No remote model code is enabled. The older publisher example's MLX-VLM 0.3.9 was incompatible with the current processor; this lab uses the verified runtime in its lock file.
