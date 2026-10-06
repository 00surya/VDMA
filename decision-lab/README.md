# Decision Lab

A standalone, local GLiNER2.5-Decide playground inside VDMA. Its dark green/lime interface follows the supplied state → question → options → ranked answers reference. It does not import the camera application or trigger detection, dispatch, calls or messages.

- Playground: <http://127.0.0.1:8771/>
- Architecture explainer: <http://127.0.0.1:8771/explainer>
- Weights: `decision-lab/models/GLiNER2.5-Decide/` (about 1.95 GB decimal).
- Runtime: dedicated Python 3.11 `.venv`, CPU with two PyTorch threads, one inference slot, loopback only.

## Start

From the VDMA repository:

```sh
./decision-lab/run.sh
```

Wait for **Local model · CPU**. Initial loading can take roughly 20 seconds on this laptop. An already-running instance owns port 8771; do not start duplicates. `run.sh` runs in the foreground, so Ctrl-C stops that instance. For the detached development instance, PID and diagnostics are under ignored `decision-lab/tmp/server.pid` and `server.log`; verify the listener/process before stopping it. There is no login/startup service.

To reinstall, with internet available:

```sh
./decision-lab/setup.sh
```

Setup installs `requirements.lock` into its own environment and downloads only the pinned checkpoint assets. Inference forces `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` and loads from the local directory. No API key is needed. User inputs/results are not persisted, and access logging is disabled. Package caches, model files, environments and local QA outputs are ignored by Git.

## Use

Enter the state and question, edit/add/remove 2–12 options, then **Run decision**. The result shows every option, the top choice, its percentage-point lead, encoded token count and measured model-call duration. Inspect/copy the actual JSON response below. Editing any input clears the old result; an in-flight response cannot be applied to changed inputs. Example presets include the screenshot’s damaged-package request, a fictional incident description and a passage question.

Limits: English model; state ≤6,000 characters, question ≤300, each option ≤80. The complete encoded text + schema must fit within this app’s 512-token budget. Oversized requests are rejected without truncation. Reserved tokenizer markers and duplicate options are rejected. A second concurrent request gets HTTP 429; no inference queue accumulates.

## Local API

`GET /api/status` returns `loading`, `ready` or `error` with checkpoint and device metadata. `POST /api/decide` accepts JSON and requires `X-Decision-Client: local-ui`; browser callers must have the same origin. No CORS access or public tunnel is configured. Bind is fixed to `127.0.0.1`.

```sh
curl http://127.0.0.1:8771/api/decide \
  -H 'Content-Type: application/json' \
  -H 'X-Decision-Client: local-ui' \
  -d '{"state":"My package arrived damaged and I want a refund.","question":"Which queue should handle this?","options":["billing","shipping","technical","general"]}'
```

Response fields: `answer`, sorted `scores` (`label`, `score`), `latency_ms`, `input_tokens`, `margin`, `model`, `revision`, `device`, `score_type` and `calibrated`. Scores are native softmax outputs. GLiNER’s `multi_label=True` is used solely to retain all labels at `cls_threshold=0`; `class_act="softmax"` explicitly preserves single-choice scoring. Native single-label output is checked for exact top-label/score parity. Model-call timing excludes initial loading, app prevalidation and HTTP transit.

## Model and source notes

The exact [Fastino checkpoint](https://huggingface.co/fastino/GLiNER2.5-Decide) is pinned at `7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6` (Apache 2.0). The publisher calls it 340M; the downloaded weight file size is a separate quantity. Its saved config says **span**, **SpanExtractor**, **DeBERTa-v3-large**. AutoExtractor chooses that architecture. Do not substitute a GLiNER2.5 boundary model or Jev. GLiNER2 2.0.0 and transformers 4.57.6 use eager attention for this encoder; the library warns when falling back from the checkpoint’s SDPA preference. No model code was patched.

The explainer uses the English (US) auto-generated subtitles of [CampusX, “Jev by TypeSafe AI | What is a System-1 Decision Model”](https://www.youtube.com/watch?v=0zFfcEr1e9U), published 24 September 2026. Retrieved 2,092 timestamped segments on 27 September. Full subtitles stay in ignored `../tmp/decision-lab-research/`; the page contains short original paraphrases and source timestamp links. Relevant segments: 04:06 decision interface; 12:35 parallel questions; 16:02 calibration; 55:20 architecture caveat; 70:15 proposed answer head; 76:26 proposed parallel paths; 86:03 planner + decision-model workflow.

The presenter’s Jev architecture is explicitly conjectural. The page distinguishes it from the installed encoder/classifier, and does not claim Jev’s RLCD training, calibration or latency for GLiNER.

## Checks and limitations

```sh
decision-lab/.venv/bin/python -m pytest -q decision-lab/test_app.py
decision-lab/.venv/bin/python decision-lab/smoke_model.py
node --check decision-lab/static/app.js
```

The smoke check blocks socket connections, loads real local weights, verifies native score parity and repeats three cases to check stable output. It tests plumbing, not model accuracy. Unit checks cover validation, concurrency, token limits, score integrity, origin/host restrictions, body limits and static routes.

Observed on this Apple M2 laptop: model calls around **256–397 ms** in initial checks, dependent on input and system load. The screenshot’s shipping 0.71 / ~120 ms is illustrative. Actual damaged-package output was **billing 0.34245, shipping 0.33079, general 0.19895, technical 0.12781**. The model also answered a treaty-date question incorrectly with high confidence in the smoke check. Softmax is not calibrated correctness, and this is not a reasoning or explanation generator. Keep an unknown/review option where needed and validate on labelled examples before using scores in operational workflows.
