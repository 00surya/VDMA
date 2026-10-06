# Gemini incident analysis and conversational calls

This copy now supports Vertex AI with Google Cloud authentication. An API key is optional; OAuth client ID/secret identify the application, while a signed-in Google identity authorizes Vertex access. No cloud call occurs at startup. Analysis uploads selected incident images; it does not expose the camera stream URL or upload the whole recording library.

## Install and authenticate

Run from this workspace, not the separate Desktop copy:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[test,ai]'
```

`.env` is already present with private placeholders and is loaded by `python -m vmd` and `python -m vmd.voice`. Exported environment settings take precedence. Do not commit this file. Replace `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET`, then verify `VMD_AI_VERTEX_PROJECT`: the supplied example is `climbrk-production`; choose the project covered by your credits. Enable Vertex AI and billing there, and give the signed-in identity permission to use Vertex AI (typically `roles/aiplatform.user`) and consume the billing project's services where required.

```sh
.venv/bin/python scripts/google_login.py
```

The helper opens an explicit Google consent flow for `cloud-platform` scope and saves the refresh credential privately to `data/google-credentials.json`, mode 0600. Use a Desktop OAuth client, or configure the web client's exact redirect `http://localhost:8089/`. Your consent app may require adding your Google account as a test user. No client secret or refresh token is printed. Never share these credentials.

Alternatively, use Application Default Credentials:

```sh
gcloud auth application-default login
gcloud auth application-default set-quota-project YOUR_PROJECT
```

Remove the `VMD_GOOGLE_CREDENTIALS_FILE` setting if you want ADC instead of a saved local OAuth credential. `GOOGLE_APPLICATION_CREDENTIALS` is supported through Google's ADC loader. If using AI Studio instead, set `VMD_AI_BACKEND=api` and `GEMINI_API_KEY`; that key is not required for Vertex. `CLIMBRK_AI_VERTEX_PROJECT`, `CLIMBRK_AI_VERTEX_LOCATION`, `CLIMBRK_AI_FAST_MODEL` and `CLIMBRK_AI_QUALITY_MODEL` are recognized when corresponding `VMD_AI_*` values are absent.

The default analysis models are the user's `gemini-3.1-flash-lite` and `gemini-3.5-flash`. Vertex access and regional support must be checked in your actual project; there is no silent switch to another paid model. Status means local credentials/configuration exist, not that a cloud call has succeeded.

## Dashboard workflow

Run `.venv/bin/python -m vmd` when port 8765 is free. The Desktop VDMA service currently owns that port; starting this copy does not update it. Preserve its state and stop/restart it only when intentionally switching copies. The environment in this copy has core, test, AI and vision dependencies installed. When recreating it, include the `vision` extra for neural camera processing. Camera/model runtime performance has not been revalidated by this integration.

1. Open a real saved incident with a clip. Choose **Gemini incident analyst** in the inspector, or **Gemini analysis & questions** in Incident Library.
2. **Analyze clip** uploads bounded sampled JPEGs to Vertex. Choose Fast or Quality. Processing occurs on a bounded background queue; alerts/calls continue independently.
3. Read the summary, visible subjects, relative clip timeline, supporting/contradicting/unclear assessment, uncertainties and draft call briefing. This does not change detector output, reviews, countdowns or dispatch.
4. **Approve briefing for future calls** allows the current draft to enrich later Twilio announcements and existing configured SMS. Withdrawal stops future use. Already submitted calls/messages cannot be changed. Analysis does not delay the ten-second automatic call, so a later report cannot enrich a call already submitted.
5. **Follow aftermath** analyzes fresh head-blurred frames from that same physical-camera session every ten seconds, for at most two minutes. It stops on stale/stopped/reconnected camera, false-positive review, evidence upgrade, expiry, explicit Stop, or two windows reporting involved subjects absent. Appearance matching remains tentative; this is not durable person re-identification. Recording analysis cannot follow a live camera.
6. Ask a question about the clip and saved later observations. Each answer rereads bounded clip frames and states limitations. Questions/answers are not persisted; initial and later reports are stored locally. Download the PDF report with a representative evidence image and clickable authenticated video link, or download JSON. Response-package ZIPs include both formats when the AI report is ready. Completed analysis also saves a private PDF under `data/ai-reports/`. Set `VMD_REPORT_BASE_URL` to the dashboard URL for links in these automatically saved PDFs; browser downloads use the current dashboard origin. Localhost links work only on the dashboard computer, and links do not bypass sign-in.

The separate aftermath buffer holds at most 30 seconds / 8 MiB of already rendered frames, sampled at 2 FPS independently of the incident evidence buffer. It is cleared on camera session replacement. It observes processed frames: quiet Eco's 2 FPS remains a temporal limit, and model gaps are not reconstructed. Initial clip sampling uses at most 48 frames / 8 MiB, caps resolution at 768 pixels and rejects clips longer than two minutes. At most eight queued jobs and one analysis/question provider request run per process. The persistent hourly request cap defaults to 60 and counts attempts; SDK retries are disabled. Reports are versioned against incident type, evidence clip and escalation so old approvals cannot attach to upgraded events.

Automatic incident analysis is enabled by default: `VMD_AI_AUTO_ANALYZE=true` queues eligible incidents as soon as their clips are saved, without an operator click. Set it to false to opt out. Clips finishing later than two minutes after detection are still picked up during the same runtime. Current saved reports and failed attempts are not automatically resubmitted on restart; errors can be retried explicitly. Historical incidents are not bulk uploaded. `VMD_AI_AUTO_FOLLOW=true` opts into bounded aftermath sessions for newly requested dispatches; follow remains false by default. Restart does not automatically resume old follow sessions. Automatic mode still does not approve reports or suppress local alerts.

## Conversational calling

The existing default `TWILIO_VOICE_MODE=incident` continues to use Twilio speech, enriched only by an approved report. Optional `gemini_live` mode connects Twilio to Gemini Live for responder questions.

1. Select a Live model and location available to your Vertex project using `VMD_AI_LIVE_MODEL` / `VMD_AI_LIVE_LOCATION`. The example is `gemini-3.8-live` / `global`; availability is not assumed or proven.
2. Configure an HTTPS reverse proxy/tunnel exposing **only** the new loopback voice gateway on 8769. Set `VMD_VOICE_PUBLIC_URL` to its HTTPS origin. Never expose dashboard port 8765. Existing evidence hosting on 8768 remains separate.
3. Start `.venv/bin/python -m vmd.voice` from the same directory/data configuration as main VDMA.
4. Set `TWILIO_VOICE_MODE=gemini_live`, restart main VDMA, and enable calling through the existing centre setting when ready. Existing Twilio credentials, recipient snapshots, hospital choice and one-attempt-per-incident/phone behavior remain in force.

The gateway exposes `/health`, signed `/twiml/{token}` and signed `/stream` only. It verifies Twilio signatures against the configured public origin, binds each expiring hashed token to its saved recipient and call SID, and consumes it once. Two concurrent conversations maximum; each lasts at most three minutes. Audio converts Twilio mu-law 8 kHz to Gemini PCM 16 kHz, and Gemini PCM 24 kHz back to mu-law. Transcripts/audio are not recorded or logged. Disable proxy access logging for token-bearing TwiML paths.

The AI introduces itself, answers from saved incident metadata and timestamped analyzed observations, and has one read-only context tool. It cannot dispatch, send evidence, call another person, identify people, or diagnose injuries. If asked for a video, it directs the responder to the operator's existing sharing workflow. Evidence delivery remains the existing explicitly configured SMS workflow; there is no new autonomous delivery action. False-positive review/cancellation closes an active bridge at its next guard check. Call outcome is still tracked by the original Twilio adapter.

If the local voice gateway is unavailable when submitting a call, the adapter uses the original spoken announcement. If Live fails after a call connects, TwiML announces that the AI briefing ended and directs the recipient to the centre operator. No new automatic redial is added. Phone-account permissions, tunnel reachability, audio quality, model access, human answers and carrier outcomes require real commissioning.

## Verification

```sh
.venv/bin/python -m pytest -q tests/test_gemini.py tests/test_gemini_voice.py tests/test_calling.py tests/test_response.py
.venv/bin/node --test tests/*.test.mjs
```

Tests use generated clips, fake model responses and simulated Twilio audio. They verify integration and guards, not footage-analysis accuracy. No real Google inference, private evidence upload or phone call has been made by implementation. Keep pose tracking, weapon detection, depth and local heuristics; Gemini is a second reviewer and narrative layer. The standalone local labs remain available for comparisons.

Official references: [Vertex authentication](https://docs.cloud.google.com/docs/authentication/provide-credentials-adc), [Google Gen AI SDK](https://github.com/googleapis/python-genai), [video understanding](https://ai.google.dev/gemini-api/docs/video-understanding), [Live API](https://ai.google.dev/gemini-api/docs/live-api), [Twilio Media Streams](https://www.twilio.com/docs/voice/media-streams).

## Generated-scene demonstration

An isolated demo is available at `http://127.0.0.1:8876/` when running:

```sh
.venv/bin/python scripts/gemini_demo.py
```

It generates a short animation, assigns a test-only possible-fight label, and lets you analyze the actual generated frames or ask evidence questions using real Google inference. It has its own ignored `tmp/gemini-demo/` storage and no response, calling, SMS or sharing services. Existing saved reports display without another model request. `--run` explicitly submits a new analysis at startup. Browser Analyze/Ask actions also consume model requests. The animation and its label do not measure the local detector's accuracy.

First real demonstration: Quality recognized the arm exchange and fall, but missed the red figure moving away, describing it as staying nearby. Compare generated observations with the video; treat the report as a draft.
