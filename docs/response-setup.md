# Centre registration and incident response

The local response agent is a deterministic workflow. Voice calls use Twilio; no LLM or public tunnel is needed for calling.

## Operator flow

1. Open the dashboard and register this installation's centre with a password of at least ten characters, one or more authority contacts, and a hospital name/phone. Use phone numbers with a country code, for example `+12025550101`. One centre is supported per installation; existing cameras, clips and reviews remain intact. The password is stored as a salted scrypt hash. The backend requires a session cookie; the old demo role selector no longer grants API access.
2. Single-click a camera card to select it with a red border; double-click to open its large viewer. Each fresh live incident selects its camera once. In the inspector choose **Set camera place & coordinates**, or enter them when adding the camera. Use the camera's actual location, not a fictional sector marker or the laptop's location unless it is at the camera. Locations persist across recovery and are captured with each incident; later edits do not rewrite old locations.
3. Click **Enable alarm sound** on the monitoring device. Browsers require a user gesture before sound can play. Keep that dashboard open and the device audio audible; sound cannot play from a closed/suspended browser. Enable sound in only one monitoring tab to avoid overlapping alarms.
4. A new physical live-camera **fight, knife or snatching** incident starts a beep and a ten-second countdown. The beep volume increases over that interval. **Acknowledge**, confirming a review during the countdown, or **Mark False Positive** stops automatic escalation. Acknowledgement is separate from incident review. The server owns the deadline, so a page refresh does not reset it.
5. With no action, dispatch is requested at ten seconds. With calling configured and enabled, **Slide to dispatch** requests authority calls immediately and turns gray/disabled. A large centered prompt appears afterward to ask whether the saved hospital should be called too. **Request hospital** reopens it while that incident remains in the inspector, until the hospital is requested. Adding the hospital does not redial authorities. Gun and other live-camera incidents can be dispatched manually. Saved recording-analysis incidents can also be dispatched explicitly from the finished feed or **Incident Library → Review dispatch**; the saved-incident dialog works even after its analysis camera is removed. It does not dispatch merely by opening the dialog. Recordings, pipeline tests, presentation scenarios and historical pre-upgrade incidents never automatically notify contacts.
6. Edit destination contacts in **Centre settings**. Calling defaults off; enable it after configuring Twilio and verifying a designated test recipient. **Recent response details** holds recipient statuses and provider errors, leaving the inspector focused on the incident and actions. Queued does not mean answered; completed does not prove a human understood or acted on the alert. Evidence downloads remain in **Incident Library**; **Watch evidence** remains in the inspector.

Acknowledging or disabling calling stops calls not yet submitted; it cannot recall an in-flight or submitted call. Declining or closing the hospital prompt leaves the authority request active. Marking an incident false positive also revokes its evidence links. Existing SMS messages remain on recipients' devices.

For physical-camera incidents, thirty seconds after the first dispatch request or false-positive review, the saved incident clears from the operational inspector and a connected camera returns to **Monitoring**. Refreshing or restarting does not reset this countdown. Detection continues throughout, and the next distinct incident from the camera has its own evidence and dispatch controls. The timeout preserves the original incident, clip and response history, including failed-call details; it does not mark delivery successful or cancel calls. A hospital dialog already open stays bound to its original incident. Unhandled incidents stay available for review. Recording reviews stay available after handling without claiming that live monitoring resumed.

## Voice calling setup

**Current integration (September 26): incident calls now use the earlier cloud-caller's `To` / `From` / `Url` request format.** The server generates the incident's XML and URL-encodes it into `https://twimlets.com/echo?Twiml=...`. Echo returns the XML for Twilio to read. The prior inline `Twiml` request was rejected by this active Trial account; that does not establish that every hosted custom-speech route is forbidden or that an upgrade is the only solution. The user's current calling toggle is preserved.

Echo GET and POST were checked with fictional incident text and returned the matching speech plus Hangup. This verifies hosting and encoding, not this account's permission to create an outbound Echo call or handset delivery. No new real call has been placed for this change. Trial account and verified-recipient restrictions still apply; see [Twilio's trial Voice documentation](https://www.twilio.com/docs/usage/trials/try-out-voice#using-the-voice-api). No automatic fallback to a demo is used.

Set `mode` in private `data/twilio.json`, or `TWILIO_VOICE_MODE` in the environment, to `incident` for the custom announcement. The explicit optional `trial_template` mode now uses the requested `https://webhooks.twilio.com/v1/Voice/Template/voice_text_to_speech` URL with only `To`, `From` and `Url`. This selects Twilio's hosted TTS sample, not the incident announcement. Twilio documents no custom-text parameter for that template. Earlier real-call verification used the speech-recognition template; the replacement has passed a stubbed payload test but has not been called. It is not this installation's selected mode. Invalid modes disable configuration; there is no automatic fallback to the demo.

To commission incident speech, enable **Twilio voice calls** in Centre settings and save, then use a reviewed incident and designated recipient to verify the announcement. Attempting a manual call while disabled opens settings; the API refuses to queue it. Trial destination numbers must be verified in Twilio. App tests use stub senders and never place real calls.

Set `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN` and `TWILIO_FROM_NUMBER` in the dashboard process environment, or use the private, Git-ignored `data/twilio.json` file with string keys `sid`, `token` and `sender`. Restrict this file to its owner (`chmod 600 data/twilio.json`). Environment variables override the file. The file belongs to that data directory; an isolated test installation does not inherit it. The browser never receives the token. `.env.example` is a template. Copy it to `.env` and install the `ai` extra to let `python -m vmd` load it through python-dotenv; existing process environment variables take precedence.

The sender must be a Twilio voice-capable number permitted by your account. The destination is each saved centre contact, in international format. Twilio's account permissions and destination restrictions can still reject a call; the dashboard shows the rejection code. This connects to configured phone numbers, not a special emergency dispatch network.

In **incident mode**, the app uses the [Twilio Calls API](https://www.twilio.com/docs/voice/api/call-resource) with a dynamic Twimlets Echo instruction URL: “Hello, we are calling from {centre}. {Incident} was detected at {place} at {time}.” It includes the camera and original coordinates, asks for verification and a response, repeats twice, then hangs up. The time is the original detection timestamp in the server's local timezone, including the date and timezone; it is not the later dispatch time. Recording announcements instead identify recorded footage, analysis time, the source-video offset when available, and configured location as unverified. They do not claim that the incident occurred at the analysis time. Missing location is stated explicitly. Speech uses `Polly.Joanna` / `en-US`. Both modes submit only `To`, `From` and `Url`, so ringing and call duration use provider defaults. The Echo URL contains the announcement (including location) and is sent to Twilio/Twimlets; anyone holding that URL can retrieve the text. It contains no account credentials and must not be logged or shared. Neither requires clip encoding or Cloudflare.

Each incident/phone gets one submission attempt, even when hospital and authority share a number. Repeated polling, slider actions and hospital requests cannot duplicate a submitted call. Contact destinations are frozen when calling starts. Timeout, server error or interruption during submission is marked uncertain and is not automatically retried. Check Twilio logs before any recovery. Known call IDs are polled every five seconds for up to ten minutes; unavailable final status is labelled uncertain. No callback URL is required. Requests waiting over five minutes do not call later just because credentials appear; review and dispatch again to request still-unattempted recipients. Pre-upgrade SMS requests do not become voice calls automatically.

## Evidence video links through Cloudflare

**Incident Library → Video link** prepares a link separately from dispatch. Opening the dialog reads sharing status and existing link counts. **Generate video link** prepares the saved evidence as H.264 MP4 and returns an unguessable HTTPS link valid for one hour. **Copy link** and **Open link** appear after generation. Anyone holding the link can view the clip until expiry; keep the laptop, gateway, tunnel and Internet connection running. Closing the dialog discards its link address; the database stores only a token hash, so reopening shows counts and allows generating another link.

**Revoke all links** disables every existing link for that incident, including links issued by SMS. It does not change dispatch or call state. False-positive reviews and cancelled responses also block playback. Undispatched physical-camera and recording incidents are supported; synthetic/demo events and unknown source provenance are excluded. Missing recording location does not prevent preparing a video link. Only the video is public through this link; assembling and sending incident details/location through a messaging API is the next phase.

From the repository, with the main environment and `cloudflared` installed:

```sh
.venv/bin/python scripts/evidence_tunnel.py start
.venv/bin/python scripts/evidence_tunnel.py status
.venv/bin/python scripts/evidence_tunnel.py stop
```

`start` launches only the evidence gateway on **127.0.0.1:8768** and a free Quick Tunnel to that port. It verifies the local gateway and public health endpoint before writing the HTTPS origin and owned-process identities to private `data/evidence-host.json`. VDMA reads this file dynamically; no restart is needed when the tunnel origin changes. The helper refuses occupied ports, isolates Cloudflare configuration, uses HTTP/2, and writes owner-only logs under `data/evidence-runtime/`. `stop` clears configuration and stops only its own tunnel/gateway; it leaves VDMA and cameras running. This is a POSIX helper tested on macOS; it is not installed as an operating-system startup service.

The dashboard **8765** stays local. Quick Tunnels are free for testing, have changing addresses after restart, and have no uptime guarantee. Generate fresh links after restarting the tunnel. For a later stable deployment, use a named tunnel or object storage. See [Cloudflare Quick Tunnels](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/).

For separately managed hosting, the main process can use `VMD_EVIDENCE_BASE_URL=https://your-evidence-origin` instead. This environment override takes priority over the managed file; remove it when switching back to the helper. HTTPS configuration is independent of Twilio. The public gateway exposes only `/health` and token-gated `/e/{token}` GET/HEAD requests, supports range seeking, and sends no-store headers. The video remains on the laptop; no storage bucket is created.

Public HTTPS playback, HEAD/range requests, browser playback/seeking and revocation were verified with a generated sample in isolated storage. No private incident clip, call or message was sent during that check.

## Optional SMS configuration (separate, later setup)

Configure the four variables below in the process environment or in an ignored `.env` file, then restart the dashboard. `python -m vmd` loads `.env` when python-dotenv from the `ai` extra is installed. Keep actual credentials in an ignored/private file or your environment; do not paste them into Git or the browser.

- `TWILIO_ACCOUNT_SID`
- `TWILIO_AUTH_TOKEN`
- `TWILIO_FROM_NUMBER`: an SMS-capable Twilio sender
- `VMD_EVIDENCE_BASE_URL`: the HTTPS origin of the running evidence gateway

Twilio account capabilities, destination permissions and sender configuration still determine whether SMS is accepted/delivered. This is an integration with the phone numbers you configure, not a connection to an emergency dispatch system. Use a designated test recipient for commissioning. The implementation posts a form-encoded message to the [Twilio Messages API](https://www.twilio.com/docs/messaging/api/message-resource). A direct video attachment through a media channel would still require an accessible media URL and channel-specific setup; see [Twilio media requirements](https://www.twilio.com/docs/messaging/guides/accepted-mime-types).

The managed tunnel does not enable SMS or copy its URL into the SMS adapter. After explicitly configuring SMS, enable **Twilio SMS dispatch** in Centre settings. Missing configuration, missing physical-camera coordinates or missing/unencodable evidence prevents sending; the app does not silently substitute fictional locations or omit the clip. A manually dispatched recording may have unknown location: its package uses null and its SMS explicitly says location unknown without a map. If a recording has a configured location, the message identifies it as unverified. Clip preparation can delay delivery beyond the ten-second request time.

Each incident/recipient gets one submission attempt. Repeated polling and slider actions do not resend it. A timeout or app interruption during sending is marked uncertain; inspect Twilio's logs before any manual recovery to avoid duplicate messages. No automatic retry of failed/uncertain SMS is implemented. Requests waiting more than five minutes are not released automatically when credentials later appear; review the incident and slide again. Contacts are fixed when preparation starts, so later contact edits do not reroute existing requests. Failed video preparation can be retried with a new slider action.

## Local persistence and checks

The existing ignored SQLite database stores the centre, session hashes, response deadlines, contact snapshots, call/delivery results and hashed share tokens. No secrets are added to API responses. Learning from reviewed clips remains deferred until last.

```sh
.venv/bin/python -m pytest -q
node --test tests/*.test.mjs
```

Tests use fictional contacts and stub senders. They cover registration/session protection, countdown persistence, hospital requests, dynamic contacts, acknowledgement/review cancellation, duplicate suppression, disabled/expired/pre-upgrade requests, manual-only recording dispatch, synthetic/unknown-source exclusion, original-time live speech and recording analysis-time speech and TwiML escaping, call status updates, clip packaging, isolated gateway/range requests, token expiry/revocation and uncertain outcomes. No real phone calls or SMS are made by the tests. Custom spoken announcements and device alarm volume still require a designated real test after provider setup.
