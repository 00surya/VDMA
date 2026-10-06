# Event details and video-link API

A local client can request one saved event's place, location, detection time and a fresh HTTPS video link from VDMA. The API returns JSON to the caller; it does not send messages, call contacts, dispatch a response or change the review state.

## Endpoint and authentication

```text
POST http://127.0.0.1:8765/api/incidents/{incident_id}/event-package
X-VMD-Client: dashboard
Cookie: vmd_session=<centre session>
```

No request body is required. Sign in with `POST /api/auth/login`, the same client header, and JSON `{"password":"<centre password>"}`; retain the returned session cookie. Existing Host/Origin checks apply. A browser must use the same origin as the dashboard; a local non-browser client may omit `Origin`. This is the main centre session, not a mobile bearer token.

`GET /api/incidents` returns a JSON array of the latest 100 saved records; use the chosen record's `id` in the package URL. That dashboard list contains more fields than the event package: downstream consumers should use the package's allowlisted response.

## Response

Example only, with fictional coordinates and an unusable placeholder link:

```json
{
  "incident_id": "0123456789abcdef0123456789abcdef",
  "type": "knife_detected",
  "camera_id": "camera-example",
  "camera_name": "Entrance camera",
  "detected_at": "2026-09-27T10:15:30.000Z",
  "occurred_at": null,
  "source_kind": "live_camera",
  "source_seconds": null,
  "place": "Example entrance",
  "location": {
    "latitude": 12.34,
    "longitude": 56.78,
    "place": "Example entrance",
    "source": "browser",
    "verified": false,
    "accuracy_meters": 20.0,
    "captured_at": "2026-09-27T10:00:00.000Z"
  },
  "review_status": "unreviewed",
  "video_url": "https://example.invalid/e/<unguessable-token>",
  "video_expires_at": "2026-09-27T11:16:00.000Z"
}
```

- `detected_at` is the stored detection/analysis time, in UTC ISO 8601 with milliseconds. `occurred_at` remains `null`: the system does not establish the original occurrence time, especially for uploaded recordings.
- `source_kind` is `live_camera` or `recording`. `source_seconds` is the recording offset when available; it is not a clock time.
- `location` is the saved incident snapshot, not the laptop's current position. Missing or invalid coordinates produce `null`; top-level `place` is also nullable. Location provenance is `browser`, `manual` or `unknown`, and `verified` is always `false`. Laptop geolocation does not independently verify a remote camera's incident site.
- `type` is the stored event code; `review_status` is the current saved review label. Neither means an independently verified incident.
- `video_url` serves the saved incident clip as browser-playable MP4. Anyone holding it can watch until expiry or revocation. No stream URL, file path, contact list or raw detection signals are included in this package.

## Local Python example

This uses only Python's standard library and prompts for the password without echoing it. Run it on the VDMA laptop. It lists incident identifiers, then generates a link only for the identifier you choose.

```python
import getpass
import http.cookiejar
import json
from urllib.parse import quote
from urllib.request import HTTPCookieProcessor, Request, build_opener

base = "http://127.0.0.1:8765"
client = build_opener(HTTPCookieProcessor(http.cookiejar.CookieJar()))

def api(path, method="GET", body=None, timeout=15):
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = Request(base + path, data=data, method=method, headers={
        "Content-Type": "application/json",
        "X-VMD-Client": "dashboard",
    })
    with client.open(request, timeout=timeout) as response:
        return json.load(response)

api("/api/auth/login", "POST", {"password": getpass.getpass("Centre password: ")})
for event in api("/api/incidents"):
    print(event["id"], event["camera_name"], event["event_type"])

incident_id = input("Incident ID to prepare: ").strip()
package = api(
    f"/api/incidents/{quote(incident_id, safe='')}/event-package",
    "POST", timeout=195,
)
print(json.dumps(package, indent=2))  # Treat the returned bearer link as private.
```

## Hosting, expiry and errors

Start hosting from the project directory with `.venv/bin/python scripts/evidence_tunnel.py start`. The existing [evidence sharing setup](response-setup.md#evidence-video-links-through-cloudflare) covers `status` and `stop`.

Every successful POST creates a **new one-hour link**. The first request may need up to three minutes for H.264 conversion; cached clips return faster. Request packages when needed rather than polling this POST. If a client times out, a link may still have been issued; `GET /api/incidents/{id}/share` reports active counts, but cannot recover raw URLs. Repeating the POST creates another link.

`DELETE /api/incidents/{id}/share`, with the same session/client header, revokes **all links for that incident**, including dashboard- and SMS-issued links. False-positive review or response cancellation also blocks playback. The laptop, evidence gateway, tunnel and internet connection must remain online; restarting a Quick Tunnel changes its hostname and requires new links.

| HTTP status | Meaning |
| --- | --- |
| `200` | Package and fresh video link prepared. |
| `401` | Missing/expired centre session or incorrect login password. |
| `403` | Missing mutation header or disallowed origin. |
| `404` | Incident or saved/playable evidence file unavailable. |
| `409` | Incident is ineligible, cancelled, a false positive, or changed during preparation. |
| `503` | Evidence hosting is unavailable or video preparation failed. |

Only saved real-source incidents with evidence are eligible; synthetic/presentation and unknown-provenance records are excluded. Errors use JSON `{"detail":"..."}`.

The Cloudflare tunnel exposes only the evidence gateway on **8768**; it does not expose this API or the dashboard on **8765**. The separate [mobile API on 8766](mobile-api.md) remains read-only metadata and does not issue video links. Public/mobile access to this package endpoint and automatic delivery to a receiving API are not part of this implementation.
