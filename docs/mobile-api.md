# Incident API for React Native

The metadata gateway exposes saved incident locations and detection times from this installation. It opens the existing SQLite database in **read-only mode**. It does not start cameras, models, dispatch workers or migrations. The dashboard stays on port 8765; the optional gateway runs separately on port 8766.

## Start locally

Register the installation's centre in the dashboard first. From the repository directory:

```sh
.venv/bin/python -m uvicorn vmd.mobile:app --host 127.0.0.1 --port 8766 --workers 1 --no-access-log
```

`VMD_DATA_DIR` can point at an existing installation's data directory. Missing/unavailable data returns 503; the gateway never creates a new database. No new packages are needed.

For a phone outside the laptop, configure an HTTPS reverse proxy to **this gateway only**, then set `VMD_MOBILE_HOSTS` to the exact external hostname before starting it. For example, `VMD_MOBILE_HOSTS=incidents.example.org` (replace this example with your hostname). Keep the gateway bound to loopback and let the proxy preserve the external Host and scheme. Trust forwarded headers only from your actual local proxy, never arbitrary network clients. Do not forward the dashboard's port 8765. The existing evidence gateway, if used, is separate on port 8768.

Remote HTTP is rejected; plain HTTP is accepted only from a loopback client for local development or the proxy hop. The proxy must enforce HTTPS on its public listener. Default accepted hosts are `127.0.0.1`, `localhost` and `[::1]`; wildcard hosts are rejected. Native React Native requests do not need browser CORS. Browser requests from another origin are rejected.

No domain, certificate, public tunnel or phone connection is created by this implementation. HTTPS deployment and real-device integration must be verified before remote use.

## Authentication and access

| Method / path | Purpose |
| --- | --- |
| `POST /api/v1/auth/token` | JSON body `{"password":"the centre password"}`; returns a one-hour bearer token |
| `POST /api/v1/auth/logout` | Revoke the supplied bearer token; returns 204 |
| `GET /api/v1/incidents` | Paginated incident metadata |
| `GET /api/v1/incidents/{incident_id}` | Current metadata and review status for one incident |

Every incident request and logout requires `Authorization: Bearer <access_token>`. Password login is the only unauthenticated API operation; registration remains local. Tokens are opaque random values, with only their SHA-256 hashes retained in process memory. They expire after 3600 seconds, can be revoked by logout, and are all revoked by restarting the gateway. A changed centre password invalidates existing tokens. Dashboard cookies do not grant gateway access, and mobile tokens do not grant dashboard access.

**Scope:** one centre per installation, matching the current VDMA data model. A centre login grants read access to that installation's real incidents. This is not individual staff authentication, per-camera permissions or a public/community incident feed. Do not distribute the centre password to public app users. Those use cases require separate identities and access rules before deployment. Store the returned token in the phone's protected credential storage; do not embed credentials in app source, URLs, logs or persistent plain-text storage.

The gateway returns no camera URLs, responder contacts, scores, raw detection signals, clip paths or video. It has no camera, review, dispatch, signup or administrative routes. All responses use `Cache-Control: no-store`.

Rate limits per process: 5 login attempts per minute (including successful logins), 120 requests per token per minute, and 600 requests total per minute. A 429 response includes `Retry-After` in seconds. At most 128 active tokens are retained. These bounds suit a small, single-centre deployment. Run **one worker**; multiple workers would have separate tokens and counters. A reverse proxy should also bound request size, connection duration and request volume.

## Data contract

Fictional response from `GET /api/v1/incidents`:

```json
{
  "incidents": [
    {
      "incident_id": "example-incident",
      "type": "gun_detected",
      "camera_id": "camera-1",
      "camera_name": "Example entrance",
      "detected_at": "2026-09-27T10:30:00.000Z",
      "occurred_at": null,
      "source_kind": "live_camera",
      "source_seconds": null,
      "location": {
        "latitude": 28.6139,
        "longitude": 77.209,
        "place": "Example entrance",
        "source": "browser",
        "verified": false,
        "accuracy_meters": 25,
        "captured_at": "2026-09-27T10:00:00.000Z"
      },
      "review_status": "unreviewed"
    }
  ],
  "next_cursor": null,
  "server_time": "2026-09-27T10:30:05.000Z"
}
```

- `detected_at` is the original saved detection/analysis time, in UTC ISO 8601. Escalation preserves that original timestamp. It is not the upload's original capture time or the moment a request was sent. Display it locally using `new Date(record.detected_at).toLocaleString()`.
- `occurred_at` is currently always `null`: the existing pipeline has no verified original occurrence timestamp. Keep that distinction in the app UI.
- `source_kind` is `live_camera` or `recording`, based on an actual stored boolean provenance flag. Demos, presentation records, invalid/missing provenance and future detection timestamps are excluded.
- Recording `source_seconds` is a valid nonnegative offset in the source video, when available. Physical-camera relative capture clocks are not exposed as video offsets.
- `location` comes from the incident's saved snapshot, never today's camera configuration or the viewer phone's GPS. It is `null` for missing/invalid coordinates. Its `source` is `browser`, `manual` or `unknown`. Browser coordinates describe the laptop's reported location; they do not independently locate a remote camera. Legacy source-less locations remain `unknown`.
- `location.verified` is always `false`: neither a typed location nor browser GPS has been independently verified as the incident's location. Even human confirmation of an incident does not change this flag. Accuracy and capture time are nullable metadata. Map renderers must handle their own supported latitude range without inventing replacement coordinates.
- `review_status` uses the existing `unreviewed`, `confirmed` and `false_positive` values. An automated event type is not itself human confirmation.

## Filters and pagination

| Query | Default | Meaning |
| --- | --- | --- |
| `limit` | `50` | Page size, 1–100 |
| `cursor` | absent | Opaque `next_cursor` from the previous page |
| `since` | absent | Inclusive detection-time lower bound, with timezone, e.g. `2026-09-27T00:00:00Z` |
| `source` | `live_camera` | `live_camera`, `recording` or `all` |
| `include_false_positives` | `false` | Include false positives so clients can reconcile cached records |

Results sort by detection time descending, with incident ID as a stable tie-breaker. Follow `next_cursor` until null, keeping all filters unchanged, to reach records beyond the dashboard's latest-100 cap. Pagination does not freeze the database; concurrent reviews/escalations can change the records. Refresh the first page for new incidents.

For foreground polling, fetch every 5 seconds while the app is active, with `include_false_positives=true`, update records by `incident_id`, and remove false positives from map markers. Use `source=live_camera` for the live map. Do not plot missing locations. Re-fetch cached older incidents by ID when opened and periodically reconcile the relevant date range: `since` filters the **original detection time**, not modification time, so it is not a change feed. The detail endpoint deliberately returns false-positive status for an incident already cached by a client.

401 means sign in again; 404 means the incident is missing or ineligible; 422 means invalid input; 429 means wait `Retry-After`; 503 means data/session capacity is unavailable. Show connection errors as unavailable/stale data, not as zero incidents. Notifications while the phone app is in the background require a separate push integration.

## Minimal React Native request flow

This is integration example code, not a new mobile application. Use your configured HTTPS hostname:

```javascript
const API = 'https://incidents.example.org';

async function signIn(password) {
  const response = await fetch(`${API}/api/v1/auth/token`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ password }),
  });
  if (!response.ok) throw new Error(`Sign-in failed (${response.status})`);
  return response.json(); // access_token, expires_in, expires_at, scope
}

async function loadIncidents(accessToken) {
  const response = await fetch(
    `${API}/api/v1/incidents?source=live_camera&include_false_positives=true`,
    { headers: { Authorization: `Bearer ${accessToken}` } },
  );
  if (!response.ok) throw new Error(`Incidents unavailable (${response.status})`);
  return response.json(); // incidents, next_cursor, server_time
}
```

The app must handle token expiry, rate-limit backoff, pagination and foreground lifecycle, and render saved location provenance honestly. Secure external hosting and real-phone testing remain deployment tasks.
