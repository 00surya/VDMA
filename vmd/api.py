import os
import base64
import time
import uuid
import json
import io
import zipfile
from urllib.parse import unquote
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, StrictBool, ValidationError, create_model, model_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.concurrency import run_in_threadpool

from .capture import validate_source
from .engine import Engine
from .cameras import CameraManager, EDITABLE_SETTINGS
from .media import MediaLibrary, VIDEO_EXTENSIONS, MAX_UPLOAD
from .evidence import EvidenceSharing
from .heuristics import Rules
from .storage import Store
from .alerts import AlertAgent
from .centre import Centre, CentreDetails, CentreSignup, CentreLogin
from .delivery import Delivery
from .calling import Calling
from .gemini import IncidentAI


class CameraLocation(BaseModel):
    place: str = Field(min_length=1, max_length=200)
    latitude: float = Field(ge=-90, le=90, allow_inf_nan=False)
    longitude: float = Field(ge=-180, le=180, allow_inf_nan=False)
    source: Literal['manual', 'browser'] = 'manual'
    accuracy_meters: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    captured_at: float | None = Field(default=None, ge=0, allow_inf_nan=False)

    @model_validator(mode='after')
    def meaningful_place(self):
        self.place = self.place.strip()
        if not self.place:
            raise ValueError('Enter the camera location name')
        return self


class StartRequest(BaseModel):
    name: str = Field(default="", max_length=64)
    mode: Literal["live", "demo"] = "demo"
    source: str = Field(default="", max_length=2048)
    recording_id: str | None = Field(default=None, pattern=r'^[a-f0-9]{32}$')
    device: Literal["auto", "cpu", "mps", "cuda:0"] = "auto"
    depth: Literal["off", "ZipDepth", "MiDaS_small", "DPT_Hybrid", "DPT_Large"] = "off"
    detection_mode: Literal["responsive", "depth_confirmed"] = "responsive"
    eco_mode: StrictBool = False
    object_detection: StrictBool = False
    unattended_objects: StrictBool = False
    snatching_vehicles: StrictBool = False
    unattended_seconds: float = Field(default=60, ge=10, le=3600, allow_inf_nan=False)
    person_down_seconds: float = Field(default=3, ge=1, le=60, allow_inf_nan=False)
    object_fps: float = Field(default=1, ge=.2, le=2)
    threshold: float = Field(default=Rules.threshold, ge=.4, le=.95)
    hold_seconds: float = Field(default=Rules.hold_seconds, ge=.3, le=5)
    fight_confirmation_seconds: float = Field(default=3, ge=.5, le=10)
    target_fps: int = Field(default=8, ge=2, le=30)
    depth_fps: float = Field(default=.5, ge=.1, le=2)
    location: CameraLocation | None = None

    @model_validator(mode="after")
    def source_valid(self):
        if self.eco_mode and self.mode != "live":
            raise ValueError("Eco mode is available for camera and recorded-video inputs")
        if self.object_detection and self.mode != "live":
            raise ValueError("Object detection needs a camera or recorded-video input")
        if self.unattended_objects and self.mode != "live":
            raise ValueError("Unattended-item detection needs a camera or recorded-video input")
        if self.snatching_vehicles and self.mode != "live":
            raise ValueError("Rider-snatching review needs a camera or recorded-video input")
        if self.detection_mode == "depth_confirmed" and (self.depth == "off" or self.depth_fps < .5):
            raise ValueError("Depth-confirmed detection needs a depth model sampling at least 0.5 times per second")
        if self.recording_id and (self.mode != 'live' or self.source.strip()):
            raise ValueError('Choose either an uploaded recording or a camera source')
        if self.mode == "live" and not self.recording_id:
            validate_source(self.source.strip())
            self.source = self.source.strip()
        return self


# Use the creation field limits for edits too; omitted fields retain their saved values.
CameraSettingsRequest = create_model('CameraSettingsRequest', __config__=ConfigDict(extra='forbid'),
    **{name: (StartRequest.model_fields[name].annotation, StartRequest.model_fields[name])
       for name in EDITABLE_SETTINGS})


class AIAnalysisRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    quality: StrictBool = True


class AIQuestionRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    question: str = Field(min_length=1, max_length=600)


class AIToggleRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    enabled: StrictBool


class ScenarioRequest(BaseModel):
    scenario: Literal["calm", "interaction", "camera"]


class ReviewRequest(BaseModel):
    decision: Literal["confirmed", "false_positive", "unreviewed"]


class EcoRequest(BaseModel):
    enabled: StrictBool


class ResponseRequest(BaseModel):
    action: Literal['acknowledge', 'cancel', 'dispatch', 'hospital', 'call_hospital']
    hospital: StrictBool | None = None


def create_app(data_dir=None, model_dir=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.store = Store(data_dir or os.getenv("VMD_DATA_DIR", "data"))
        app.state.centre = Centre(app.state.store)
        app.state.alerts = AlertAgent(app.state.store)
        app.state.store.on_incident = app.state.alerts.register
        models = model_dir or os.getenv("VMD_MODEL_DIR", "models")
        app.state.media = MediaLibrary(app.state.store)
        app.state.evidence = EvidenceSharing(app.state.store, app.state.media)
        app.state.delivery = Delivery(app.state.store, app.state.centre, app.state.alerts, app.state.media)
        app.state.calling = Calling(app.state.store, app.state.centre, app.state.alerts)
        app.state.manager = CameraManager(app.state.store, models, StartRequest)
        app.state.cameras = app.state.manager.cameras
        app.state.camera_lock = app.state.manager.lock
        app.state.engine = app.state.cameras.get('camera-1') or Engine(app.state.store, models)
        app.state.ai = IncidentAI(app.state.store, lambda key: app.state.cameras.get(key))
        app.state.calling.ai = app.state.ai
        app.state.delivery.ai = app.state.ai
        yield
        app.state.ai.close()
        app.state.calling.close()
        app.state.delivery.close()
        app.state.manager.close()
        app.state.engine.stop()
        app.state.store.close()
        app.state.alerts.close()

    app = FastAPI(title="VMD Shield local API", lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]", "testserver"])

    @app.middleware("http")
    async def local_origin(request: Request, call_next):
        origin = request.headers.get("origin")
        if origin and origin != f"{request.url.scheme}://{request.headers.get('host')}":
            return JSONResponse({"detail": "Cross-origin access is not allowed"}, status_code=403)
        if request.method in {"POST", "PUT", "PATCH", "DELETE"} and request.headers.get("x-vmd-client") != "dashboard":
            return JSONResponse({"detail": "Local client header required"}, status_code=403)
        public = {'/api/auth/status', '/api/auth/signup', '/api/auth/login'}
        if (request.url.path.startswith('/api/') and request.url.path not in public
                and not app.state.centre.authenticated(request.cookies.get('vmd_session'))):
            return JSONResponse({'detail': 'Sign in to your centre'}, status_code=401)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' data: https://tile.openstreetmap.org; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        else:
            # Revalidate the local UI after edits instead of retaining stale controls.
            response.headers["Cache-Control"] = "no-cache"
        return response

    @app.get('/api/auth/status')
    def auth_status(request: Request):
        signed_in = app.state.centre.authenticated(request.cookies.get('vmd_session'))
        details = app.state.centre.details()
        return {'registered': bool(details), 'authenticated': signed_in,
                'centre': details if signed_in else None}

    def signed_in_response(token):
        response = JSONResponse({'centre': app.state.centre.details()})
        # The dashboard is loopback HTTP. The public gateway never accepts this cookie.
        response.set_cookie('vmd_session', token, httponly=True, samesite='strict', max_age=86400)
        return response

    @app.post('/api/auth/signup')
    def signup(body: CentreSignup):
        try:
            return signed_in_response(app.state.centre.signup(body))
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post('/api/auth/login')
    def login(body: CentreLogin):
        try:
            return signed_in_response(app.state.centre.login(body.password))
        except ValueError as exc:
            raise HTTPException(401, str(exc)) from exc

    @app.post('/api/auth/logout')
    def logout(request: Request):
        app.state.centre.logout(request.cookies.get('vmd_session'))
        response = JSONResponse({'signed_out': True})
        response.delete_cookie('vmd_session')
        return response

    @app.get('/api/centre')
    def centre_details():
        return {'centre': app.state.centre.details(), 'messaging': app.state.delivery.status(), 'calling': app.state.calling.status()}

    @app.post('/api/centre')
    def update_centre(body: CentreDetails):
        return app.state.centre.update(body)

    @app.get("/api/state")
    def state():
        return app.state.engine.snapshot()

    @app.post("/api/start")
    def start(settings: StartRequest):
        try:
            app.state.manager.connect(resolve_recording(settings), app.state.engine)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"status": "starting"}

    @app.post("/api/stop")
    def stop():
        try:
            app.state.manager.stop(app.state.engine)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        return app.state.engine.snapshot()

    @app.post("/api/scenario")
    def scenario(body: ScenarioRequest):
        if app.state.engine.snapshot().get("mode") != "demo":
            raise HTTPException(409, "Start a synthetic demo first")
        app.state.engine.scenario = body.scenario
        return {"scenario": body.scenario}

    @app.get("/api/frame/{view}")
    def frame(view: Literal["pose", "depth"], overlays: bool = True):
        values, state = app.state.engine.preview_frames(overlays)
        jpeg = values.get(view)
        meta = app.state.engine.depth_metadata(state)
        if view == "depth" and (meta["stale"] or meta["status"] == "error"):
            jpeg = None
        return Response(jpeg, media_type="image/jpeg") if jpeg else Response(status_code=204)

    @app.get("/api/frames")
    def frames(overlays: bool = True):
        # Atomic snapshot, with separate provenance for lower-rate depth.
        values, state = app.state.engine.preview_frames(overlays)
        depth_meta = app.state.engine.depth_metadata(state)
        if depth_meta["stale"] or depth_meta["status"] == "error":
            values["depth"] = None
        return {"sequence": state.get("sequence", 0), "depth_meta": depth_meta, **{key: base64.b64encode(values[key]).decode() if values.get(key) else None for key in ("pose", "depth")}}

    def camera(camera_id):
        with app.state.camera_lock:
            engine = app.state.cameras.get(camera_id)
        if engine is None:
            raise HTTPException(404, "Camera not found")
        return engine

    def frame_bundle(engine, depth=True, overlays=True):
        values, state = engine.preview_frames(overlays)
        meta = engine.depth_metadata(state)
        if meta["stale"] or meta["status"] == "error": values["depth"] = None
        objects = engine.object_metadata(state)
        available = not objects["stale"] and objects["status"] == "ready"
        if not available: values["objects"] = None
        return {"sequence": state.get("sequence", 0), "depth_meta": meta, "object_meta": objects,
                "scene_objects": state.get("scene_objects", []) if available else [],
                **{key: base64.b64encode(values[key]).decode() if values.get(key) else None for key in (("pose", "depth", "objects") if depth else ("pose",))}}

    @app.get("/api/cameras")
    def cameras():
        with app.state.camera_lock:
            engines = list(app.state.cameras.values())
        return {"cameras": [engine.snapshot() for engine in engines], "limit": 4,
                "settings_error": app.state.manager.error,
                "recording_slot_available": app.state.manager.recording_slot_available()}

    def resolve_recording(settings):
        if settings.recording_id:
            try:
                source = app.state.media.source(settings.recording_id)
            except FileNotFoundError as exc:
                raise HTTPException(404, str(exc)) from exc
            return settings.model_copy(update={'source': str(source), 'recording_id': None})
        return settings

    @app.post("/api/cameras")
    def add_camera(settings: StartRequest):
        try:
            engine = app.state.manager.connect(resolve_recording(settings))
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        return engine.snapshot()

    @app.get("/api/camera-frames")
    def camera_frames(overlays: bool = True):
        with app.state.camera_lock:
            engines = list(app.state.cameras.values())
        return {engine.camera_id: frame_bundle(engine, depth=False, overlays=overlays) for engine in engines}

    @app.get("/api/cameras/{camera_id}/frames")
    def camera_detail_frames(camera_id: str, overlays: bool = True):
        return frame_bundle(camera(camera_id), overlays=overlays)

    @app.post("/api/cameras/{camera_id}/stop")
    def stop_camera(camera_id: str):
        engine = camera(camera_id)
        try:
            app.state.manager.stop(engine)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        return engine.snapshot()

    @app.post("/api/cameras/{camera_id}/eco")
    def set_eco(camera_id: str, body: EcoRequest):
        engine = camera(camera_id)
        try:
            app.state.manager.set_eco(engine, body.enabled)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        return engine.snapshot()

    @app.post('/api/cameras/{camera_id}/location')
    def set_camera_location(camera_id: str, body: CameraLocation):
        engine = camera(camera_id)
        try:
            app.state.manager.set_location(engine, body)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        return engine.snapshot()

    @app.patch('/api/cameras/{camera_id}/settings')
    def update_camera_settings(camera_id: str, body: CameraSettingsRequest):
        engine = camera(camera_id)
        try:
            app.state.manager.update_settings(engine, body.model_dump(exclude_unset=True))
        except ValidationError as exc:
            # The merged input includes the private source; never include it in errors.
            raise HTTPException(422, exc.errors(include_input=False, include_context=False,
                                               include_url=False)) from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        return engine.snapshot()

    @app.post("/api/cameras/{camera_id}/restart")
    def restart_camera(camera_id: str):
        engine = camera(camera_id)
        if engine.settings is None:
            raise HTTPException(409, "No input settings are available for this camera")
        try:
            app.state.manager.connect(engine.settings, engine)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        return engine.snapshot()

    @app.post("/api/cameras/{camera_id}/remove")
    def remove_camera(camera_id: str):
        engine = camera(camera_id)
        try:
            app.state.manager.remove(engine)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"removed": camera_id, "evidence_preserved": True}

    @app.post("/api/cameras/{camera_id}/scenario")
    def camera_scenario(camera_id: str, body: ScenarioRequest):
        engine = camera(camera_id)
        if engine.snapshot().get("mode") != "demo":
            raise HTTPException(409, "This camera is not a synthetic demo")
        engine.scenario = body.scenario
        return {"scenario": body.scenario}

    @app.get("/api/incidents")
    def incidents():
        return app.state.store.incidents()

    @app.get('/api/evidence/status')
    def evidence_status():
        return app.state.evidence.status()

    def evidence_action(action, incident_id):
        try:
            return action(incident_id)
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(404, 'Evidence clip is unavailable') from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.get('/api/incidents/{incident_id}/share')
    def evidence_share_summary(incident_id: str):
        return evidence_action(app.state.evidence.summary, incident_id)

    @app.post('/api/incidents/{incident_id}/share')
    def create_evidence_share(incident_id: str):
        return evidence_action(app.state.evidence.create, incident_id)

    @app.post('/api/incidents/{incident_id}/event-package')
    def create_event_package(incident_id: str):
        return evidence_action(app.state.evidence.event_package, incident_id)

    @app.delete('/api/incidents/{incident_id}/share')
    def revoke_evidence_shares(incident_id: str):
        return evidence_action(app.state.evidence.revoke, incident_id)

    def ai_action(action, *args):
        try:
            return action(*args)
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from None
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from None

    @app.get('/api/ai/status')
    def ai_status():
        return app.state.ai.status()

    @app.get('/api/incidents/{incident_id}/ai')
    def ai_report(incident_id: str):
        return ai_action(app.state.ai.report, incident_id)

    @app.post('/api/incidents/{incident_id}/ai')
    def ai_analyze(incident_id: str, body: AIAnalysisRequest):
        return ai_action(app.state.ai.start, incident_id, body.quality)

    @app.post('/api/incidents/{incident_id}/ai/ask')
    def ai_question(incident_id: str, body: AIQuestionRequest):
        return ai_action(app.state.ai.ask, incident_id, body.question)

    @app.post('/api/incidents/{incident_id}/ai/approve')
    def ai_approve(incident_id: str, body: AIToggleRequest):
        return ai_action(app.state.ai.approve, incident_id, body.enabled)

    @app.post('/api/incidents/{incident_id}/ai/follow')
    def ai_follow(incident_id: str, body: AIToggleRequest):
        return ai_action(app.state.ai.follow, incident_id, body.enabled)

    @app.get('/api/incidents/{incident_id}/ai/report')
    def download_ai_report(incident_id: str):
        data = ai_action(app.state.ai.report, incident_id)
        return Response(json.dumps(data, indent=2), media_type='application/json',
            headers={'Content-Disposition': 'attachment; filename="incident-ai-report.json"'})

    @app.get('/api/incidents/{incident_id}/ai/report.pdf')
    def download_ai_pdf(incident_id: str, request: Request):
        content = ai_action(app.state.ai.pdf, incident_id, str(request.base_url).rstrip('/'))
        return Response(content, media_type='application/pdf', headers={
            'Content-Disposition': 'attachment; filename="incident-ai-report.pdf"'})

    @app.get('/api/response-alerts')
    def response_alerts():
        result = app.state.alerts.snapshot()
        result['messaging'] = app.state.delivery.status()
        result['calling'] = app.state.calling.status()
        for alert in result['alerts']:
            alert['deliveries'] = app.state.delivery.records(alert['id'])
            alert['calls'] = app.state.calling.records(alert['id'])
            alert['call_issues'] = []
            if not result['calling']['configured']:
                alert['call_issues'].append('Twilio calling credentials are not configured')
            elif not result['calling']['enabled']:
                alert['call_issues'].append('Voice calls are disabled in Centre settings')
            if result['calling']['error']:
                alert['call_issues'].append(result['calling']['error'])
            details = app.state.centre.details()
            targets = json.loads(alert['voice_contacts']) if alert.get('voice_contacts') else (
                [('authority', item) for item in details['authorities']] + [('hospital', details['hospital'])])
            expected = {item['phone'] for kind, item in targets if kind != 'hospital' or alert['hospital']}
            alert['calls_submitted'] = bool(expected) and expected <= {
                item['phone'] for item in alert['calls'] if item['status'] in {'queued', 'ringing', 'in-progress', 'completed'}}
            if not result['messaging']['configured']:
                alert['delivery_issues'].append('Twilio credentials and public evidence URL are not configured; nothing has been sent')
            elif not result['messaging']['enabled']:
                alert['delivery_issues'].append('SMS dispatch is disabled in Centre settings')
        return result

    @app.get('/api/incidents/{incident_id}/response')
    def incident_response(incident_id: str):
        alert = app.state.alerts.get(incident_id)
        return app.state.alerts.describe(alert) if alert else None

    @app.post('/api/incidents/{incident_id}/response')
    def response_action(incident_id: str, body: ResponseRequest):
        if body.action in {'dispatch', 'call_hospital'}:
            details = app.state.centre.details()
            if not details.get('calling_enabled') and (body.action == 'call_hospital' or not details.get('messaging_enabled')):
                raise HTTPException(409, 'Voice calls are off. Enable Twilio voice calls in Centre settings and save, then dispatch again.')
        try:
            return app.state.alerts.act(incident_id, body.action, body.hospital)
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get('/api/incidents/{incident_id}/response-package')
    def response_package(incident_id: str, request: Request):
        try:
            package = app.state.alerts.package(incident_id)
            path = app.state.media.safe_path(app.state.store.clips, package['clip'])
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from exc
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(409, str(exc)) from exc
        output = io.BytesIO()
        with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_STORED) as archive:
            archive.writestr('incident.json', json.dumps(package, indent=2))
            try:
                ai = app.state.ai.report(incident_id)
                if ai['status'] == 'ready':
                    archive.writestr('ai-report.json', json.dumps(ai, indent=2))
                    archive.writestr('ai-report.pdf', app.state.ai.pdf(incident_id, str(request.base_url).rstrip('/')))
            except (LookupError, ValueError):
                pass
            archive.write(path, arcname=path.name)
        return Response(output.getvalue(), media_type='application/zip',
                        headers={'Content-Disposition': 'attachment; filename="incident-response.zip"'})

    @app.post("/api/incidents/{incident_id}/review")
    def review(incident_id: str, body: ReviewRequest):
        if not app.state.alerts.review(incident_id, body.decision):
            raise HTTPException(404, "Incident not found")
        return {"review": body.decision}

    @app.get("/api/incidents/{incident_id}/clip")
    def clip(incident_id: str):
        event = app.state.store.incident(incident_id)
        if not event or not event["clip"]:
            raise HTTPException(404, "Clip not available")
        try:
            path = app.state.media.safe_path(app.state.store.clips, event['clip'])
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        return FileResponse(path, media_type="video/x-msvideo", filename=event["clip"])

    def playable(path):
        try:
            result = app.state.media.playback(path)
        except FileNotFoundError as exc:
            raise HTTPException(404, 'Video is unavailable') from exc
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc
        return FileResponse(result, media_type='video/mp4')

    @app.get('/api/incidents/{incident_id}/play')
    def incident_playback(incident_id: str):
        event = app.state.store.incident(incident_id)
        if not event or not event['clip']:
            raise HTTPException(404, 'Clip not available')
        try:
            source = app.state.media.safe_path(app.state.store.clips, event['clip'])
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        return playable(source)

    @app.get('/api/recordings')
    def recordings():
        return app.state.media.recordings()

    @app.post('/api/recordings')
    async def upload_recording(request: Request):
        name = unquote(request.headers.get('x-file-name', '')).replace('\\', '/').split('/')[-1][:200]
        suffix = Path(name).suffix.lower()
        if suffix not in VIDEO_EXTENSIONS:
            raise HTTPException(422, 'Choose an MP4, AVI, MOV, MKV, WebM, or M4V video.')
        try:
            declared = int(request.headers.get('content-length', 0))
        except ValueError:
            raise HTTPException(400, 'Invalid upload size') from None
        if declared > MAX_UPLOAD:
            raise HTTPException(413, 'Videos must be 1 GB or smaller.')
        recording_id = uuid.uuid4().hex
        temporary = app.state.media.directory / (recording_id + '.upload' + suffix)
        size = 0
        try:
            fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(fd, 'wb') as stream:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > MAX_UPLOAD:
                        raise HTTPException(413, 'Videos must be 1 GB or smaller.')
                    stream.write(chunk)
            return await run_in_threadpool(app.state.media.register, temporary, name, recording_id, suffix)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        finally:
            temporary.unlink(missing_ok=True)

    @app.get('/api/recordings/{recording_id}/{action}')
    def recording_file(recording_id: str, action: Literal['play', 'download']):
        try:
            source = app.state.media.source(recording_id)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        return playable(source) if action == 'play' else FileResponse(source, filename=source.name)

    @app.get("/api/analytics")
    def analytics():
        return app.state.store.analytics(time.time()-86400)

    @app.get('/api/analytics/summary')
    def analytics_summary(days: Literal['7', '30'] = '7'):
        with app.state.camera_lock:
            names = {engine.camera_id: engine.name for engine in app.state.cameras.values()}
        return app.state.store.analytics_summary(int(days), camera_names=names)

    static = Path(__file__).parent / "static"
    app.mount("/", StaticFiles(directory=static, html=True), name="dashboard")
    return app


app = create_app()
