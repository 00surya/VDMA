"""Standalone local camera-description lab; run only from vision-lab/.venv."""
from contextlib import asynccontextmanager
import copy
import multiprocessing as mp
from pathlib import Path
import queue
import threading
import time
from urllib.parse import urlsplit, urlunsplit
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field, field_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware

from workers import MODEL_ID, capture_worker, model_worker

ROOT = Path(__file__).resolve().parent


def camera_url(value):
    value = value.strip()
    if not value or len(value) > 2048 or any(c.isspace() or ord(c) < 32 for c in value):
        raise ValueError('Enter a camera IP address or HTTP/RTSP video URL without spaces.')
    if '://' not in value:
        value = 'http://' + value
    try:
        parts = urlsplit(value)
        if parts.scheme not in ('http', 'https', 'rtsp', 'rtsps') or not parts.hostname or parts.fragment:
            raise ValueError()
        if parts.port is not None and not 1 <= parts.port <= 65535:
            raise ValueError()
    except ValueError:
        raise ValueError('Use an HTTP(S) or RTSP(S) camera video URL with a valid host and port.') from None
    # Android IP Webcam root addresses are control pages; its stream is /video.
    if parts.scheme in ('http', 'https') and parts.path in ('', '/') and not parts.query:
        parts = parts._replace(path='/video')
    return urlunsplit(parts)


class CameraInput(BaseModel):
    url: str = Field(min_length=1, max_length=2048)
    name: str = Field(default='IP camera', min_length=1, max_length=80)
    interval: float = Field(default=5, ge=2, le=60, allow_inf_nan=False)

    @field_validator('url')
    @classmethod
    def valid_url(cls, value):
        return camera_url(value)


class Settings(BaseModel):
    interval: float = Field(ge=2, le=60, allow_inf_nan=False)


class Lab:
    def __init__(self, *, background=True):
        self.lock = threading.RLock()
        self.actions = threading.Lock()
        self.closed = threading.Event()
        self.ctx = mp.get_context('spawn')
        self.camera_process = self.model_process = None
        self.camera_stop = self.model_stop = None
        self.camera_output = self.model_output = self.requests = None
        self.jpeg = None
        self.job_started = 0
        self.next_analysis = 0
        self.last_sequence = 0
        self.model_started = 0
        self.state = self.initial_state()
        self.worker = None
        if background:
            self.worker = threading.Thread(target=self.work, daemon=True, name='vision-lab-coordinator')
            self.worker.start()

    @staticmethod
    def initial_state():
        return {'running': False, 'interval': 5,
            'camera': {'status': 'idle', 'name': 'IP camera', 'error': None,
                'frame_sequence': 0, 'frame_at': None, 'width': None, 'height': None},
            'model': {'status': 'idle', 'name': MODEL_ID, 'error': None,
                'peak_memory_mb': None, 'process_memory_mb': None},
            'analysis': {'status': 'idle', 'error': None}, 'observations': []}

    def snapshot(self):
        with self.lock:
            return {**copy.deepcopy(self.state), 'server_time': time.time()}

    def start(self, settings):
        with self.actions:
            self.stop_locked()
            with self.lock:
                self.state = self.initial_state()
                self.state.update(running=True, interval=settings.interval)
                self.state['camera'].update(status='connecting', name=settings.name)
                self.state['analysis']['status'] = 'waiting'
                self.camera_output = self.ctx.Queue(maxsize=1)
                self.camera_stop = self.ctx.Event()
                self.camera_process = self.ctx.Process(target=capture_worker,
                    args=(settings.url, self.camera_output, self.camera_stop), daemon=True)
                try:
                    self.camera_process.start()
                except Exception:
                    self.state['running'] = False
                    self.state['camera'].update(status='error', error='Cannot start the camera process. Try reconnecting.')
                self.last_sequence = 0
                self.next_analysis = 0
        return self.snapshot()

    @staticmethod
    def halt(process, stop):
        if process:
            stop.set()
            if process.pid is None:
                process.close()
                return
            process.join(timeout=.3)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=1)
            process.close()

    @classmethod
    def reap(cls, process, stop, channels):
        cls.halt(process, stop)
        for channel in channels:
            if channel:
                channel.cancel_join_thread()
                channel.close()

    def stop_locked(self):
        # Detach old queues under the lock so late results cannot enter a new session.
        with self.lock:
            processes = [(self.camera_process, self.camera_stop), (self.model_process, self.model_stop)]
            channels = [self.camera_output, self.model_output, self.requests]
            self.camera_process = self.model_process = None
            self.camera_output = self.model_output = self.requests = None
            self.state['running'] = False
            self.state['camera'].update(status='stopped', frame_at=None, frame_sequence=0, error=None)
            self.state['model'].update(status='idle', error=None, peak_memory_mb=None, process_memory_mb=None)
            self.state['analysis'].update(status='idle', error=None)
            self.jpeg = None
            self.job_started = 0
        for process, stop in processes:
            self.halt(process, stop)
        for channel in channels:
            if channel:
                channel.cancel_join_thread()
                channel.close()

    def stop(self):
        with self.actions:
            self.stop_locked()
        return self.snapshot()

    def analyze(self):
        with self.lock:
            camera = self.state['camera']
            if not self.state['running'] or not self.jpeg or time.time() - camera['frame_at'] > 3:
                raise ValueError('Wait for fresh video before requesting a description.')
            if self.state['model']['status'] == 'error':
                raise ValueError('Stop and reconnect to reload the model.')
            self.next_analysis = 0
            self.last_sequence = 0
        return self.snapshot()

    def settings(self, interval):
        with self.lock:
            self.state['interval'] = interval
            self.next_analysis = min(self.next_analysis, time.monotonic() + interval)
        return self.snapshot()

    def tick(self):
        with self.lock:
            if not self.state['running']:
                return
            camera, model = self.state['camera'], self.state['model']
            if self.camera_output:
                try:
                    update = self.camera_output.get_nowait()
                except queue.Empty:
                    pass
                else:
                    if 'jpeg' in update:
                        self.jpeg = update.pop('jpeg')
                    camera.update(update)
            if self.camera_process and not self.camera_process.is_alive():
                camera.update(status='error', error='Camera decoder stopped. Stop and reconnect the camera.')
            if camera['frame_at'] and time.time() - camera['frame_at'] > 8 and camera['status'] == 'streaming':
                camera.update(status='reconnecting', error='Video is stale. Waiting for the camera to reconnect.')
            if self.model_output:
                try:
                    update = self.model_output.get_nowait()
                except queue.Empty:
                    pass
                else:
                    observation = update.pop('observation', None)
                    model.update(update)
                    if observation:
                        self.state['observations'].insert(0, observation)
                        del self.state['observations'][30:]
                        self.job_started = 0
                        self.next_analysis = time.monotonic() + self.state['interval']
                        self.state['analysis'].update(status='waiting', error=None)
                    elif update['status'] == 'error':
                        self.job_started = 0
                        self.state['analysis'].update(status='idle', error=update['error'])
            if self.model_process and not self.model_process.is_alive() and model['status'] != 'error':
                model.update(status='error', error='Model process stopped. Stop and reconnect to retry.')
                self.state['analysis'].update(status='idle', error=model['error'])
            if self.model_process and model['status'] in ('loading', 'analyzing'):
                started = self.job_started or self.model_started
                if time.monotonic() - started > 120:
                    # The bounded reap runs separately so HTTP state stays responsive.
                    expired = self.model_process
                    channels = [self.model_output, self.requests]
                    self.model_process = None
                    self.model_output = self.requests = None
                    threading.Thread(target=self.reap,
                        args=(expired, self.model_stop, channels), daemon=True).start()
                    model.update(status='error', error='Model exceeded its two-minute limit. Stop and reconnect.')
                    self.state['analysis'].update(status='idle', error=model['error'])
            fresh = camera['status'] == 'streaming' and camera['frame_at'] and time.time() - camera['frame_at'] < 3
            if not fresh or not self.jpeg:
                return
            if not self.model_process and model['status'] != 'error':
                self.requests = self.ctx.Queue(maxsize=1)
                self.model_output = self.ctx.Queue(maxsize=4)
                self.model_stop = self.ctx.Event()
                self.model_process = self.ctx.Process(target=model_worker,
                    args=(self.requests, self.model_output, self.model_stop), daemon=True)
                model.update(status='loading', error=None)
                self.model_started = time.monotonic()
                try:
                    self.model_process.start()
                except Exception:
                    model.update(status='error', error='Cannot start the model process. Stop and reconnect.')
            if model['status'] == 'ready' and not self.job_started and time.monotonic() >= self.next_analysis:
                if camera['frame_sequence'] == self.last_sequence:
                    return
                sample = {'id': uuid.uuid4().hex, 'frame_at': camera['frame_at'],
                    'frame_sequence': camera['frame_sequence'], 'jpeg': self.jpeg}
                try:
                    self.requests.put_nowait(sample)
                except queue.Full:
                    return
                self.job_started = time.monotonic()
                self.last_sequence = camera['frame_sequence']
                model['status'] = 'analyzing'
                self.state['analysis'].update(status='analyzing', error=None)

    def work(self):
        while not self.closed.wait(.1):
            try:
                self.tick()
            except Exception as exc:
                with self.lock:
                    self.state['analysis']['error'] = f'Coordinator error ({type(exc).__name__}); stop and reconnect.'

    def close(self):
        self.closed.set()
        if self.worker:
            self.worker.join(timeout=2)
        self.stop()


def create_app(lab_factory=Lab):
    @asynccontextmanager
    async def lifespan(app):
        app.state.lab = lab_factory()
        yield
        app.state.lab.close()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=['127.0.0.1', 'localhost', '[::1]', 'testserver'])

    @app.middleware('http')
    async def local_only(request: Request, call_next):
        origin = request.headers.get('origin')
        if origin and origin != f'{request.url.scheme}://{request.headers.get("host")}':
            return JSONResponse({'detail': 'Only the local lab page can access this API.'}, status_code=403)
        if request.method not in ('GET', 'HEAD') and request.headers.get('x-vision-lab') != 'dashboard':
            return JSONResponse({'detail': 'Use the local lab UI.'}, status_code=403)
        response = await call_next(request)
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'no-referrer'
        response.headers['Content-Security-Policy'] = (
            "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
        return response

    # Pydantic's default errors echo submitted input, potentially including URL credentials.
    from fastapi.exceptions import RequestValidationError
    @app.exception_handler(RequestValidationError)
    async def invalid(request, exc):
        return JSONResponse({'detail': 'Check the camera URL, name, and interval (2–60 seconds).'}, status_code=422)

    @app.get('/')
    def index():
        return FileResponse(ROOT / 'static' / 'index.html')

    @app.get('/api/state')
    def state(request: Request):
        return request.app.state.lab.snapshot()

    @app.get('/api/frame.jpg')
    def frame(request: Request):
        lab = request.app.state.lab
        with lab.lock:
            jpeg = lab.jpeg
        if not jpeg:
            raise HTTPException(404, 'No camera frame yet.')
        return Response(jpeg, media_type='image/jpeg')

    @app.post('/api/camera')
    def connect(settings: CameraInput, request: Request):
        return request.app.state.lab.start(settings)

    @app.post('/api/stop')
    def stop(request: Request):
        return request.app.state.lab.stop()

    @app.post('/api/analyze')
    def analyze(request: Request):
        try:
            return request.app.state.lab.analyze()
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None

    @app.post('/api/settings')
    def settings(settings: Settings, request: Request):
        return request.app.state.lab.settings(settings.interval)

    return app


app = create_app()

if __name__ == '__main__':
    import faulthandler
    import uvicorn
    faulthandler.enable()
    uvicorn.run(app, host='127.0.0.1', port=8770, access_log=False)
