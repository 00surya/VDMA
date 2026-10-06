"""Incident metadata gateway: read-only SQLite, independent of camera/dispatch services."""
import base64
import hashlib
import hmac
import ipaddress
import json
import math
import os
import re
import secrets
import sqlite3
import threading
import time
from contextlib import closing, contextmanager
from datetime import datetime
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .centre import Centre, CentreLogin
from .incident_payload import MAX_TIMESTAMP, incident_record, number, utc


TOKEN_SECONDS = 3600
IDENTITY = re.compile(r'^[A-Za-z0-9_-]{1,128}$')


def decode_cursor(value):
    try:
        payload = base64.b64decode(value + '=' * (-len(value) % 4), altchars=b'-_', validate=True)
        created, identity = json.loads(payload)
        if not number(created) or not isinstance(identity, str) or not IDENTITY.fullmatch(identity):
            raise ValueError
        return created, identity
    except (ValueError, TypeError, UnicodeError):
        raise HTTPException(422, 'Invalid pagination cursor') from None


class Access:
    """Bounded, process-local credentials and rate counters; restart revokes all tokens."""
    def __init__(self):
        self.lock = threading.Lock()
        self.tokens = {}
        self.windows = {}

    def limit(self, key, maximum):
        now = time.monotonic()
        start, count = self.windows.get(key, (now, 0))
        if now - start >= 60:
            start, count = now, 0
        if count >= maximum:
            raise HTTPException(429, 'Too many requests', headers={'Retry-After': str(max(1, math.ceil(60 - (now - start))))})
        self.windows[key] = (start, count + 1)

    def admit(self, key, maximum):
        with self.lock:
            self.limit(key, maximum)

    def issue(self, credentials, password):
        with self.lock:
            if not hmac.compare_digest(Centre.digest(password, credentials['salt']), credentials['password_hash']):
                raise HTTPException(401, 'Invalid credentials', headers={'WWW-Authenticate': 'Bearer'})
            now = time.monotonic()
            for digest, session in list(self.tokens.items()):
                if session['expires'] <= now:
                    self.tokens.pop(digest)
                    self.windows.pop(digest, None)
            if len(self.tokens) >= 128:
                raise HTTPException(503, 'Session capacity reached; sign out an existing device')
            token = secrets.token_urlsafe(32)
            self.tokens[hashlib.sha256(token.encode()).hexdigest()] = {
                'expires': now + TOKEN_SECONDS, 'password_hash': credentials['password_hash'],
            }
        return {'access_token': token, 'token_type': 'bearer', 'expires_in': TOKEN_SECONDS,
                'expires_at': utc(time.time() + TOKEN_SECONDS), 'scope': 'incidents:read'}

    def authenticate(self, header, credentials):
        scheme, _, token = (header or '').partition(' ')
        if scheme.lower() != 'bearer' or not re.fullmatch(r'[A-Za-z0-9_-]{43}', token):
            raise HTTPException(401, 'Bearer token required', headers={'WWW-Authenticate': 'Bearer'})
        digest = hashlib.sha256(token.encode()).hexdigest()
        with self.lock:
            session = self.tokens.get(digest)
            if (not session or session['expires'] <= time.monotonic()
                    or not hmac.compare_digest(session['password_hash'], credentials['password_hash'])):
                self.tokens.pop(digest, None)
                self.windows.pop(digest, None)
                raise HTTPException(401, 'Token expired or invalid', headers={'WWW-Authenticate': 'Bearer'})
            self.limit(digest, 120)
        return digest

    def revoke(self, digest):
        with self.lock:
            self.tokens.pop(digest, None)
            self.windows.pop(digest, None)


def create_mobile_app(data_dir=None, *, allowed_hosts=None):
    app = FastAPI(title='VMD incident metadata API', docs_url=None, redoc_url=None, openapi_url=None)
    database = Path(data_dir or os.getenv('VMD_DATA_DIR', 'data')).resolve() / 'telemetry.sqlite3'
    hosts = allowed_hosts if allowed_hosts is not None else os.getenv('VMD_MOBILE_HOSTS', '127.0.0.1,localhost,[::1]').split(',')
    hosts = [host.strip() for host in hosts if host.strip()]
    if not hosts or any('*' in host or '/' in host for host in hosts):
        raise ValueError('VMD_MOBILE_HOSTS must list exact hostnames, without schemes or wildcards')
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=hosts)
    access = Access()

    @contextmanager
    def connect():
        try:
            with closing(sqlite3.connect(database.as_uri() + '?mode=ro', uri=True, timeout=2)) as conn:
                conn.row_factory = sqlite3.Row
                yield conn
        except sqlite3.Error:
            raise HTTPException(503, 'Incident data unavailable') from None

    def credentials():
        with connect() as conn:
            row = conn.execute('SELECT salt,password_hash FROM centre WHERE id=1').fetchone()
        if not row:
            raise HTTPException(503, 'Register this installation in the local dashboard first')
        return row

    @app.middleware('http')
    async def protect(request, call_next):
        try:
            try:
                loopback = request.client is not None and ipaddress.ip_address(request.client.host).is_loopback
            except ValueError:
                loopback = False
            if request.url.scheme != 'https' and not loopback:
                raise HTTPException(403, 'HTTPS required')
            if request.headers.get('origin') not in (None, f'{request.url.scheme}://{request.url.netloc}'):
                raise HTTPException(403, 'Cross-origin access is not allowed')
            access.admit('all', 600)
            response = await call_next(request)
        except HTTPException as exc:
            response = JSONResponse({'detail': exc.detail}, status_code=exc.status_code, headers=exc.headers)
        response.headers.update({'Cache-Control': 'no-store', 'Pragma': 'no-cache',
                                 'X-Content-Type-Options': 'nosniff', 'Referrer-Policy': 'no-referrer'})
        return response

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        # FastAPI's default validation response echoes input; never reflect passwords/tokens.
        return JSONResponse({'detail': 'Invalid request parameters'}, status_code=422)

    def authenticated(request: Request):
        return access.authenticate(request.headers.get('authorization'), credentials())

    @app.post('/api/v1/auth/token')
    async def token(request: Request):
        access.admit('login', 5)
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > 4096:
                raise HTTPException(413, 'Login request too large')
        try:
            login = CentreLogin.model_validate_json(body)
        except (ValidationError, ValueError):
            raise HTTPException(422, 'A password is required') from None
        return await run_in_threadpool(access.issue, await run_in_threadpool(credentials), login.password)

    @app.post('/api/v1/auth/logout', status_code=204)
    def logout(digest=Depends(authenticated)):
        access.revoke(digest)
        return Response(status_code=204)

    # Guard every JSON operation; credentials grant this one installation only.
    base_query = '''SELECT id,created,event_type,camera_id,camera_name,review,signals FROM incidents
        WHERE mode='live' AND created>=0 AND created<=?
        AND CASE WHEN json_valid(signals) THEN json_type(signals, '$.live_camera') END IN ('true','false')
        AND COALESCE(json_extract(CASE WHEN json_valid(signals) THEN signals ELSE '{}' END, '$.presentation'), 0)=0'''

    @app.get('/api/v1/incidents', dependencies=[Depends(authenticated)])
    def incidents(limit: int = Query(default=50, ge=1, le=100),
                  cursor: str | None = Query(default=None, max_length=512),
                  since: datetime | None = None,
                  source: Literal['live_camera', 'recording', 'all'] = 'live_camera',
                  include_false_positives: bool = False):
        query, args = base_query, [min(time.time(), MAX_TIMESTAMP)]
        if since is not None:
            if since.tzinfo is None or not number(since.timestamp()):
                raise HTTPException(422, 'since must be a timestamp with a timezone, at or after 1970')
            query += ' AND created>=?'
            args.append(since.timestamp())
        if source != 'all':
            query += " AND json_type(CASE WHEN json_valid(signals) THEN signals ELSE '{}' END, '$.live_camera')=?"
            args.append('true' if source == 'live_camera' else 'false')
        if not include_false_positives:
            query += " AND review!='false_positive'"
        if cursor:
            created, identity = decode_cursor(cursor)
            query += ' AND (created<? OR (created=? AND id<?))'
            args.extend((created, created, identity))
        query += ' ORDER BY created DESC,id DESC LIMIT ?'
        args.append(limit + 1)
        with connect() as conn:
            rows = conn.execute(query, args).fetchall()
        next_cursor = None
        if len(rows) > limit:
            last = rows[limit - 1]
            next_cursor = base64.urlsafe_b64encode(json.dumps([last['created'], last['id']]).encode()).decode().rstrip('=')
        return {'incidents': [incident_record(row) for row in rows[:limit]],
                'next_cursor': next_cursor, 'server_time': utc(time.time())}

    @app.get('/api/v1/incidents/{incident_id}', dependencies=[Depends(authenticated)])
    def incident(incident_id: str):
        if not IDENTITY.fullmatch(incident_id):
            raise HTTPException(404, 'Incident not found')
        with connect() as conn:
            row = conn.execute(base_query + ' AND id=?', (min(time.time(), MAX_TIMESTAMP), incident_id)).fetchone()
        if not row:
            raise HTTPException(404, 'Incident not found')
        # Return current review state even if it became a false positive after being fetched.
        return incident_record(row)

    return app


app = create_mobile_app()
