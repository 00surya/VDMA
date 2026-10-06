"""Evidence-only gateway. Expose this port through HTTPS, never the dashboard."""
import hashlib
import os
import re
import sqlite3
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from .evidence import SHARE_ELIGIBLE_SQL


def create_share_app(data_dir=None):
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    directory = Path(data_dir or os.getenv('VMD_DATA_DIR', 'data'))

    @app.middleware('http')
    async def private_headers(request, call_next):
        response = await call_next(request)
        response.headers['Cache-Control'] = 'no-store'
        response.headers['Referrer-Policy'] = 'no-referrer'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response

    @app.get('/health')
    def health():
        return {'status': 'ok', 'service': 'vdma-evidence'}

    @app.api_route('/e/{token}', methods=['GET', 'HEAD'])
    def evidence(token: str):
        if not re.fullmatch(r'[A-Za-z0-9_-]{43}', token):
            raise HTTPException(404, 'Evidence link unavailable or expired')
        try:
            database_uri = (directory / 'telemetry.sqlite3').resolve().as_uri() + '?mode=ro'
            with sqlite3.connect(database_uri, uri=True) as conn:
                row = conn.execute('''SELECT s.filename FROM evidence_shares s
                    JOIN incidents i ON i.id=s.incident_id
                    LEFT JOIN response_alerts a ON a.id=s.incident_id
                    WHERE s.token_hash=? AND s.expires>? AND ''' + SHARE_ELIGIBLE_SQL,
                    (hashlib.sha256(token.encode()).hexdigest(), time.time())).fetchone()
        except sqlite3.Error:
            row = None
        if not row:
            raise HTTPException(404, 'Evidence link unavailable or expired')
        cache = (directory / 'playback').resolve()
        path = (cache / row[0]).resolve()
        if not path.is_relative_to(cache) or not path.is_file():
            raise HTTPException(404, 'Evidence unavailable')
        return FileResponse(path, media_type='video/mp4', headers={
            'Cache-Control': 'no-store', 'Referrer-Policy': 'no-referrer',
            'X-Content-Type-Options': 'nosniff', 'Content-Disposition': 'inline; filename="incident.mp4"'})
    return app


app = create_share_app()
