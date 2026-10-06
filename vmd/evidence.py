"""Prepare expiring evidence links without requesting any incident response."""
import hashlib
import json
import os
import secrets
import sqlite3
import subprocess
import time
from urllib.parse import urlsplit

from .incident_payload import event_package_record


# Shared by link summaries and the public gateway. JSON booleans distinguish
# saved real inputs (including recordings) from synthetic/unknown provenance.
SHARE_ELIGIBLE_SQL = """i.mode='live'
    AND CASE WHEN json_valid(i.signals) THEN json_type(i.signals, '$.live_camera') END IN ('true','false')
    AND COALESCE(json_extract(CASE WHEN json_valid(i.signals) THEN i.signals ELSE '{}' END, '$.presentation'), 0)=0
    AND i.review!='false_positive' AND i.clip IS NOT NULL
    AND (a.id IS NULL OR a.state!='cancelled')"""
_MISSING = object()


def public_origin(value):
    if not isinstance(value, str) or not value or any(c.isspace() or ord(c) < 32 for c in value):
        return None
    try:
        url = urlsplit(value)
        if (url.scheme != 'https' or not url.hostname or url.username is not None
                or url.password is not None or url.query or url.fragment
                or url.path not in {'', '/'} or '\\' in value):
            return None
        url.port  # Reject malformed or out-of-range ports too.
    except ValueError:
        return None
    return value.rstrip('/')


def _alive(pid, expected_birth=_MISSING):
    if type(pid) is not int or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    if expected_birth is not _MISSING:
        if not isinstance(expected_birth, str) or not expected_birth:
            return False
        try:
            result = subprocess.run(['ps', '-p', str(pid), '-o', 'stat=', '-o', 'lstart='],
                                    capture_output=True, text=True, timeout=2)
        except (OSError, subprocess.SubprocessError):
            return False
        fields = result.stdout.strip().split(maxsplit=1)
        return (result.returncode == 0 and len(fields) == 2 and 'Z' not in fields[0]
                and fields[1] == expected_birth)
    return True


def configuration(directory):
    configured = os.getenv('VMD_EVIDENCE_BASE_URL')
    if configured is not None:
        base = public_origin(configured)
        message = ('Evidence hosting is configured. Links expire after one hour.' if base
                   else 'Set VMD_EVIDENCE_BASE_URL to a valid public HTTPS origin.')
    else:
        try:
            host = json.loads((directory / 'evidence-host.json').read_text())
        except (OSError, ValueError):
            host = {}
        if not isinstance(host, dict):
            host = {}
        base = public_origin(host.get('base_url'))
        if not base or not all(_alive(host.get(kind + '_pid'), host.get(kind + '_birth', _MISSING))
                               for kind in ('gateway', 'tunnel')):
            base = None
        message = ('Evidence hosting processes are running. Keep this laptop and its tunnel online.' if base
                   else 'Start evidence hosting to prepare a public video link.')
    return {'configured': bool(base), 'base_url': base, 'message': message}


class EvidenceSharing:
    def __init__(self, store, media):
        self.store, self.media = store, media
        with store.connect() as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS evidence_shares (
                token_hash TEXT PRIMARY KEY, incident_id TEXT NOT NULL,
                filename TEXT NOT NULL, expires REAL NOT NULL)''')

    def status(self):
        return configuration(self.store.directory)

    @staticmethod
    def _incident(conn, incident_id, *, eligible=False):
        conn.row_factory = sqlite3.Row
        row = conn.execute('''SELECT i.*, a.state AS response_state,
            (''' + SHARE_ELIGIBLE_SQL + ''') AS share_eligible FROM incidents i
            LEFT JOIN response_alerts a ON a.id=i.id WHERE i.id=?''', (incident_id,)).fetchone()
        if row is None:
            raise LookupError('Incident not found')
        if eligible and not row['share_eligible']:
            raise ValueError('Only saved real-source incidents with evidence, not false positives or cancelled responses, can be shared.')
        return row

    def summary(self, incident_id):
        with self.store.connect() as conn:
            event = self._incident(conn, incident_id)
            row = conn.execute('SELECT COUNT(*), MAX(expires) FROM evidence_shares '
                               'WHERE incident_id=? AND expires>?', (incident_id, time.time())).fetchone()
        return {'active_count': row[0] if event['share_eligible'] else 0,
                'latest_expires_at': row[1] if event['share_eligible'] else None}

    def create(self, incident_id):
        return self._create(incident_id)

    def event_package(self, incident_id):
        return self._create(incident_id, include_event=True)

    def _create(self, incident_id, *, include_event=False):
        with self.store.connect() as conn:
            event = self._incident(conn, incident_id, eligible=True)
        if not self.status()['configured']:
            raise RuntimeError('Evidence hosting is not running or configured. Start it and try again.')
        source = self.media.safe_path(self.store.clips, event['clip'])
        converted = self.media.playback(source)
        # Only the designated playback cache can become public, even if a
        # converter or an existing cache entry resolves through a symlink.
        clip = self.media.safe_path(self.media.cache, converted.name)
        if clip != converted.resolve():
            raise FileNotFoundError('Prepared evidence is unavailable')
        token = secrets.token_urlsafe(32)
        with self.store.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            current = self._incident(conn, incident_id, eligible=True)
            if current['clip'] != event['clip']:
                raise ValueError('Evidence changed during preparation. Try again for the latest clip.')
            # Encoding can take time: review/cancellation and managed hosting
            # must still permit the link when it is actually issued.
            status = self.status()
            if not status['configured']:
                raise RuntimeError('Evidence hosting stopped during preparation. Start it and try again.')
            expires = time.time() + 3600
            share = {'url': status['base_url'] + '/e/' + token,
                     'expires_at': expires, 'incident_id': incident_id}
            # Build the allowlisted event from the final row before committing
            # its link, so conversion cannot leave metadata stale or orphan a token.
            result = event_package_record(current, share) if include_event else share
            conn.execute('INSERT INTO evidence_shares VALUES (?,?,?,?)',
                         (hashlib.sha256(token.encode()).hexdigest(), incident_id, clip.name, expires))
        return result

    def revoke(self, incident_id):
        with self.store.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            self._incident(conn, incident_id)
            count = conn.execute('DELETE FROM evidence_shares WHERE incident_id=?', (incident_id,)).rowcount
        return {'revoked_count': count}
