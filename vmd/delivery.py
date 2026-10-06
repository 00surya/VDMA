"""Twilio SMS delivery with expiring evidence links and one attempt per recipient."""
import base64
import hashlib
import json
import os
import re
import secrets
import sqlite3
import threading
import time
from urllib.error import HTTPError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, HTTPRedirectHandler, build_opener


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def configuration():
    sid, token = os.getenv('TWILIO_ACCOUNT_SID', ''), os.getenv('TWILIO_AUTH_TOKEN', '')
    sender, base = os.getenv('TWILIO_FROM_NUMBER', ''), os.getenv('VMD_EVIDENCE_BASE_URL', '').rstrip('/')
    try:
        url = urlsplit(base)
    except ValueError:
        url = urlsplit('')
    ready = bool(re.fullmatch(r'AC[0-9a-fA-F]{32}', sid) and token
                 and re.fullmatch(r'\+[1-9][0-9]{7,14}', sender)
                 and url.scheme == 'https' and url.hostname and not url.username
                 and not url.password and not url.query and not url.fragment and url.path in {'', '/'})
    return {'sid': sid, 'token': token, 'sender': sender, 'base': base, 'ready': ready}


def send_sms(phone, body):
    config = configuration()
    if not config['ready']:
        raise ValueError('Twilio and public evidence URL are not configured')
    auth = base64.b64encode(f"{config['sid']}:{config['token']}".encode()).decode()
    request = Request(f"https://api.twilio.com/2010-04-01/Accounts/{config['sid']}/Messages.json",
                      data=urlencode({'To': phone, 'From': config['sender'], 'Body': body}).encode(),
                      headers={'Authorization': f'Basic {auth}', 'Content-Type': 'application/x-www-form-urlencoded'})
    try:
        with build_opener(NoRedirect).open(request, timeout=8) as response:
            value = json.loads(response.read(65536))
    except HTTPError as exc:
        # Do not echo Twilio's response, which can contain phone numbers or credentials.
        return {'status': 'failed', 'sid': None, 'error': f'Twilio rejected the message (HTTP {exc.code})'}
    if not re.fullmatch(r'SM[0-9a-fA-F]{32}', value.get('sid', '')):
        raise ValueError('Twilio response did not include a message ID')
    if value.get('status') in {'failed', 'undelivered', 'canceled'}:
        return {'status': 'failed', 'sid': value['sid'], 'error': 'Twilio reported the message was not delivered'}
    return {'status': 'accepted', 'sid': value['sid'], 'error': None}


class Delivery:
    def __init__(self, store, centre, alerts, media, *, background=True, sender=send_sms):
        self.store, self.centre, self.alerts, self.media = store, centre, alerts, media
        self.ai = None
        self.sender = sender
        self.closed = threading.Event()
        with store.connect() as conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS response_deliveries (
                    id INTEGER PRIMARY KEY, incident_id TEXT NOT NULL, kind TEXT NOT NULL,
                    name TEXT NOT NULL, phone TEXT NOT NULL, status TEXT NOT NULL,
                    sid TEXT, error TEXT, updated REAL NOT NULL,
                    UNIQUE(incident_id,phone));
                CREATE TABLE IF NOT EXISTS evidence_shares (
                    token_hash TEXT PRIMARY KEY, incident_id TEXT NOT NULL,
                    filename TEXT NOT NULL, expires REAL NOT NULL);
            ''')
            columns = {row[1] for row in conn.execute('PRAGMA table_info(response_alerts)')}
            if 'contacts' not in columns:
                conn.execute('ALTER TABLE response_alerts ADD COLUMN contacts TEXT')
            for name, declaration in [('delivery_error', 'TEXT'), ('preparation_attempt', 'REAL NOT NULL DEFAULT 0')]:
                if name not in columns:
                    conn.execute(f'ALTER TABLE response_alerts ADD COLUMN {name} {declaration}')
            conn.execute("UPDATE response_deliveries SET status='uncertain', error='App stopped during sending; check Twilio logs before sending again' WHERE status='sending'")
        self.worker = None
        if background:
            self.worker = threading.Thread(target=self._work, daemon=True, name='response-delivery')
            self.worker.start()

    def status(self):
        details = self.centre.details()
        return {'configured': configuration()['ready'],
                'enabled': bool(details and details.get('messaging_enabled')),
                'message': 'SMS includes the place, coordinates, map link and a video link valid for one hour.'}

    def records(self, incident_id):
        with self.store.connect() as conn:
            conn.row_factory = sqlite3.Row
            return [dict(row) for row in conn.execute(
                'SELECT * FROM response_deliveries WHERE incident_id=? ORDER BY id', (incident_id,))]

    def process_once(self):
        details = self.centre.details()
        if not details or not details.get('messaging_enabled') or not configuration()['ready']:
            return
        with self.store.connect() as conn:
            ids = [row[0] for row in conn.execute("SELECT id FROM response_alerts WHERE state='requested' ORDER BY created")]
        for incident_id in ids:
            if self.closed.is_set():
                return
            alert = self.alerts.get(incident_id)
            # Newly enabling credentials must not dispatch a backlog of old events.
            if time.time()-alert['updated'] > 300:
                continue
            if alert.get('delivery_error') and alert['updated'] <= alert['preparation_attempt']:
                continue  # A new manual dispatch request retries failed preparation.
            with self.store.connect() as conn:
                contacts = [('authority', contact) for contact in details['authorities']]
                contacts.append(('hospital', details['hospital']))
                conn.execute('UPDATE response_alerts SET contacts=? WHERE id=? AND contacts IS NULL',
                             (json.dumps(contacts), incident_id))
                contacts = json.loads(conn.execute('SELECT contacts FROM response_alerts WHERE id=?', (incident_id,)).fetchone()[0])
            contacts = [(kind, contact) for kind, contact in contacts if kind != 'hospital' or alert['hospital']]
            attempted = {row['phone'] for row in self.records(incident_id)}
            contacts = [(kind, contact) for kind, contact in contacts if contact['phone'] not in attempted]
            if not contacts:
                continue
            try:
                package = self.alerts.package(incident_id)
                source = self.media.safe_path(self.store.clips, package['clip'])
                clip = self.media.playback(source)
            except (LookupError, ValueError):
                continue
            except (FileNotFoundError, RuntimeError):
                with self.store.connect() as conn:
                    conn.execute('UPDATE response_alerts SET delivery_error=?, preparation_attempt=? WHERE id=?',
                                 ('Could not prepare the video link. Check local evidence; slide again to retry.', time.time(), incident_id))
                continue
            with self.store.connect() as conn:
                conn.execute('UPDATE response_alerts SET delivery_error=NULL WHERE id=?', (incident_id,))
            for kind, contact in contacts:
                if self.closed.is_set():
                    return
                token = secrets.token_urlsafe(32)
                now = time.time()
                with self.store.connect() as conn:
                    conn.execute('BEGIN IMMEDIATE')
                    current = conn.execute('SELECT state,hospital FROM response_alerts WHERE id=?', (incident_id,)).fetchone()
                    if not current or current[0] != 'requested' or kind == 'hospital' and not current[1]:
                        continue
                    inserted = conn.execute('''INSERT OR IGNORE INTO response_deliveries
                        (incident_id,kind,name,phone,status,updated) VALUES (?,?,?,?,?,?)''',
                        (incident_id, kind, contact['name'], contact['phone'], 'sending', now)).rowcount
                    if not inserted:
                        continue
                    conn.execute('INSERT INTO evidence_shares VALUES (?,?,?,?)',
                                 (hashlib.sha256(token.encode()).hexdigest(), incident_id, clip.name, now+3600))
                loc = package['location']
                link = configuration()['base'] + '/e/' + token
                location_text = (f"{loc['place']} ({loc['latitude']}, {loc['longitude']}). "
                                 f"Map: https://maps.google.com/?q={loc['latitude']},{loc['longitude']} "
                                 if loc else 'Recording location unknown. ')
                body = (f"{details['name']}: {package['event_type'].replace('_', ' ')} detected. "
                        f"Human review required. {location_text}"
                        f"Video (expires in 1 hour): {link}")
                if package.get('source_kind') == 'recording':
                    body = ('Recorded-footage review; capture time and location unverified. '
                            + ('Location below is the configured location. ' if loc else '') + body)
                briefing = self.ai.briefing(incident_id) if self.ai else None
                if briefing:
                    body += ' Operator-approved AI observation: ' + briefing[:700]
                try:
                    result = self.sender(contact['phone'], body)
                except Exception:
                    # An HTTP timeout can happen after Twilio accepted the request.
                    result = {'status': 'uncertain', 'sid': None,
                              'error': 'Delivery outcome unknown. Check Twilio logs; not retried to avoid duplicates.'}
                with self.store.connect() as conn:
                    conn.execute('''UPDATE response_deliveries SET status=?,sid=?,error=?,updated=?
                        WHERE incident_id=? AND phone=?''',
                        (result['status'], result['sid'], result['error'], time.time(), incident_id, contact['phone']))

    def _work(self):
        while not self.closed.wait(.5):
            try:
                self.process_once()
            except sqlite3.Error:
                self.alerts.error = 'Response delivery could not access local storage'

    def close(self):
        self.closed.set()
        if self.worker:
            self.worker.join(timeout=10)
