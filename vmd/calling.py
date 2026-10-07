"""Incident voice calls to saved centre contacts, independent of video hosting."""
import base64
import json
import os
import re
import sqlite3
import threading
import time
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, build_opener
from xml.etree.ElementTree import Element, SubElement, tostring

from .delivery import NoRedirect


ACTIVE = {'queued', 'ringing', 'in-progress'}
STATUSES = ACTIVE | {'completed', 'busy', 'failed', 'no-answer', 'canceled'}
TRIAL_TEMPLATE = 'https://webhooks.twilio.com/v1/Voice/Template/voice_text_to_speech'


def configuration(directory=None):
    saved = {}
    if directory is not None:
        try:
            value = json.loads((Path(directory) / 'twilio.json').read_text())
            if isinstance(value, dict):
                saved = value
        except (OSError, ValueError):
            pass
    values = {key: os.getenv(env, saved.get(key, '')) for key, env in (
        ('sid', 'TWILIO_ACCOUNT_SID'), ('token', 'TWILIO_AUTH_TOKEN'), ('sender', 'TWILIO_FROM_NUMBER'))}
    values = {key: value if isinstance(value, str) else '' for key, value in values.items()}
    values['mode'] = os.getenv('TWILIO_VOICE_MODE', saved.get('mode', 'incident'))
    values['ready'] = bool(re.fullmatch(r'AC[0-9a-fA-F]{32}', values['sid'])
                           and values['token'] and re.fullmatch(r'\+[1-9][0-9]{7,14}', values['sender'])
                           and values['mode'] in ('incident', 'trial_template', 'gemini_live'))
    return values


def call_instructions(centre_name, event, briefing=None):
    signals = event.get('signals', {})
    location = signals.get('location')
    detected = datetime.fromtimestamp(event['created'], tz=timezone.utc).astimezone()
    when = detected.strftime('%I:%M %p on %d %B %Y %Z').lstrip('0')
    incident = {'fight': 'A fight', 'possible_fight': 'A possible fight',
                'possible_snatching': 'A possible snatching incident', 'snatching_detected': 'A snatching incident', 'knife_detected': 'A knife',
                'gun_detected': 'A gun', 'fall': 'A possible fall', 'possible_fall': 'A possible fall',
                'unattended_object': 'A possibly unattended item',
                'person_down': 'A person down', 'person_down_after_fight': 'A person down after a possible fight',
                'hands_up': 'A person with hands up'}.get(event['event_type'], event['event_type'].replace('_', ' '))
    place = location['place'] if location else f"camera {event['camera_name']}"
    if event.get('mode') == 'live' and signals.get('live_camera') is False:
        offset = signals.get('source_seconds')
        position = (f"About {offset:.1f} seconds into the recording. "
                    if isinstance(offset, (int, float)) and not isinstance(offset, bool)
                    and isfinite(offset) and offset >= 0 else '')
        message = (f"Hello, we are calling from {centre_name}. "
                   f"{incident} was identified in recorded footage labelled {event['camera_name']}. "
                   f"The footage was analyzed at {when}. " + position
                   + (f"Configured location: {place}. Latitude {location['latitude']}, longitude {location['longitude']}. "
                      if location else 'No location has been configured for this recording. ')
                   + 'The original recording time and capture location have not been verified. '
                   + 'Please review the footage and coordinate a response.')
    else:
        message = (f"Hello, we are calling from {centre_name}. "
                   f"{incident} was detected at {place} at {when}. "
                   + (f"Camera {event['camera_name']}. Latitude {location['latitude']}, longitude {location['longitude']}. "
                      if location else 'The camera location has not been configured. ')
                   + 'Please verify the incident and coordinate a response.')
    if briefing:
        message += ' Operator-approved AI observation: ' + briefing[:700]
    response = Element('Response')
    SubElement(response, 'Say', {'voice': 'Polly.Joanna', 'language': 'en-US', 'loop': '2'}).text = re.sub(r'[\x00-\x1f]', ' ', message)
    SubElement(response, 'Hangup')
    return tostring(response, encoding='unicode')


def request_call(config, phone=None, twiml=None, sid=None, instruction_url=None):
    if not config['ready']:
        raise ValueError('Twilio calling credentials are not configured')
    if sid is not None and not re.fullmatch(r'CA[0-9a-fA-F]{32}', sid):
        raise ValueError('Invalid call ID')
    auth = base64.b64encode(f"{config['sid']}:{config['token']}".encode()).decode()
    suffix = f'/{sid}.json' if sid else '.json'
    body = None
    if sid is None:
        # Match the earlier caller's To/From/Url request; this account rejected inline Twiml.
        # Echo receives the announcement in its URL. Never log this URL or include credentials in it.
        url = instruction_url or (TRIAL_TEMPLATE if config.get('mode') == 'trial_template'
               else 'https://twimlets.com/echo?' + urlencode({'Twiml': twiml}))
        body = urlencode({'To': phone, 'From': config['sender'], 'Url': url}).encode()
    request = Request(f"https://api.twilio.com/2010-04-01/Accounts/{config['sid']}/Calls{suffix}",
                      data=body, headers={'Authorization': f'Basic {auth}',
                                          'Content-Type': 'application/x-www-form-urlencoded'})
    try:
        with build_opener(NoRedirect).open(request, timeout=8) as response:
            value = json.loads(response.read(65536))
    except HTTPError as exc:
        code, message = None, ''
        try:
            error = json.loads(exc.read(65536))
            code, message = error.get('code'), str(error.get('message', ''))
        except (ValueError, AttributeError):
            pass
        detail = f', code {code}' if isinstance(code, int) else ''
        if sid:
            raise RuntimeError('Could not refresh call status') from None
        if exc.code >= 500:
            return {'sid': None, 'status': 'uncertain',
                    'error': 'Twilio server error; call outcome unknown. Check Twilio logs before calling again.'}
        if exc.code == 400 and 'trial accounts have limited parameter access' in message.lower():
            return {'sid': None, 'status': 'failed',
                    'error': 'Twilio trial rejected this call request. Check outbound Voice permissions and the allowed instruction URL in Twilio; no call was created.'}
        return {'sid': None, 'status': 'failed', 'error': f'Twilio rejected the call (HTTP {exc.code}{detail})'}
    if (not isinstance(value, dict) or not re.fullmatch(r'CA[0-9a-fA-F]{32}', value.get('sid', ''))
            or value.get('status') not in STATUSES or sid and value['sid'] != sid):
        raise RuntimeError('Unrecognized call response')
    return {'sid': value['sid'], 'status': value['status'], 'error': None}


class Calling:
    def __init__(self, store, centre, alerts, *, background=True, sender=None, fetcher=None):
        self.store, self.centre, self.alerts = store, centre, alerts
        self.sender = sender or (lambda phone, twiml: request_call(configuration(store.directory), phone, twiml))
        self.fetcher = fetcher or (lambda sid: request_call(configuration(store.directory), sid=sid))
        self.ai = None
        self.custom_sender = sender is not None
        self.closed = threading.Event()
        self.error = None
        with store.connect() as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS response_calls (
                id INTEGER PRIMARY KEY, incident_id TEXT NOT NULL, kind TEXT NOT NULL,
                name TEXT NOT NULL, phone TEXT NOT NULL, status TEXT NOT NULL, sid TEXT,
                error TEXT, created REAL NOT NULL, updated REAL NOT NULL,
                UNIQUE(incident_id,phone))''')
            conn.execute("UPDATE response_calls SET status='uncertain', error='App stopped during call submission; check Twilio logs before calling again' WHERE status='submitting'")
        self.worker = None
        if background:
            self.worker = threading.Thread(target=self._work, daemon=True, name='response-calling')
            self.worker.start()

    def status(self):
        details = self.centre.details()
        config = configuration(self.store.directory)
        return {'configured': config['ready'], 'mode': config.get('mode', 'incident'),
                'enabled': bool(details and details.get('calling_enabled')), 'error': self.error}

    def records(self, incident_id):
        with self.store.connect() as conn:
            conn.row_factory = sqlite3.Row
            return [dict(row) for row in conn.execute(
                'SELECT * FROM response_calls WHERE incident_id=? ORDER BY id', (incident_id,))]

    def process_once(self):
        details = self.centre.details()
        if not details or not details.get('calling_enabled') or not configuration(self.store.directory)['ready']:
            return
        with self.store.connect() as conn:
            ids = [row[0] for row in conn.execute("""SELECT id FROM response_alerts
                WHERE state='requested' AND voice_requested_at>=? ORDER BY voice_requested_at""", (time.time()-300,))]
        for incident_id in ids:
            if self.closed.is_set():
                return
            with self.store.connect() as conn:
                contacts = [('authority', item) for item in details['authorities']] + [('hospital', details['hospital'])]
                conn.execute('UPDATE response_alerts SET voice_contacts=? WHERE id=? AND voice_contacts IS NULL',
                             (json.dumps(contacts), incident_id))
            alert = self.alerts.get(incident_id)
            contacts = json.loads(alert['voice_contacts'])
            briefing = self.ai.briefing(incident_id) if self.ai else None
            instructions = call_instructions(details['name'], alert['event'], briefing)
            for kind, contact in contacts:
                if self.closed.is_set() or not self.centre.details().get('calling_enabled'):
                    return
                now = time.time()
                with self.store.connect() as conn:
                    conn.execute('BEGIN IMMEDIATE')
                    current = conn.execute('SELECT state,hospital FROM response_alerts WHERE id=?', (incident_id,)).fetchone()
                    if current[0] != 'requested' or kind == 'hospital' and not current[1]:
                        continue
                    inserted = conn.execute('''INSERT OR IGNORE INTO response_calls
                        (incident_id,kind,name,phone,status,created,updated) VALUES (?,?,?,?,?,?,?)''',
                        (incident_id, kind, contact['name'], contact['phone'], 'submitting', now, now)).rowcount
                    if not inserted:
                        continue
                try:
                    config = configuration(self.store.directory)
                    if config.get('mode') == 'gemini_live' and not self.custom_sender:
                        from .voice import issue_voice_url
                        # Gateway health failure uses the existing announcement immediately.
                        instruction_url = issue_voice_url(self.store, incident_id, contact['phone'])
                        if instruction_url:
                            result = request_call(config, contact['phone'], instructions, instruction_url=instruction_url)
                        else:
                            result = request_call({**config, 'mode': 'incident'}, contact['phone'], instructions)
                    else:
                        result = self.sender(contact['phone'], instructions)
                except Exception:
                    result = {'status': 'uncertain', 'sid': None,
                              'error': 'Call outcome unknown. Check Twilio logs; not retried to avoid duplicate calls.'}
                with self.store.connect() as conn:
                    conn.execute('UPDATE response_calls SET status=?,sid=?,error=?,updated=? WHERE incident_id=? AND phone=?',
                                 (result['status'], result['sid'], result['error'], time.time(), incident_id, contact['phone']))

    def poll_once(self):
        if not configuration(self.store.directory)['ready']:
            return
        now = time.time()
        with self.store.connect() as conn:
            conn.execute("""UPDATE response_calls SET status='uncertain',
                error='Final call status unavailable after ten minutes; check Twilio logs', updated=?
                WHERE status IN ('queued','ringing','in-progress') AND created<=?""", (now, now-600))
            rows = conn.execute("""SELECT id,sid FROM response_calls WHERE sid IS NOT NULL
                AND status IN ('queued','ringing','in-progress') AND updated<? AND created>?""", (now-5, now-600)).fetchall()
        for row_id, sid in rows:
            if self.closed.is_set():
                return
            try:
                result = self.fetcher(sid)
            except Exception:
                with self.store.connect() as conn:
                    conn.execute('UPDATE response_calls SET error=?,updated=? WHERE id=?',
                                 ('Status refresh unavailable; showing the last known call state', now, row_id))
                continue
            with self.store.connect() as conn:
                conn.execute('UPDATE response_calls SET status=?,error=?,updated=? WHERE id=?',
                             (result['status'], result['error'], now, row_id))

    def _work(self):
        while not self.closed.wait(.5):
            try:
                self.process_once()
                self.poll_once()
                self.error = None
            except sqlite3.Error:
                self.error = 'Calling could not access local storage'

    def close(self):
        self.closed.set()
        if self.worker:
            self.worker.join(timeout=10)
