"""Durable incident response countdowns; delivery waits for a configured integration."""
import json
import sqlite3
import threading
import time


AUTO_EVENTS = {'fight', 'knife_detected', 'snatching_detected'}


def manual_dispatch_eligible(event):
    # A file uses the real pipeline's live mode but explicitly records False.
    # Missing/legacy provenance and synthetic modes are not real response inputs.
    return (event.get('mode') == 'live'
            and type(event.get('signals', {}).get('live_camera')) is bool)


class AlertAgent:
    def __init__(self, store, *, background=True):
        self.store = store
        self.error = None
        self.closed = threading.Event()
        with store.connect() as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS response_alerts (
                id TEXT PRIMARY KEY, event TEXT NOT NULL, created REAL NOT NULL,
                deadline REAL NOT NULL, state TEXT NOT NULL, hospital INTEGER NOT NULL DEFAULT 0,
                trigger TEXT, updated REAL NOT NULL
            )''')
            columns = {row[1] for row in conn.execute('PRAGMA table_info(response_alerts)')}
            for name, declaration in [('voice_requested_at', 'REAL'), ('voice_contacts', 'TEXT'), ('handled_at', 'REAL')]:
                if name not in columns:
                    conn.execute(f'ALTER TABLE response_alerts ADD COLUMN {name} {declaration}')
            # Upgrade historical responses without starting a fresh 30-second
            # operator hold each time the server starts.
            conn.execute("""UPDATE response_alerts SET handled_at=CASE
                WHEN trigger='automatic' THEN deadline
                WHEN trigger IS NOT NULL OR state='requested' THEN COALESCE(voice_requested_at, updated)
                ELSE updated END
                WHERE handled_at IS NULL AND (trigger IS NOT NULL OR state='requested'
                    OR (state='cancelled' AND id IN (SELECT id FROM incidents WHERE review='false_positive')))""")
            conn.execute("""UPDATE incidents SET handled_at=(SELECT handled_at FROM response_alerts WHERE id=incidents.id)
                WHERE handled_at IS NULL AND id IN (SELECT id FROM response_alerts WHERE handled_at IS NOT NULL)""")
            # Old gun false positives may have no response record or review time.
            conn.execute("UPDATE incidents SET handled_at=created WHERE review='false_positive' AND handled_at IS NULL")
        self.worker = None
        if background:
            self.worker = threading.Thread(target=self._work, daemon=True, name='incident-response')
            self.worker.start()

    def register(self, event):
        if not manual_dispatch_eligible(event):
            return
        with self.store.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            incident = conn.execute('SELECT review, handled_at FROM incidents WHERE id=?', (event['id'],)).fetchone()
            existing = conn.execute('SELECT id FROM response_alerts WHERE id=?', (event['id'],)).fetchone()
            if existing:
                # Upgrade the event description, preserving its dispatch/cancel state,
                # recipients and original deadline. Never dial twice for an upgrade.
                conn.execute('UPDATE response_alerts SET event=? WHERE id=?', (json.dumps(event), event['id']))
                return
            # Recording analysis can update a manual request, never create a countdown.
            if (event['signals']['live_camera'] is not True
                    or event.get('event_type') not in AUTO_EVENTS):
                return
            if incident and (incident[0] != 'unreviewed' or incident[1] is not None):
                return
            conn.execute('''INSERT OR IGNORE INTO response_alerts
                (id,event,created,deadline,state,updated) VALUES (?,?,?,?,?,?)''',
                (event['id'], json.dumps(event), event['created'], event['created']+10,
                 'pending', time.time()))

    def tick(self, now=None):
        now = time.time() if now is None else now
        with self.store.connect() as conn:
            changed = conn.execute("""UPDATE response_alerts SET state='requested', trigger='automatic',
                updated=deadline, voice_requested_at=deadline, handled_at=COALESCE(handled_at, deadline)
                WHERE state='pending' AND deadline<=?""", (now,)).rowcount
            if changed:
                conn.execute("""UPDATE incidents SET handled_at=(SELECT handled_at FROM response_alerts WHERE id=incidents.id)
                    WHERE handled_at IS NULL AND id IN (SELECT id FROM response_alerts WHERE handled_at IS NOT NULL)""")

    def _work(self):
        while not self.closed.wait(.2):
            try:
                self.tick()
                self.error = None
            except sqlite3.Error:
                self.error = 'Response countdown could not be saved. Check local storage.'

    @staticmethod
    def _decode(row):
        return {**dict(row), 'event': json.loads(row['event']), 'hospital': bool(row['hospital'])}

    def get(self, incident_id):
        with self.store.connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute('SELECT * FROM response_alerts WHERE id=?', (incident_id,)).fetchone()
        return self._decode(row) if row else None

    def snapshot(self):
        with self.store.connect() as conn:
            conn.row_factory = sqlite3.Row
            # Active alerts are never lost behind the incident library's 100-row limit.
            rows = conn.execute("""SELECT * FROM response_alerts WHERE state IN ('pending','requested')
                OR id IN (SELECT id FROM response_alerts ORDER BY created DESC LIMIT 100) ORDER BY created""").fetchall()
        return {'alerts': [self.describe(self._decode(row)) for row in rows],
                'server_time': time.time(), 'error': self.error}

    def describe(self, alert):
        event = self.store.incident(alert['id'])
        location = alert['event'].get('signals', {}).get('location')
        issues = []
        if not location and alert['event'].get('signals', {}).get('live_camera') is not False:
            issues.append('Camera place and coordinates are missing')
        if not event:
            issues.append('Evidence is still being saved')
        elif not event.get('clip'):
            issues.append(event.get('clip_error') or 'Evidence clip unavailable')
        if alert.get('delivery_error'):
            issues.append(alert['delivery_error'])
        return {**alert, 'incident_saved': event is not None, 'clip_ready': bool(event and event.get('clip')),
                'delivery_issues': issues}

    def act(self, incident_id, action, hospital=None):
        now = time.time()
        with self.store.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            conn.row_factory = sqlite3.Row
            row = conn.execute('SELECT * FROM response_alerts WHERE id=?', (incident_id,)).fetchone()
            incident = conn.execute('SELECT * FROM incidents WHERE id=?', (incident_id,)).fetchone()
            if not row:
                if not incident:
                    raise LookupError('Incident not found')
                event = {**dict(incident), 'signals': json.loads(incident['signals']),
                         'reasons': json.loads(incident['reasons'])}
                if (not manual_dispatch_eligible(event)
                        or action not in {'dispatch', 'call_hospital'}):
                    raise ValueError('Only saved camera or recording incidents can be dispatched')
                conn.execute('''INSERT INTO response_alerts
                    (id,event,created,deadline,state,updated) VALUES (?,?,?,?,?,?)''',
                    (incident_id, json.dumps(event), event['created'], event['created']+10, 'acknowledged', now))
                row = conn.execute('SELECT * FROM response_alerts WHERE id=?', (incident_id,)).fetchone()
            if action in {'dispatch', 'call_hospital'} and not manual_dispatch_eligible(json.loads(row['event'])):
                raise ValueError('Only saved camera or recording incidents can be dispatched')
            if action in {'dispatch', 'call_hospital'} and (row['state'] == 'cancelled' or incident and incident['review'] == 'false_positive'):
                raise ValueError('A false-positive incident cannot be dispatched')
            if action == 'acknowledge':
                if row['state'] in {'pending', 'requested'}:
                    conn.execute("UPDATE response_alerts SET state='acknowledged', updated=? WHERE id=?", (now, incident_id))
            elif action == 'cancel':
                conn.execute("UPDATE response_alerts SET state='cancelled', updated=? WHERE id=?", (now, incident_id))
            elif action in {'dispatch', 'call_hospital'}:
                # Repeated slider requests reuse the same durable incident request.
                conn.execute("""UPDATE response_alerts SET state='requested',
                    trigger=COALESCE(trigger,'manual'), updated=?, voice_requested_at=?,
                    handled_at=COALESCE(handled_at, ?, ?) WHERE id=?""",
                    (now, now, incident['handled_at'] if incident else None, now, incident_id))
                conn.execute("""UPDATE incidents SET handled_at=COALESCE(handled_at,
                    (SELECT handled_at FROM response_alerts WHERE id=?)) WHERE id=?""", (incident_id, incident_id))
                if action == 'call_hospital':
                    hospital = True
            elif action != 'hospital':
                raise ValueError('Unknown response action')
            if hospital is not None:
                conn.execute('UPDATE response_alerts SET hospital=?, updated=? WHERE id=?', (hospital, now, incident_id))
        return self.describe(self.get(incident_id))

    def review(self, incident_id, decision):
        # Review and cancellation share one transaction with the countdown's database.
        now = time.time()
        with self.store.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            exists = conn.execute("""UPDATE incidents SET review=?, handled_at=CASE WHEN ?='false_positive'
                THEN COALESCE(handled_at, ?) ELSE handled_at END WHERE id=?""",
                (decision, decision, now, incident_id)).rowcount
            if decision == 'false_positive':
                conn.execute("""UPDATE response_alerts SET state='cancelled', updated=?, handled_at=COALESCE(handled_at,
                    (SELECT handled_at FROM incidents WHERE id=?), ?) WHERE id=?""", (now, incident_id, now, incident_id))
            elif decision == 'confirmed':
                conn.execute("UPDATE response_alerts SET state='acknowledged', updated=? WHERE id=? AND state='pending'", (now, incident_id))
        return bool(exists)

    def package(self, incident_id):
        alert = self.get(incident_id)
        if not alert:
            raise LookupError('Response request not found')
        event = self.store.incident(incident_id)
        if not event or not event.get('clip'):
            raise ValueError('Evidence clip is not available yet')
        location = alert['event'].get('signals', {}).get('location')
        recorded = event.get('signals', {}).get('live_camera') is False
        if not location and not recorded:
            raise ValueError('Camera place and coordinates were not configured at detection time')
        return {'incident_id': incident_id, 'event_type': event['event_type'],
                'detected_at': event['created'], 'camera_id': event['camera_id'],
                'camera_name': event['camera_name'], 'confidence': event['score'],
                'source_kind': 'recording' if recorded else 'live_camera',
                'source_seconds': event.get('signals', {}).get('source_seconds'),
                'assessment': ('Recorded footage analyzed at detected_at; original capture time and location unverified. '
                               if recorded else '') + 'Automated detection; requires human verification',
                'location': location, 'clip': event['clip'],
                'recipients': ['authority'] + (['hospital'] if alert['hospital'] else []),
                'trigger': alert['trigger']}

    def close(self):
        self.closed.set()
        if self.worker:
            self.worker.join(timeout=3)
