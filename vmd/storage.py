"""SQLite writes and evidence encoding run on one bounded background queue."""
import json
import queue
import sqlite3
import threading
import time
from collections import deque
from datetime import datetime, time as day_time, timedelta
from pathlib import Path

import cv2
import numpy as np


class EvidenceBuffer:
    def __init__(self, seconds=10, max_bytes=32 * 1024 * 1024):
        self.seconds, self.max_bytes = seconds, max_bytes
        self.frames = deque()
        self.bytes = 0

    def append(self, timestamp, jpeg):
        self.frames.append((timestamp, jpeg))
        self.bytes += len(jpeg)
        while self.frames and (timestamp - self.frames[0][0] > self.seconds or self.bytes > self.max_bytes):
            self.bytes -= len(self.frames.popleft()[1])

    def snapshot(self):
        return list(self.frames)


class Store:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.clips = self.directory / "clips"
        self.clips.mkdir(exist_ok=True)
        self.db = self.directory / "telemetry.sqlite3"
        with self.connect() as conn:
            conn.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS incidents (
                    id TEXT PRIMARY KEY, created REAL NOT NULL, mode TEXT NOT NULL,
                    score REAL NOT NULL, reasons TEXT NOT NULL, signals TEXT NOT NULL,
                    review TEXT NOT NULL DEFAULT 'unreviewed', clip TEXT, clip_error TEXT
                );
                CREATE TABLE IF NOT EXISTS telemetry (
                    created REAL NOT NULL, people INTEGER NOT NULL, score REAL NOT NULL, mode TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS telemetry_time ON telemetry(created);
                CREATE INDEX IF NOT EXISTS incidents_time ON incidents(created);
            """)
            for table, columns in {
                "incidents": {"camera_id": "TEXT NOT NULL DEFAULT 'camera-1'", "camera_name": "TEXT NOT NULL DEFAULT 'Camera 1'", "event_type": "TEXT NOT NULL DEFAULT 'fight'", "handled_at": "REAL"},
                "telemetry": {"camera_id": "TEXT NOT NULL DEFAULT 'camera-1'", "live_camera": "INTEGER"},
            }.items():
                existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
                for column, declaration in columns.items():
                    if column not in existing:
                        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")
        self.jobs = queue.Queue(maxsize=32)
        self.error = None
        self.dropped = 0
        self.on_incident = None
        self.worker = threading.Thread(target=self._work, daemon=True, name="evidence-writer")
        self.worker.start()

    def connect(self):
        return sqlite3.connect(self.db, timeout=10)

    def enqueue(self, kind, payload):
        try:
            self.jobs.put_nowait((kind, payload))
            if kind == 'incident' and self.on_incident:
                try:
                    self.on_incident(payload[0])
                except Exception:
                    self.error = 'Incident saved to evidence queue, but response alert registration failed'
            return True
        except queue.Full:
            self.dropped += 1
            self.error = "Storage queue full; a record was not saved"
            return False

    def _work(self):
        while True:
            job = self.jobs.get()
            try:
                if job is None:
                    return
                kind, payload = job
                if kind == "incident":
                    self._incident(*payload)
                else:
                    # Legacy callers cannot distinguish recordings from physical cameras.
                    if len(payload) == 4:
                        payload = (*payload, "camera-1", None)
                    elif len(payload) == 5:
                        payload = (*payload, None)
                    with self.connect() as conn:
                        conn.execute("INSERT INTO telemetry (created,people,score,mode,camera_id,live_camera) VALUES (?, ?, ?, ?, ?, ?)", payload)
            except Exception as exc:
                self.error = f"Storage failed: {type(exc).__name__}"
            finally:
                self.jobs.task_done()

    def _incident(self, event, frames):
        clip, error = None, None
        previous = self.incident(event['id'])
        rank = {'possible_fight': 1, 'possible_snatching': 2, 'fight': 3, 'snatching_detected': 4}
        if previous:
            if (previous['event_type'] not in {'possible_fight', 'possible_snatching', 'fight'}
                    or rank.get(event.get('event_type'), 0) <= rank.get(previous['event_type'], 0)):
                return
            event = {**event, 'signals': {**event['signals'], 'escalated_at': event['created'],
                'stages': [*previous['signals'].get('stages', []),
                           {'event_type': previous['event_type'], 'created': previous['signals'].get('escalated_at', previous['created'])}]}}
        if len(frames) >= 2:
            filename = event["id"] + ('-' + event['event_type'] if previous else '') + ".avi"
            first = cv2.imdecode(np.frombuffer(frames[0][1], np.uint8), cv2.IMREAD_COLOR)
            duration = frames[-1][0] - frames[0][0]
            writer = None
            try:
                if first is None or duration <= 0:
                    raise ValueError("No decodable evidence")
                # Resample against source timestamps, preserving gaps and clip duration.
                fps = 10
                writer = cv2.VideoWriter(str(self.clips / filename), cv2.VideoWriter_fourcc(*"MJPG"), fps, (first.shape[1], first.shape[0]))
                if not writer.isOpened():
                    raise RuntimeError("Video encoder unavailable")
                index = 0
                for offset in np.arange(0, duration + 0.001, 1 / fps):
                    while index + 1 < len(frames) and frames[index+1][0] <= frames[0][0] + offset:
                        index += 1
                    frame = cv2.imdecode(np.frombuffer(frames[index][1], np.uint8), cv2.IMREAD_COLOR)
                    if frame is None:
                        raise ValueError("Invalid buffered frame")
                    writer.write(frame)
                clip = filename
            except Exception as exc:
                error = f"Clip could not be encoded: {type(exc).__name__}"
            finally:
                if writer is not None:
                    writer.release()
        else:
            error = "Not enough pre-event frames yet"
        with self.connect() as conn:
            conn.execute("""INSERT INTO incidents (id,created,mode,score,reasons,signals,clip,clip_error,camera_id,camera_name,event_type) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET score=excluded.score, reasons=excluded.reasons,
                    signals=excluded.signals, clip=COALESCE(excluded.clip, incidents.clip),
                    clip_error=CASE WHEN excluded.clip IS NOT NULL THEN NULL ELSE excluded.clip_error END,
                    event_type=excluded.event_type""", (
                event["id"], event["created"], event["mode"], event["score"], json.dumps(event["reasons"]), json.dumps(event["signals"]), clip, error, event.get("camera_id", "camera-1"), event.get("camera_name", "Camera 1"), event.get("event_type", "fight")))
            # Automatic dispatch can precede slow clip encoding. Keep its original
            # handling time when the durable incident finally becomes available.
            if conn.execute("SELECT 1 FROM pragma_table_info('response_alerts') WHERE name='handled_at'").fetchone():
                conn.execute("""UPDATE incidents SET handled_at=(SELECT handled_at FROM response_alerts WHERE id=?)
                    WHERE id=? AND handled_at IS NULL""", (event['id'], event['id']))

    def incidents(self):
        with self.connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM incidents ORDER BY created DESC LIMIT 100").fetchall()
        return [{**dict(row), "reasons": json.loads(row["reasons"]), "signals": json.loads(row["signals"])} for row in rows]

    def incident(self, incident_id):
        with self.connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute('SELECT * FROM incidents WHERE id=?', (incident_id,)).fetchone()
        return {**dict(row), 'reasons': json.loads(row['reasons']), 'signals': json.loads(row['signals'])} if row else None

    def review(self, incident_id, decision):
        with self.connect() as conn:
            return conn.execute("""UPDATE incidents SET review=?, handled_at=CASE WHEN ?='false_positive'
                THEN COALESCE(handled_at, ?) ELSE handled_at END WHERE id=?""",
                (decision, decision, time.time(), incident_id)).rowcount > 0

    def analytics(self, since):
        with self.connect() as conn:
            rows = conn.execute("SELECT CAST(created / 3600 AS INTEGER)*3600, AVG(people), MAX(people), COUNT(*), camera_id FROM telemetry WHERE created>=? AND mode='live' GROUP BY 1, camera_id ORDER BY 1, camera_id", (since,)).fetchall()
        return [{"hour": row[0], "average": round(row[1], 1), "peak": row[2], "samples": row[3], "camera_id": row[4]} for row in rows]

    def analytics_summary(self, days=7, *, now=None, camera_names=None):
        """Calendar-day incident totals and observed physical-camera crowd samples."""
        if days not in (7, 30):
            raise ValueError('Choose 7 or 30 days')
        now = time.time() if now is None else now
        today = datetime.fromtimestamp(now).date()
        first = today - timedelta(days=days - 1)
        start = datetime.combine(first, day_time()).timestamp()
        today_start = datetime.combine(today, day_time()).timestamp()
        crowd_since = now - 86400
        base = """WITH physical AS (
            SELECT * FROM incidents WHERE created >= ? AND created <= ? AND mode = 'live'
            AND CASE WHEN json_valid(signals) THEN json_type(signals, '$.live_camera') END = 'true'
        ), included AS (SELECT * FROM physical WHERE review != 'false_positive') """
        with self.connect() as conn:
            conn.row_factory = sqlite3.Row
            # Keep every chart on the same SQLite snapshot while new incidents arrive.
            conn.execute('BEGIN')
            totals = conn.execute(base + """SELECT
                COUNT(*) AS total,
                COALESCE(SUM(created >= ?), 0) AS today,
                COALESCE(SUM(event_type IN ('gun_detected', 'knife_detected')), 0) AS weapon,
                COALESCE(SUM(review = 'confirmed'), 0) AS confirmed,
                COALESCE(SUM(review = 'unreviewed'), 0) AS unreviewed,
                (SELECT COUNT(*) FROM physical WHERE review = 'false_positive') AS false_positives
                FROM included""", (start, now, today_start)).fetchone()
            trend = dict(conn.execute(base + """SELECT date(created, 'unixepoch', 'localtime'), COUNT(*)
                FROM included GROUP BY 1""", (start, now)).fetchall())
            by_type = [dict(row) for row in conn.execute(base + """SELECT event_type, COUNT(*) AS value
                FROM included GROUP BY event_type ORDER BY value DESC, event_type""", (start, now))]
            by_camera = [dict(row) for row in conn.execute(base + """SELECT camera_id,
                (SELECT camera_name FROM included AS named WHERE named.camera_id = grouped.camera_id
                 ORDER BY created DESC, id DESC LIMIT 1) AS camera_name, COUNT(*) AS value
                FROM included AS grouped GROUP BY camera_id ORDER BY value DESC, camera_id""", (start, now))]
            names = {row['camera_id']: row['camera_name'] for row in by_camera}
            names.update(camera_names or {})
            crowd = [dict(row) for row in conn.execute("""SELECT CAST(created / 3600 AS INTEGER)*3600 AS hour,
                ROUND(AVG(people), 1) AS average, MAX(people) AS peak, COUNT(*) AS samples, camera_id
                FROM telemetry WHERE created >= ? AND created <= ? AND mode = 'live' AND live_camera = 1
                GROUP BY hour, camera_id ORDER BY hour, camera_id""", (crowd_since, now))]
        return {'days': days, 'generated_at': now, 'start': start,
                'timezone': datetime.fromtimestamp(now).astimezone().tzname(), **dict(totals),
                'trend': [{'date': (first + timedelta(days=index)).isoformat(),
                           'value': trend.get((first + timedelta(days=index)).isoformat(), 0)}
                          for index in range(days)],
                'by_type': by_type, 'by_camera': by_camera,
                'crowd': [{**row, 'camera_name': names.get(row['camera_id'], row['camera_id'])} for row in crowd],
                'crowd_since': crowd_since}

    def close(self):
        self.jobs.put(None)
        self.worker.join(timeout=20)
