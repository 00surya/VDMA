"""Local video library and cached browser-compatible evidence playback."""
import hashlib
import math
import shutil
import sqlite3
import subprocess
import threading
import time
import uuid
from pathlib import Path

import cv2

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}
MAX_UPLOAD = 1024 * 1024 * 1024


class MediaLibrary:
    def __init__(self, store):
        self.store = store
        self.directory = store.directory / 'recordings'
        self.cache = store.directory / 'playback'
        self.directory.mkdir(exist_ok=True)
        self.cache.mkdir(exist_ok=True)
        # ponytail: serialize local conversions; use a job pool if throughput demands it.
        self.encode_lock = threading.Lock()
        with store.connect() as conn:
            conn.execute('CREATE TABLE IF NOT EXISTS recordings (id TEXT PRIMARY KEY, name TEXT NOT NULL, '
                         'filename TEXT NOT NULL, created REAL NOT NULL, size INTEGER NOT NULL, duration REAL NOT NULL)')

    def recordings(self):
        with self.store.connect() as conn:
            conn.row_factory = sqlite3.Row
            return [dict(row) for row in conn.execute('SELECT id,name,created,size,duration FROM recordings ORDER BY created DESC')]

    @staticmethod
    def safe_path(directory, filename):
        path = (directory / filename).resolve()
        if not path.is_relative_to(directory.resolve()) or not path.is_file():
            raise FileNotFoundError('Media file is unavailable')
        return path

    def source(self, recording_id):
        with self.store.connect() as conn:
            row = conn.execute('SELECT filename FROM recordings WHERE id=?', (recording_id,)).fetchone()
        if not row:
            raise FileNotFoundError('Recording not found')
        return self.safe_path(self.directory, row[0])

    def register(self, temporary, name, recording_id, suffix):
        cap = cv2.VideoCapture(str(temporary))
        try:
            ok, frame = cap.read()
            fps, count = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = count / fps if fps > 0 else 0
            if not ok or frame is None or not math.isfinite(duration) or duration <= 0:
                raise ValueError('This file could not be read as a supported video.')
        finally:
            cap.release()
        filename = recording_id + suffix
        final = self.directory / filename
        temporary.replace(final)
        item = {'id': recording_id, 'name': name, 'created': time.time(),
                'size': final.stat().st_size, 'duration': round(duration, 3)}
        try:
            with self.store.connect() as conn:
                conn.execute('INSERT INTO recordings VALUES (?,?,?,?,?,?)',
                    (recording_id, name, filename, item['created'], item['size'], item['duration']))
        except Exception:
            final.unlink(missing_ok=True)
            raise
        return item

    def playback(self, source):
        stat = source.stat()
        token = hashlib.sha256(f'{source.resolve()}:{stat.st_size}:{stat.st_mtime_ns}'.encode()).hexdigest()
        target = self.cache / f'{token}.mp4'
        with self.encode_lock:
            if target.is_file():
                return target
            executable = shutil.which('ffmpeg')
            if not executable:
                try:
                    import imageio_ffmpeg
                    executable = imageio_ffmpeg.get_ffmpeg_exe()
                except (ImportError, RuntimeError):
                    raise RuntimeError('Install the project playback dependencies to enable the media player.') from None
            temporary = self.cache / f'{uuid.uuid4().hex}.tmp.mp4'
            try:
                subprocess.run([executable, '-nostdin', '-y', '-v', 'error', '-i', str(source),
                    '-map', '0:v:0', '-an', '-vf', 'scale=min(1280\\,iw):-2',
                    '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', '-pix_fmt', 'yuv420p',
                    '-threads', '2', '-movflags', '+faststart', str(temporary)],
                    check=True, capture_output=True, timeout=180)
                if not temporary.is_file() or not temporary.stat().st_size:
                    raise RuntimeError('Playback conversion produced no video')
                temporary.replace(target)
            except (OSError, subprocess.SubprocessError):
                raise RuntimeError('Playback conversion failed. Download the original video instead.') from None
            finally:
                temporary.unlink(missing_ok=True)
        return target
