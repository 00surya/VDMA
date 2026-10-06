"""Start/stop only this lab's detached local service."""
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request
import psutil

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / '.runtime'
PID_FILE = RUNTIME / 'server.json'
URL = 'http://127.0.0.1:8770'


def owned_process():
    if not PID_FILE.exists():
        return None
    record = json.loads(PID_FILE.read_text())
    try:
        process = psutil.Process(record['pid'])
        if (process.create_time() != record['created'] or Path(process.cwd()) != ROOT
                or str(ROOT / 'app.py') not in process.cmdline()):
            return None
        return process
    except psutil.NoSuchProcess:
        return None


def main():
    action = sys.argv[1] if len(sys.argv) == 2 else 'status'
    process = owned_process()
    if action == 'stop':
        if process:
            process.terminate()
            try:
                process.wait(timeout=10)
            except psutil.TimeoutExpired:
                process.kill()
            PID_FILE.unlink(missing_ok=True)
        print('Vision Lab stopped. VDMA was not touched.')
    elif action == 'start':
        if process:
            print(f'Vision Lab is already running at {URL}')
            return
        with socket.socket() as sock:
            if sock.connect_ex(('127.0.0.1', 8770)) == 0:
                raise SystemExit('Port 8770 is occupied. No process was changed.')
        RUNTIME.mkdir(exist_ok=True)
        with (RUNTIME / 'server.log').open('ab') as log:
            child = subprocess.Popen([str(ROOT / '.venv/bin/python'), str(ROOT / 'app.py')],
                cwd=ROOT, stdout=log, stderr=log, stdin=subprocess.DEVNULL, start_new_session=True)
        PID_FILE.write_text(json.dumps({'pid': child.pid, 'created': psutil.Process(child.pid).create_time()}))
        for _ in range(40):
            if child.poll() is not None:
                raise SystemExit('Lab startup failed; see vision-lab/.runtime/server.log.')
            try:
                with urllib.request.urlopen(URL + '/api/state', timeout=.5) as response:
                    if response.status == 200:
                        print(f'Vision Lab ready: {URL} (PID {child.pid})')
                        return
            except (OSError, TimeoutError):
                time.sleep(.2)
        raise SystemExit('Lab did not become ready; inspect .runtime/server.log.')
    elif action == 'status':
        print(f'Vision Lab: {URL} (PID {process.pid})' if process else 'Vision Lab is stopped.')
    else:
        raise SystemExit('Usage: .venv/bin/python manage.py start|stop|status')


if __name__ == '__main__':
    main()
