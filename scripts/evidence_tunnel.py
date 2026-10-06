"""Manage a free Cloudflare Quick Tunnel to the evidence-only gateway (POSIX)."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ORIGIN = re.compile(r'https://[a-z0-9]+(?:-[a-z0-9]+)*\.trycloudflare\.com\b')


def private_write(path, value):
    temporary = path.with_suffix('.tmp')
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(value, stream)
    temporary.replace(path)


def birth(pid):
    if type(pid) is not int or pid <= 1:
        return None
    result = subprocess.run(['ps', '-p', str(pid), '-o', 'stat=', '-o', 'lstart='],
                            capture_output=True, text=True, check=False)
    value = result.stdout.strip()
    if result.returncode or not value or value.startswith('Z'):
        return None
    return value.split(maxsplit=1)[1]


def owned(record, kind):
    expected = record.get(kind + '_birth')
    return bool(expected and birth(record.get(kind + '_pid')) == expected)


def health(origin):
    try:
        with urlopen(origin + '/health', timeout=3) as response:
            return response.status == 200 and json.load(response) == {
                'status': 'ok', 'service': 'vdma-evidence'}
    except (OSError, ValueError):
        return False


def load(path):
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def stop_owned(record):
    for kind in ('tunnel', 'gateway'):
        if not owned(record, kind):
            continue
        pid = record[kind + '_pid']
        try:
            os.kill(pid, signal.SIGTERM)
            deadline = time.monotonic() + 8
            while owned(record, kind) and time.monotonic() < deadline:
                time.sleep(.1)
            if owned(record, kind):
                os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def launch(command, log_path, env):
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'wb') as log:
        os.fchmod(log.fileno(), 0o600)
        return subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=log, start_new_session=True)


def start(directory, port, state_path):
    record = load(state_path)
    local = f'http://127.0.0.1:{port}'
    if (record.get('port') == port and owned(record, 'gateway') and owned(record, 'tunnel')
            and health(local)):
        print(json.dumps({'status': 'running', 'base_url': record['base_url'], 'port': port}))
        return
    if any(owned(record, kind) for kind in ('gateway', 'tunnel')):
        raise RuntimeError('A partial evidence service is running. Run stop, then start again.')
    with socket.socket() as sock:
        try:
            sock.bind(('127.0.0.1', port))
        except OSError:
            raise RuntimeError(f'Port {port} is occupied; no existing process was changed.') from None
    cloudflared = shutil.which('cloudflared')
    if not cloudflared:
        raise RuntimeError('cloudflared is missing. Install it, then run start again.')
    runtime = directory / 'evidence-runtime'
    runtime.mkdir(mode=0o700, exist_ok=True)
    # Use the current virtualenv executable as invoked; resolving its symlink loses the venv.
    python = sys.executable
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('TUNNEL_', 'TWILIO_', 'VMD_EVIDENCE_'))}
    env['VMD_DATA_DIR'] = str(directory)
    record = {'port': port, 'base_url': None}
    children = []
    state_path.unlink(missing_ok=True)
    try:
        gateway = launch([python, '-m', 'uvicorn', 'vmd.share:app', '--host', '127.0.0.1',
                          '--port', str(port), '--no-access-log', '--no-proxy-headers'],
                         runtime / 'gateway.log', env)
        children.append(gateway)
        record.update(gateway_pid=gateway.pid, gateway_birth=birth(gateway.pid))
        deadline = time.monotonic() + 12
        while not health(local):
            if gateway.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError('Evidence gateway did not start; inspect its private runtime log.')
            time.sleep(.2)
        tunnel_log = runtime / 'cloudflared.log'
        tunnel = launch([cloudflared, 'tunnel', '--config', os.devnull, '--no-autoupdate',
                         '--protocol', 'http2', '--edge-ip-version', '4',
                         '--metrics', '127.0.0.1:0', '--grace-period', '5s',
                         '--loglevel', 'info', '--url', local], tunnel_log, env)
        children.append(tunnel)
        record.update(tunnel_pid=tunnel.pid, tunnel_birth=birth(tunnel.pid))
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            if gateway.poll() is not None or tunnel.poll() is not None:
                raise RuntimeError('Evidence tunnel stopped during startup; inspect its private runtime log.')
            log = tunnel_log.read_text(errors='replace')
            match = PUBLIC_ORIGIN.search(log)
            if match and 'Registered tunnel connection' in log and health(match[0]):
                record['base_url'] = match[0]
                record['started_at'] = time.time()
                private_write(state_path, record)
                print(json.dumps({'status': 'running', 'base_url': match[0], 'port': port}))
                return
            time.sleep(.5)
        raise RuntimeError('Cloudflare did not become reachable. Check Internet/firewall access and the private runtime log.')
    except BaseException:
        stop_owned(record)
        for child in children:
            if child.poll() is None:
                child.terminate()
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=3)
        state_path.unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('start', 'status', 'stop'))
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--port', type=int, default=8768)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535 or args.port in (8765, 8770, 8781):
        parser.error('Choose a dedicated unprivileged evidence port; the default is 8768.')
    directory = args.data_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / 'evidence-host.json'
    lock = os.open(directory / 'evidence-host.lock', os.O_RDWR | os.O_CREAT, 0o600)
    try:
        # ponytail: one local manager lock; separate data directories isolate test gateways.
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = load(state_path)
        if args.action == 'start':
            start(directory, args.port, state_path)
        elif args.action == 'stop':
            # Clear public configuration before shutting down; link generation fails closed.
            state_path.unlink(missing_ok=True)
            stop_owned(record)
            print(json.dumps({'status': 'stopped'}))
        else:
            running = (owned(record, 'gateway') and owned(record, 'tunnel')
                       and health(f"http://127.0.0.1:{record.get('port', args.port)}"))
            print(json.dumps({'status': 'running' if running else 'stopped',
                              'base_url': record.get('base_url') if running else None,
                              'port': record.get('port', args.port)}))
    except (RuntimeError, BlockingIOError) as exc:
        print(str(exc) or 'Another evidence manager is busy.', file=sys.stderr)
        return 1
    finally:
        os.close(lock)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
