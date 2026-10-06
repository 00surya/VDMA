"""Evidence hosting lifecycle checks; Cloudflare and network traffic are mocked."""
import errno
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import evidence_tunnel as manager


ORIGIN = 'https://synthetic-evidence-test.trycloudflare.com'


@pytest.fixture(autouse=True)
def no_external_health(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail('A lifecycle test must not contact an external service')
    monkeypatch.setattr(manager, 'urlopen', unexpected)


def record():
    return {'port': 8768, 'base_url': ORIGIN,
            'gateway_pid': 201, 'gateway_birth': 'gateway-start',
            'tunnel_pid': 202, 'tunnel_birth': 'tunnel-start'}


def test_private_write_replaces_config_atomically_with_owner_permissions(tmp_path, monkeypatch):
    path = tmp_path / 'evidence-host.json'
    path.write_text('{"old": true}')
    path.chmod(0o644)
    real_replace = Path.replace
    observed = []

    def checked_replace(temporary, destination):
        assert json.loads(path.read_text()) == {'old': True}
        assert json.loads(temporary.read_text()) == record()
        assert stat.S_IMODE(temporary.stat().st_mode) == 0o600
        observed.append(temporary)
        return real_replace(temporary, destination)

    monkeypatch.setattr(Path, 'replace', checked_replace)
    manager.private_write(path, record())
    assert observed and not observed[0].exists()
    assert json.loads(path.read_text()) == record()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.parametrize('content', [None, '', '{broken', '[]', 'null'])
def test_absent_or_malformed_config_is_not_a_running_service(tmp_path, content):
    path = tmp_path / 'evidence-host.json'
    if content is not None:
        path.write_text(content)
    assert manager.load(path) == {}


@pytest.mark.parametrize('pid', [None, True, 0, 1, '201'])
def test_birth_rejects_nonprocess_identifiers_without_running_ps(monkeypatch, pid):
    monkeypatch.setattr(manager.subprocess, 'run',
                        lambda *a, **kw: pytest.fail('Invalid PID must not invoke ps'))
    assert manager.birth(pid) is None


@pytest.mark.parametrize('code, output, expected', [
    (0, 'S  Sun Sep 27 12:00:00 2026\n', 'Sun Sep 27 12:00:00 2026'),
    (0, 'Z  Sun Sep 27 12:00:00 2026\n', None),
    (1, '', None),
])
def test_birth_excludes_dead_and_zombie_processes(monkeypatch, code, output, expected):
    calls = []

    def ps(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=code, stdout=output)

    monkeypatch.setattr(manager.subprocess, 'run', ps)
    assert manager.birth(201) == expected
    assert calls == [['ps', '-p', '201', '-o', 'stat=', '-o', 'lstart=']]


def test_reused_or_missing_pid_is_not_owned_and_is_never_signalled(monkeypatch):
    monkeypatch.setattr(manager, 'birth', lambda pid: 'new-process-start' if pid == 201 else None)
    sent = []
    monkeypatch.setattr(manager.os, 'kill', lambda *args: sent.append(args))
    assert not manager.owned(record(), 'gateway')
    assert not manager.owned(record(), 'tunnel')
    manager.stop_owned(record())
    assert sent == []


def test_stop_terminates_owned_tunnel_before_gateway(monkeypatch):
    births = {201: 'gateway-start', 202: 'tunnel-start'}
    monkeypatch.setattr(manager, 'birth', births.get)
    sent = []

    def terminate(pid, sig):
        sent.append((pid, sig))
        births.pop(pid, None)

    monkeypatch.setattr(manager.os, 'kill', terminate)
    manager.stop_owned(record())
    assert sent == [(202, signal.SIGTERM), (201, signal.SIGTERM)]


def test_stop_escalates_only_a_still_owned_process_after_bounded_grace(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(manager, 'owned', lambda value, kind: kind == 'tunnel')
    monkeypatch.setattr(manager.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(manager.time, 'sleep', lambda seconds: now.__setitem__(0, now[0] + seconds))
    sent = []
    monkeypatch.setattr(manager.os, 'kill', lambda *args: sent.append(args))
    manager.stop_owned(record())
    assert sent == [(202, signal.SIGTERM), (202, signal.SIGKILL)]
    assert 8 <= now[0] < 8.2


class Socket:
    def __init__(self, occupied=False):
        self.occupied = occupied
        self.address = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def bind(self, address):
        self.address = address
        if self.occupied:
            raise OSError(errno.EADDRINUSE, 'occupied')


def test_occupied_port_refuses_start_without_launch_or_kill(tmp_path, monkeypatch):
    sock = Socket(occupied=True)
    monkeypatch.setattr(manager.socket, 'socket', lambda: sock)
    monkeypatch.setattr(manager, 'launch', lambda *a: pytest.fail('Must not launch on occupied port'))
    monkeypatch.setattr(manager.os, 'kill', lambda *a: pytest.fail('Must not stop the port occupant'))
    with pytest.raises(RuntimeError, match='occupied; no existing process was changed'):
        manager.start(tmp_path, 8768, tmp_path / 'evidence-host.json')
    assert sock.address == ('127.0.0.1', 8768)
    assert not (tmp_path / 'evidence-host.json').exists()


def test_partial_owned_service_requires_explicit_stop(tmp_path, monkeypatch):
    path = tmp_path / 'evidence-host.json'
    manager.private_write(path, record())
    monkeypatch.setattr(manager, 'owned', lambda value, kind: kind == 'gateway')
    monkeypatch.setattr(manager, 'launch', lambda *a: pytest.fail('Do not replace a partial service'))
    with pytest.raises(RuntimeError, match='partial evidence service'):
        manager.start(tmp_path, 8768, path)
    assert manager.load(path) == record()


def test_launch_is_detached_with_private_fresh_log_and_no_stdin(tmp_path, monkeypatch):
    path = tmp_path / 'gateway.log'
    path.write_text('old request token must not remain')
    path.chmod(0o644)
    expected = object()

    def popen(command, **kwargs):
        assert command == ['fixture-server']
        assert kwargs['cwd'] == manager.ROOT
        assert kwargs['env'] == {'FIXTURE': 'yes'}
        assert kwargs['stdin'] == subprocess.DEVNULL
        assert kwargs['start_new_session'] is True
        assert kwargs['stdout'] is kwargs['stderr']
        assert stat.S_IMODE(os.fstat(kwargs['stdout'].fileno()).st_mode) == 0o600
        return expected

    monkeypatch.setattr(manager.subprocess, 'Popen', popen)
    assert manager.launch(['fixture-server'], path, {'FIXTURE': 'yes'}) is expected
    assert path.read_bytes() == b''
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


class Child:
    def __init__(self, pid, exited=None):
        self.pid, self.exited = pid, exited
        self.terminated = self.killed = self.waited = False

    def poll(self):
        return self.exited

    def terminate(self):
        self.terminated = True
        self.exited = -signal.SIGTERM

    def kill(self):
        self.killed = True
        self.exited = -signal.SIGKILL

    def wait(self, timeout):
        self.waited = True
        return self.exited


def test_start_publishes_only_connected_healthy_tunnel_and_isolates_environment(tmp_path, monkeypatch, capsys):
    path = tmp_path / 'evidence-host.json'
    sock = Socket()
    monkeypatch.setattr(manager.socket, 'socket', lambda: sock)
    monkeypatch.setattr(manager.shutil, 'which', lambda command: '/fixture/cloudflared')
    monkeypatch.setenv('TUNNEL_URL', 'http://127.0.0.1:8765')
    monkeypatch.setenv('TWILIO_AUTH_TOKEN', 'fixture-not-a-secret')
    monkeypatch.setenv('VMD_EVIDENCE_BASE_URL', 'https://stale.example')
    children = [Child(201), Child(202)]
    launches, health_checks = [], []
    monkeypatch.setattr(manager, 'birth', lambda pid: f'birth-{pid}')

    def launch(command, log_path, env):
        launches.append((command, env))
        assert not path.exists(), 'Do not publish configuration during startup'
        if len(launches) == 2:
            log_path.write_text(f'{ORIGIN}\nRegistered tunnel connection\n')
        return children[len(launches) - 1]

    def health(url):
        health_checks.append(url)
        assert not path.exists(), 'Public readiness precedes publishing configuration'
        return True

    monkeypatch.setattr(manager, 'launch', launch)
    monkeypatch.setattr(manager, 'health', health)
    manager.start(tmp_path, 8768, path)
    saved = manager.load(path)
    assert saved['base_url'] == ORIGIN and saved['gateway_pid'] == 201 and saved['tunnel_pid'] == 202
    assert saved['gateway_birth'] == 'birth-201' and saved['tunnel_birth'] == 'birth-202'
    assert health_checks == ['http://127.0.0.1:8768', ORIGIN]
    gateway, tunnel = launches
    assert gateway[0][1:4] == ['-m', 'uvicorn', 'vmd.share:app']
    assert gateway[0][gateway[0].index('--host') + 1] == '127.0.0.1'
    assert gateway[0][gateway[0].index('--port') + 1] == '8768'
    assert '--no-access-log' in gateway[0] and '--no-proxy-headers' in gateway[0]
    assert tunnel[0][tunnel[0].index('--url') + 1] == 'http://127.0.0.1:8768'
    assert tunnel[0][tunnel[0].index('--config') + 1] == os.devnull
    assert tunnel[0][tunnel[0].index('--protocol') + 1] == 'http2'
    assert tunnel[0][tunnel[0].index('--metrics') + 1] == '127.0.0.1:0'
    assert '--no-autoupdate' in tunnel[0]
    for _, env in launches:
        assert env['VMD_DATA_DIR'] == str(tmp_path)
        assert not any(key.startswith(('TUNNEL_', 'TWILIO_', 'VMD_EVIDENCE_')) for key in env)
    assert json.loads(capsys.readouterr().out)['status'] == 'running'


def test_failed_tunnel_start_cleans_children_and_never_publishes_url(tmp_path, monkeypatch):
    path = tmp_path / 'evidence-host.json'
    monkeypatch.setattr(manager.socket, 'socket', Socket)
    monkeypatch.setattr(manager.shutil, 'which', lambda command: '/fixture/cloudflared')
    monkeypatch.setattr(manager, 'birth', lambda pid: f'birth-{pid}')
    monkeypatch.setattr(manager, 'health', lambda url: True)
    gateway, tunnel = Child(201), Child(202, exited=1)
    children = iter([gateway, tunnel])
    monkeypatch.setattr(manager, 'launch', lambda *args: next(children))
    stopped = []
    monkeypatch.setattr(manager, 'stop_owned', lambda value: stopped.append(value.copy()))
    with pytest.raises(RuntimeError, match='stopped during startup'):
        manager.start(tmp_path, 8768, path)
    assert stopped[0]['gateway_pid'] == 201 and stopped[0]['tunnel_pid'] == 202
    assert gateway.terminated and gateway.waited and tunnel.waited
    assert not path.exists()


def test_status_does_not_reuse_dead_services_saved_public_url(tmp_path, monkeypatch, capsys):
    path = tmp_path / 'evidence-host.json'
    manager.private_write(path, record())
    monkeypatch.setattr(manager, 'owned', lambda value, kind: False)
    monkeypatch.setattr(sys, 'argv', ['evidence_tunnel.py', 'status', '--data-dir', str(tmp_path)])
    assert manager.main() == 0
    assert json.loads(capsys.readouterr().out) == {'status': 'stopped', 'base_url': None, 'port': 8768}


def test_cli_stop_clears_public_configuration_before_terminating_owned_children(tmp_path, monkeypatch, capsys):
    path = tmp_path / 'evidence-host.json'
    manager.private_write(path, record())
    stopped = []

    def stop(value):
        assert not path.exists()
        stopped.append(value)

    monkeypatch.setattr(manager, 'stop_owned', stop)
    monkeypatch.setattr(sys, 'argv', ['evidence_tunnel.py', 'stop', '--data-dir', str(tmp_path)])
    assert manager.main() == 0
    assert stopped == [record()]
    assert json.loads(capsys.readouterr().out) == {'status': 'stopped'}


@pytest.mark.parametrize('port', [80, 8765, 8770, 8781, 65536])
def test_cli_rejects_application_or_invalid_ports_before_start(tmp_path, monkeypatch, port):
    monkeypatch.setattr(sys, 'argv', ['evidence_tunnel.py', 'start', '--data-dir', str(tmp_path), '--port', str(port)])
    monkeypatch.setattr(manager, 'start', lambda *args: pytest.fail('Reserved port must not start'))
    with pytest.raises(SystemExit) as result:
        manager.main()
    assert result.value.code == 2
    assert not (tmp_path / 'evidence-host.json').exists()


def test_isolated_status_cli_needs_no_server_or_cloudflare(tmp_path):
    directory = tmp_path / 'isolated-data'
    result = subprocess.run([sys.executable, str(manager.ROOT / 'scripts/evidence_tunnel.py'),
                             'status', '--data-dir', str(directory)],
                            capture_output=True, text=True, timeout=10, check=True)
    assert json.loads(result.stdout) == {'status': 'stopped', 'base_url': None, 'port': 8768}
    assert not (directory / 'evidence-host.json').exists()
    assert list(directory.iterdir()) == [directory / 'evidence-host.lock']
