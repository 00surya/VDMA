"""Launcher wiring and cleanup; no network or real credentials."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from scripts import start_mobile


@pytest.mark.parametrize('failed', [False, True])
def test_launcher_keeps_password_in_relay_and_cleans_up(failed):
    relay = MagicMock()
    relay_module = SimpleNamespace(local_ipv4=lambda _: True, Relay=MagicMock())
    relay_module.Relay.return_value.__enter__.return_value = relay
    thread = MagicMock()
    run = MagicMock(return_value=SimpleNamespace(returncode=0))
    if failed:
        run.side_effect = OSError('sample spawn failure')
    with patch('sys.argv', ['start_mobile.py', '--host', '192.168.1.10']), \
         patch.object(start_mobile.Path, 'exists', return_value=True), \
         patch.object(start_mobile.getpass, 'getpass', return_value='fictional-test-password'), \
         patch.object(start_mobile.importlib.util, 'spec_from_file_location', return_value=MagicMock()), \
         patch.object(start_mobile.importlib.util, 'module_from_spec', return_value=relay_module), \
         patch.object(start_mobile.threading, 'Thread', return_value=thread), \
         patch.object(start_mobile.subprocess, 'run', run):
        if failed:
            with pytest.raises(SystemExit) as error:
                start_mobile.main()
            assert error.value.code == 1
        else:
            assert start_mobile.main() == 0
    relay_module.Relay.assert_called_once_with(('192.168.1.10', 8767), 'fictional-test-password')
    relay.login.assert_called_once()
    relay.shutdown.assert_called_once()
    thread.join.assert_called_once_with(timeout=5)
    command = run.call_args.args[0]
    assert command[1:] == ['start', '--go', '--lan', '--port', '8081']
    env = run.call_args.kwargs['env']
    assert env['EXPO_PUBLIC_VMD_SERVER_URL'] == 'http://192.168.1.10:8767'
    assert env['REACT_NATIVE_PACKAGER_HOSTNAME'] == '192.168.1.10'
    assert 'fictional-test-password' not in str(run.call_args)
