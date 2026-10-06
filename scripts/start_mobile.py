"""Run the installed Expo app against this Mac's existing mobile gateway."""
import argparse
import getpass
import importlib.util
import os
from pathlib import Path
import subprocess
import threading
from urllib.error import HTTPError


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / 'android-app'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', help='Private Wi-Fi IPv4; defaults to macOS en0')
    args = parser.parse_args()
    expo = APP / 'node_modules/.bin/expo'
    relay_file = APP / 'scripts/lan_relay.py'
    if not expo.exists() or not relay_file.exists():
        parser.exit(1, 'Install android-app and run npm ci inside it first.\n')
    try:
        host = args.host or subprocess.check_output(
            ['/usr/sbin/ipconfig', 'getifaddr', 'en0'], text=True).strip()
        spec = importlib.util.spec_from_file_location('vmd_lan_relay', relay_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not module.local_ipv4(host) or host.startswith('127.'):
            raise ValueError('Use this Mac\'s private Wi-Fi IPv4 address.')
        print('Connect your Android phone to the same trusted Wi-Fi as this Mac.', flush=True)
        print('The relay shares read-only incident metadata with devices on that network.', flush=True)
        password = getpass.getpass('Existing VDMA centre password (hidden; stays on this Mac): ')
        with module.Relay((host, 8767), password) as relay:
            relay.login()
            password = None
            worker = threading.Thread(target=relay.serve_forever, daemon=True)
            worker.start()
            try:
                print(f'API connected: http://{host}:8767\nScan the upcoming QR code in Expo Go. Keep this window open.\n', flush=True)
                env = {**os.environ, 'EXPO_PUBLIC_VMD_SERVER_URL': f'http://{host}:8767',
                       'REACT_NATIVE_PACKAGER_HOSTNAME': host, 'EXPO_NO_TELEMETRY': '1'}
                result = subprocess.run([str(expo), 'start', '--go', '--lan', '--port', '8081'], cwd=APP, env=env)
                return result.returncode
            finally:
                relay.shutdown()
                worker.join(timeout=5)
    except KeyboardInterrupt:
        return 0
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        status = f' (HTTP {error.code})' if isinstance(error, HTTPError) else ''
        parser.exit(1, f'Could not start mobile app{status}. Check Wi-Fi, the centre password, and gateway 127.0.0.1:8766. Ports 8767 and 8081 must be free.\n')


if __name__ == '__main__':
    raise SystemExit(main())
