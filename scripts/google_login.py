#!/usr/bin/env python3
"""Explicit local OAuth consent for Vertex; never prints credentials."""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vmd.config import load_environment
load_environment()


def main():
    parser = argparse.ArgumentParser(description='Sign in locally for Vertex AI with your OAuth client')
    parser.add_argument('--port', type=int, default=8089)
    args = parser.parse_args()
    from google_auth_oauthlib.flow import InstalledAppFlow
    client_id, secret = os.getenv('GOOGLE_CLIENT_ID', ''), os.getenv('GOOGLE_CLIENT_SECRET', '')
    if not client_id.endswith('.apps.googleusercontent.com') or not secret or secret.startswith('xxxx'):
        raise SystemExit('Replace GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in .env first.')
    flow = InstalledAppFlow.from_client_config({'installed': {
        'client_id': client_id, 'client_secret': secret,
        'auth_uri': 'https://accounts.google.com/o/oauth2/auth',
        'token_uri': 'https://oauth2.googleapis.com/token',
        'redirect_uris': [f'http://localhost:{args.port}/']}},
        scopes=['https://www.googleapis.com/auth/cloud-platform'])
    credentials = flow.run_local_server(host='localhost', port=args.port, open_browser=True,
        authorization_prompt_message='Complete the Google sign-in in your browser.',
        success_message='Vertex sign-in saved locally. You can close this tab.', timeout_seconds=180)
    target = Path(os.getenv('VMD_GOOGLE_CREDENTIALS_FILE', str(ROOT/'data/google-credentials.json')))
    if not target.is_absolute():
        target = ROOT/target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.tmp')
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, 'w') as handle:
        handle.write(credentials.to_json())
    os.chmod(temporary, 0o600)
    temporary.replace(target)
    print('Google credentials saved privately. Restart VDMA to load the configured environment.')


if __name__ == '__main__':
    main()
