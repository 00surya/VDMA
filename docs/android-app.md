# Android app through Expo Go

The separate [Android repository](https://github.com/omansharora5/vmd-shield-android-app)
is installed in `android-app/` at revision `9c40410`. It has its own Git history and
is ignored by this repository. Install its locked dependencies with `npm ci` there.

On this Mac, keep the existing mobile gateway running on `127.0.0.1:8766`, then
double-click `scripts/start-mobile.command`, or run:

```sh
.venv/bin/python scripts/start_mobile.py
```

Enter the existing centre password in the Terminal prompt. It is hidden and held
only in relay memory; it is never embedded in the app. The launcher detects the
Mac's Wi-Fi address, starts the app's read-only relay on port 8767, and launches
Expo Go development on port 8081 with the relay address preconfigured. If Wi-Fi
uses an interface other than en0, pass `--host YOUR_PRIVATE_WIFI_IPV4`.

Install a compatible Expo Go client on Android, connect to the same trusted Wi-Fi,
and scan the terminal QR code. Keep that terminal and the existing VDMA gateway
running. Ctrl-C stops the Expo server and relay. Restart the launcher if the
Mac's Wi-Fi IP changes. Browser preview is `http://localhost:8081`.

The path is phone → Wi-Fi relay (8767) → authenticated mobile API (8766) → the
existing VDMA incident database. The dashboard (8765), evidence gateway (8768),
camera processing and calling stay separate. Anyone who can reach the relay on
the trusted LAN can read its incident metadata; do not publicly tunnel or
port-forward it. Map tiles require internet. The app shows live-camera incidents
on Map, and recordings and locationless events in Alerts. This version does not
request event-package video links or send background push notifications.

This is an Expo Go development installation, not an APK. Real-device operation
still requires checking the app on the user's phone. See the installed app's
`CONNECT-VMD.md` for its relay contract and `README.md` for build instructions.
