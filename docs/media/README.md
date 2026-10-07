# README media

These are the video and original screenshots selected for the public VDMA README. Runtime recordings, private camera data, model binaries and presentation files remain excluded from Git.

| Asset | What it shows |
| --- | --- |
| `ev1.mp4` | Supplied VDMA demo; 19.7 seconds, 2520 × 1080, 30 FPS. It shows video analysis, sampled depth and a possible-snatching review alert. It is not an accuracy benchmark. A GitHub-hosted copy provides inline README playback. |
| `dashboard.png` | Original 5 October 2026 presentation screenshot at 8:04:28 PM: camera tiles and incident inspector. |
| `pose-depth.png` | Original presentation screenshot at 8:02:16 PM: pose tracks and sampled relative depth. |
| `incident-player.png` | Original presentation screenshot at 8:02:48 PM: saved evidence playback. |
| `pose-review.png` | Original presentation screenshot at 8:01:10 PM: a possible-fight review warning. |

Screenshot PNGs are losslessly re-encoded with their original colour profiles retained; their dimensions and visible pixels are unchanged. Screenshots reflect that prototype session, so visible warnings/settings are not a claim about every scene or the current configuration.

## Charts

- **Candidate test metrics:** supplied results from the [separate five-class experiment](https://github.com/00surya/vdm-shield/blob/main/docs/weapon-ordnance-experiment.md), with a full 0–100% scale. The 1,434-image public test split gives precision 72.10%, recall 65.84%, mAP@50 71.35% and mAP@50–95 50.32%. It is not the active application detector and does not establish CCTV performance.
- **Eco sampling cadence:** a configured example of pose/depth/object checks at 8/1/1 per second. Quiet targets are read from `EcoGate` in [`vmd/eco.py`](../../vmd/eco.py): 2/0.1/0.5 after 10 quiet seconds, capped by each setting. This is a scheduling illustration, not a speed, RAM, energy or accuracy measurement.

PNG files are embedded in the README; SVG files are available for reuse. To rebuild both charts from the repository root, use Python with Matplotlib installed:

```sh
.venv/bin/python scripts/render_readme_charts.py
```

Matplotlib is only needed to rebuild the documentation assets. It is not added as an application dependency. The renderer imports no VDMA runtime and makes no provider requests.
