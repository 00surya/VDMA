#!/usr/bin/env python3
"""Rebuild the README's charts with Matplotlib; no app or provider is started.

Run from any directory: .venv/bin/python scripts/render_readme_charts.py
Matplotlib is a documentation tool here, not an added application dependency.

Candidate metrics are the user's supplied five-class experiment results:
https://github.com/00surya/vdm-shield/blob/main/docs/weapon-ordnance-experiment.md
They describe its prepared public-image test split, not the active detector or
CCTV performance. Eco intervals are read directly from vmd/eco.py; 8/1/1 Hz
is an explicitly configured example, not a throughput measurement.
"""
import ast
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'tmp' / 'readme-chart-cache'
CACHE.mkdir(parents=True, exist_ok=True)
os.environ['MPLCONFIGDIR'] = str(CACHE)

import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch


# Supplied results: 1,434 held-out public images; 30 + 20 epochs with a fresh
# optimizer for phase two. This five-class candidate is not deployed in VDMA.
CANDIDATE_METRICS = {
    'Precision': 72.10,
    'Recall': 65.84,
    'mAP@50': 71.35,
    'mAP@50–95': 50.32,
}
CONFIGURED_EXAMPLE = {'pose': 8.0, 'depth': 1.0, 'objects': 1.0}
COLORS = {
    'background': '#0C1320', 'panel': '#142030', 'grid': '#2A3A4E',
    'text': '#EDF5F8', 'muted': '#A8BACB', 'teal': '#5ED6C6',
    'lime': '#B9DEA1', 'track': '#1C2A3D',
}
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 13,
    'text.color': COLORS['text'], 'axes.labelcolor': COLORS['muted'],
    'xtick.color': COLORS['muted'], 'ytick.color': COLORS['text'],
    'svg.fonttype': 'none', 'svg.hashsalt': 'vdma-readme-charts-v1',
})


def eco_settings():
    """Read literal EcoGate constants without importing the application."""
    tree = ast.parse((ROOT / 'vmd' / 'eco.py').read_text())
    gate = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                and node.name == 'EcoGate')
    constants = {}
    for node in gate.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id.startswith(('QUIET_', 'ACTIVE_')):
                constants[target.id] = ast.literal_eval(node.value)
    intervals = {'pose': 'QUIET_POSE_INTERVAL', 'depth': 'QUIET_DEPTH_INTERVAL',
                 'objects': 'QUIET_OBJECT_INTERVAL'}
    if any(constants[name] <= 0 for name in intervals.values()):
        raise ValueError('Eco intervals must be positive')
    quiet = {worker: min(CONFIGURED_EXAMPLE[worker], 1 / constants[name])
             for worker, name in intervals.items()}
    return constants['ACTIVE_HOLD_SECONDS'], quiet


def canvas(tag, title, subtitle, notes):
    figure = plt.figure(figsize=(10.8, 6.4), facecolor=COLORS['background'])
    figure.text(.08, .952, tag, color=COLORS['teal'], fontsize=10,
                weight='bold', va='top')
    figure.text(.08, .904, title, fontsize=23, weight='bold', va='top')
    figure.text(.08, .826, subtitle, fontsize=13, color=COLORS['muted'], va='top')
    figure.add_artist(FancyBboxPatch((.07, .048), .86, .122,
        boxstyle='round,pad=0.012,rounding_size=0.014', transform=figure.transFigure,
        facecolor=COLORS['panel'], edgecolor='none', zorder=0))
    figure.text(.09, .128, notes[0], fontsize=12, weight='bold', va='top')
    figure.text(.09, .084, notes[1], fontsize=11, color=COLORS['muted'], va='top')
    return figure


def axis_style(axis, maximum, ticks, label):
    axis.set_facecolor(COLORS['background'])
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_xlim(0, maximum)
    axis.set_xticks(ticks)
    axis.set_xlabel(label, fontsize=12, labelpad=13)
    axis.tick_params(axis='both', length=0, pad=10)
    axis.grid(axis='x', color=COLORS['grid'], linewidth=.7, alpha=.8)
    axis.set_axisbelow(True)


def export(figure, stem, description):
    destination = ROOT / 'docs' / 'media'
    destination.mkdir(parents=True, exist_ok=True)
    title = stem.replace('-', ' ')
    figure.savefig(destination / f'{stem}.png', dpi=180,
                   facecolor=COLORS['background'], metadata={
                       'Software': 'VDMA README chart renderer', 'Description': description})
    figure.savefig(destination / f'{stem}.svg', facecolor=COLORS['background'],
                   metadata={'Title': title, 'Description': description,
                             'Creator': 'scripts/render_readme_charts.py', 'Date': None})
    svg = destination / f'{stem}.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(figure)
    print(f'docs/media/{stem}.png + .svg')


def candidate_chart():
    figure = canvas('SEPARATE MODEL EXPERIMENT', 'Five-class candidate: public-image test',
        '1,434 test images  ·  30 + 20 training epochs  ·  fresh optimizer in phase two',
        ('Not the active VDMA detector.',
         'Public-image results do not establish CCTV accuracy or false alarms per camera-hour.'))
    axis = figure.add_axes((.23, .31, .67, .43))
    names, values = list(CANDIDATE_METRICS), list(CANDIDATE_METRICS.values())
    positions = list(range(len(names)))
    axis_style(axis, 100, [0, 25, 50, 75, 100], 'Score (%)  ·  full 0–100% scale')
    axis.barh(positions, [100] * len(names), height=.53, color=COLORS['track'], zorder=1)
    axis.barh(positions, values, height=.53, color=COLORS['teal'], zorder=2)
    axis.set_yticks(positions, names, fontsize=14)
    axis.set_ylim(len(names) - .5, -.5)
    for position, value in zip(positions, values):
        axis.text(value + 2, position, f'{value:.2f}%', fontsize=15,
                  weight='bold', va='center', zorder=3)
    export(figure, 'candidate-test-metrics',
           'Separate five-class candidate; 1,434 public-image test images; 30 + 20 epochs. '
           'Precision 72.10%, recall 65.84%, mAP@50 71.35%, mAP@50–95 50.32%. '
           'Not the active detector or a CCTV accuracy benchmark.')


def eco_chart():
    hold, quiet = eco_settings()
    figure = canvas('SCHEDULING EXAMPLE', 'Eco changes the sampling cadence',
        'Configured example: pose 8/s  ·  depth 1/s  ·  objects 1/s',
        (f'Quiet-scene targets after {hold:g}s; every rate remains capped by its setting.',
         'Configured cadence is not measured speed, RAM use, energy savings or accuracy.'))
    axis = figure.add_axes((.18, .305, .72, .405))
    workers, labels = list(CONFIGURED_EXAMPLE), ['Pose', 'Depth', 'Objects']
    positions = list(range(len(workers)))
    axis_style(axis, 9, [0, 2, 4, 6, 8], 'Configured checks per second')
    for index, worker in enumerate(workers):
        for offset, value, color, name in (
            (-.17, CONFIGURED_EXAMPLE[worker], COLORS['teal'], 'Configured active'),
            (.17, quiet[worker], COLORS['lime'], 'Quiet Eco target')):
            axis.barh(index + offset, value, height=.255, color=color,
                      label=name if index == 0 else None, zorder=2)
            axis.text(value + .14, index + offset, f'{value:g}/s', fontsize=14,
                      weight='bold', va='center', zorder=3)
    axis.set_yticks(positions, labels, fontsize=14)
    axis.set_ylim(len(workers) - .55, -.55)
    axis.legend(loc='lower left', bbox_to_anchor=(0, 1.025), ncol=2,
                frameon=False, fontsize=11, handlelength=1.2, columnspacing=2)
    export(figure, 'eco-sampling-cadence',
           f'Configured example: pose 8/s, depth 1/s, objects 1/s. After {hold:g} quiet seconds, '
           f'Eco targets pose {quiet["pose"]:g}/s, depth {quiet["depth"]:g}/s, '
           f'objects {quiet["objects"]:g}/s, capped by configured rates. '
           'These are scheduling settings, not measured speed, memory, energy or accuracy.')


if __name__ == '__main__':
    candidate_chart()
    eco_chart()
