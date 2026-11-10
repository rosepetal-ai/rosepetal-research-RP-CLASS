# RP-CLASS - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""The README's charts of the public benchmark, from the scores of every run.

    python3 benchmarks/public/plots.py [benchmarks/public/results/v1/scores.csv] [docs/images]

scores.csv is what `benchmarks/public/report.py --scores` writes, from the same predictions.csv files as the report's
tables. A dataset's score for a seed is the mean over its tasks; a group's is the mean over its datasets.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from report import BENCHMARK, GENERAL, INDUSTRIAL  # noqa: E402

METHODS = {'vit-s16-dinov3+d4': 'RP-CLASS lt', 'hgnetv2-b2+d4': 'RP-CLASS m', 'yolo11s-cls': 'YOLO11s-cls'}
COLORS = {'vit-s16-dinov3+d4': '#2a78d6', 'hgnetv2-b2+d4': '#eb6834', 'yolo11s-cls': '#1baf7a'}   # validated, fixed
NAMES = {'eurosat': 'EuroSAT', 'caltech101': 'Caltech-101', 'nct_crc_he': 'NCT-CRC-HE', 'bloodmnist': 'BloodMNIST',
         'deepweeds': 'DeepWeeds', 'beans': 'Beans', 'clevr_count': 'CLEVR count', 'visa': 'VisA', 'dagm': 'DAGM',
         'xsdd': 'X-SDD', 'rawfabrid': 'RAW-FABRID', 'wood': 'Wood', 'tyres': 'Tyres', 'sdnet': 'SDNET2018'}
SURFACE, INK, INK_2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
SUBTITLE = 'Public benchmark v1 · 14 public datasets, each cut to about 300 training images · 3 seeds per method'


def load(path):
    """{method: {dataset: {seed: macro-F1 in %}}}, each dataset the mean of its tasks."""
    tasks = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            tasks[r['method'], r['dataset'], int(r['seed'])].append(100 * float(r['macro_f1']))
    out = defaultdict(lambda: defaultdict(dict))
    for (method, dataset, seed), values in tasks.items():
        out[method][dataset][seed] = float(np.mean(values))
    return out


def group(scores, method, datasets):
    """Per seed, the mean over the datasets."""
    seeds = sorted(set.intersection(*(set(scores[method][d]) for d in datasets)))
    return np.array([np.mean([scores[method][d][s] for d in datasets]) for s in seeds])


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelcolor=INK)
    ax.grid(axis='x', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def titles(fig, title, subtitle):
    fig.text(0.02, 0.965, title, fontsize=15, fontweight='bold', color=INK, va='top')
    fig.text(0.02, 0.905, subtitle, fontsize=9.5, color=INK_2, va='top')


def legend(fig, y):
    handles = [plt.Line2D([], [], marker='o', linestyle='', markersize=9, color=COLORS[m], label=METHODS[m])
               for m in METHODS]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.015, y), ncol=3, frameon=False, fontsize=10,
               labelcolor=INK, handletextpad=0.3, columnspacing=1.6)


def chart_groups(scores, out):
    """Grouped bars: all 14, general, industrial; bar = mean, line = std, dots = seeds."""
    groups = [('All 14 datasets', BENCHMARK), ('General (7)', GENERAL), ('Industrial (7)', INDUSTRIAL)]
    fig, ax = plt.subplots(figsize=(8.6, 5.0), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    height, gap = 0.25, 0.04
    ticks = []
    for g, (label, datasets) in enumerate(groups):
        centre = g * 1.15
        ticks.append(centre)
        for k, method in enumerate(METHODS):
            y = centre + (k - 1) * (height + gap)
            seeds = group(scores, method, datasets)
            mean, std = seeds.mean(), seeds.std()
            ax.barh(y, mean, height=height, color=COLORS[method], edgecolor=SURFACE, linewidth=2)
            ax.errorbar(mean, y, xerr=std, color=INK, elinewidth=1.2, capsize=3, capthick=1.2, zorder=3)
            ax.scatter(seeds, [y] * len(seeds), s=18, facecolor=SURFACE, edgecolor=INK, linewidth=0.9, zorder=4)
            ax.text(101.5, y, f'{mean:.1f} ± {std:.1f}', va='center', ha='left', fontsize=9.5, color=INK,
                    fontweight='bold' if method != 'yolo11s-cls' else 'normal')
    ax.set_yticks(ticks, [label for label, _ in groups], fontsize=10.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel('Macro-F1 (%), the mean over the datasets · bar = mean, line = std over seeds, dots = seeds',
                  fontsize=9.5, color=INK_2)
    titles(fig, 'Macro-F1 on the public benchmark', SUBTITLE)
    legend(fig, 0.86)
    fig.subplots_adjust(left=0.17, right=0.86, top=0.76, bottom=0.12)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def chart_datasets(scores, out):
    """One row per dataset, one dot per method (mean over seeds), a thin line from the lowest to the highest."""
    rows = list(GENERAL) + [None] + list(INDUSTRIAL)
    fig, ax = plt.subplots(figsize=(8.6, 7.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    offsets = dict(zip(METHODS, (-0.17, 0.0, 0.17)))
    labels, ys = [], []
    for i, dataset in enumerate(rows):
        if dataset is None:
            continue
        means = {m: float(np.mean(list(scores[m][dataset].values()))) for m in METHODS}
        ax.plot([min(means.values()), max(means.values())], [i, i], color=GRID, linewidth=2, zorder=1,
                solid_capstyle='round')
        for method, value in means.items():
            ax.scatter(value, i + offsets[method], s=70, color=COLORS[method], edgecolor=SURFACE, linewidth=1.5,
                       zorder=3)
        labels.append(NAMES[dataset])
        ys.append(i)
    ax.set_yticks(ys, labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(30, 101)
    ax.set_ylim(len(rows) - 0.4, -1.4)
    for text, y in (('General', -0.95), ('Industrial', len(GENERAL) + 0.05)):
        ax.text(30.4, y, text.upper(), fontsize=8.5, color=INK_2, fontweight='bold', va='center')
    ax.set_xlabel('Macro-F1 (%), mean over 3 seeds · axis from 30', fontsize=9.5, color=INK_2)
    titles(fig, 'Macro-F1 per dataset', SUBTITLE)
    legend(fig, 0.875)
    fig.subplots_adjust(left=0.17, right=0.97, top=0.82, bottom=0.08)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def main():
    here = Path(__file__).resolve().parent
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else here / 'results' / 'v1' / 'scores.csv'
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else here.parents[1] / 'docs' / 'images'
    out.mkdir(parents=True, exist_ok=True)
    scores = load(src)
    chart_groups(scores, out / 'benchmark_macro_f1.png')
    chart_datasets(scores, out / 'benchmark_per_dataset.png')
    print(f'wrote {out}/benchmark_macro_f1.png and benchmark_per_dataset.png')


if __name__ == '__main__':
    main()
