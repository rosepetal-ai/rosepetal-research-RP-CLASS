# RP-CLASS - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""Out-of-fold results of a benchmark (benchmarks/run.py), one table per dataset.

The test predictions of the K folds are pooled (every image once) and scored as one split: macro-F1, top-1 and
balanced accuracy, mean ± std over seeds. For a binary OK / NOK dataset also the AUC of P(NOK), and how many OK images a
threshold must reject to catch every NOK (the operating point inspection cares about; RP-CLASS itself returns the most
probable class, this only says how separable the scores are).

    python3 benchmarks/summary.py <benchmark out> [--positive nok]
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rpclass.utils.metrics import evaluate  # noqa: E402


def auc(pos, neg):
    """P(a random positive scores above a random negative), ties counting half (Mann-Whitney)."""
    scores = np.concatenate([pos, neg])
    ranks = np.empty(len(scores))
    order = np.argsort(scores, kind='mergesort')
    ranks[order] = np.arange(1, len(scores) + 1)
    for v in np.unique(scores):                 # average ranks of ties
        tie = scores == v
        ranks[tie] = ranks[tie].mean()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def pooled(run_dirs):
    """All the folds of one (dataset, model, seed): labels, probabilities and class names."""
    labels, probs, names = [], [], None
    for d in run_dirs:
        with open(d / 'predictions.csv') as f:
            rows = list(csv.reader(f))
        names = rows[0][2:]
        labels += [names.index(r[1]) for r in rows[1:]]
        probs += [[float(x) for x in r[2:]] for r in rows[1:]]
    return np.array(labels), np.array(probs), names


def scores(labels, probs, names, positive):
    m = evaluate(np.log(np.clip(probs, 1e-12, 1)), labels, dict(enumerate(names)))
    out = {'macro_f1': m.macro_f1, 'top1': m.top1, 'balanced_accuracy': m.balanced_accuracy}
    if len(names) == 2 and positive in names:
        p = probs[:, names.index(positive)]
        pos, neg = p[labels == names.index(positive)], p[labels != names.index(positive)]
        out['auc'] = auc(pos, neg)
        out['rejected_at_full_recall'] = int((neg >= pos.min()).sum()) if len(pos) else 0
        out['negatives'] = len(neg)
    return out


def run_stats(out):
    """(dataset, model) -> mean seconds per run and mean best epoch, from runs.csv."""
    stats = {}
    path = Path(out) / 'runs.csv'
    if not path.exists():
        return stats
    rows = [r for r in csv.DictReader(open(path)) if r.get('dataset') not in (None, 'dataset')]
    for r in rows:
        stats.setdefault((r['dataset'], r['model']), []).append(r)
    return {k: (np.mean([float(r['seconds']) for r in rs]),
                np.mean([float(r['best_epoch']) for r in rs if r['best_epoch']]) if any(r['best_epoch'] for r in rs)
                else float('nan')) for k, rs in stats.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('out')
    p.add_argument('--positive', default='nok')
    p.add_argument('--folds', type=int, default=5, help='a seed counts only once all its folds are done')
    a = p.parse_args()
    out = Path(a.out)
    groups = {}
    for pred in sorted(out.glob('*/*/s*/fold*/predictions.csv')):
        dataset, model, seed = pred.parts[-5], pred.parts[-4], pred.parts[-3]
        groups.setdefault((dataset, model), {}).setdefault(seed, []).append(pred.parent)
    timing = run_stats(out)
    skipped = []
    for dataset in sorted({d for d, _ in groups}):
        print(f'\n### {dataset}\n')
        rows = []
        for (d, model), seeds in sorted(groups.items()):
            if d != dataset:
                continue
            done = {s: dirs for s, dirs in seeds.items() if len(dirs) >= a.folds}
            skipped += [f'{d}/{model}/{s} ({len(dirs)}/{a.folds} folds)' for s, dirs in seeds.items() if s not in done]
            if done:
                rows.append((model, sorted(done), [scores(*pooled(dirs), a.positive) for dirs in done.values()]))
        keys = [k for k in ('macro_f1', 'top1', 'balanced_accuracy', 'auc', 'rejected_at_full_recall')
                if all(k in s for _, _, ps in rows for s in ps)]
        print('| model | seeds | ' + ' | '.join(keys) + ' | s / run | best epoch |')
        print('|---' * (4 + len(keys)) + '|')
        for model, seeds, ps in sorted(rows, key=lambda r: -np.mean([s['macro_f1'] for s in r[2]])):
            cells = []
            for k in keys:
                v = np.array([s[k] for s in ps], float)
                cells.append(f'{v.mean():.1f} ± {v.std():.1f} / {ps[0]["negatives"]}' if k == 'rejected_at_full_recall'
                             else f'{100 * v.mean():.1f} ± {100 * v.std():.1f}')
            sec, best = timing.get((dataset, model), (float('nan'), float('nan')))
            print(f'| {model} | {len(seeds)} | ' + ' | '.join(cells) + f' | {sec:.0f} | {best:.0f} |')
    if skipped:
        print('\nnot counted yet (incomplete seeds): ' + ', '.join(skipped))


if __name__ == '__main__':
    main()
