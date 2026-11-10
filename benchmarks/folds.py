# RP-CLASS - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""K-fold layouts for small datasets: every image is tested exactly once, and no group is split across folds.

A dataset of a few hundred images gives a test split of a few dozen, too few to compare two models on. So the
benchmark trains K times: for fold k, test = fold k, val = fold k+1 (it picks the best epoch), train = the rest; the
K test predictions together cover every image once (out-of-fold). Images of one group (the same part photographed in a
burst, several views of one object) always land in the same fold, or test would see what train saw. Folds are
stratified by class.

    python3 benchmarks/folds.py manifest.csv <images root> <out> [--folds 5] [--seed 0]

manifest.csv: columns path (relative to <images root>) and label (class name), and optionally group (images that
must share a fold; none: every image on its own), strata (what to balance across folds; none: the label) or fold
(folds already decided, used as they are). Writes <out>/fold<k>/{train,val,test}/<class>/<image> as hard links (copies
if links are impossible) and <out>/folds.csv.
"""
import argparse
import csv
import os
import shutil
from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold


def assign(rows, folds, seed):
    """rows: dicts with path, label and optionally group, strata, fold -> the fold of each row."""
    if all(r.get('fold') not in (None, '') for r in rows):
        return [int(r['fold']) for r in rows]
    strata = [r.get('strata') or r['label'] for r in rows]
    groups = [r.get('group') or f'image{i}' for i, r in enumerate(rows)]
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    out = [0] * len(rows)
    for k, (_, test) in enumerate(splitter.split(rows, strata, groups)):
        for i in test:
            out[i] = k
    return out


def _place(src, dst):
    if dst.exists():
        raise ValueError(f'two images would land on {dst}: give them unique names')
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build(rows, root, out, folds=5, seed=0):
    root, out = Path(root), Path(out)
    fold_of = assign(rows, folds, seed)
    for k in range(folds):
        for r, f in zip(rows, fold_of):
            split = 'test' if f == k else 'val' if f == (k + 1) % folds else 'train'
            name = Path(r['path']).name
            _place(root / r['path'], out / f'fold{k}' / split / r['label'] / name)
    with open(out / 'folds.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['path', 'label', 'group', 'fold'])
        w.writerows([r['path'], r['label'], r.get('group', ''), k] for r, k in zip(rows, fold_of))
    return fold_of


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('manifest')
    p.add_argument('root')
    p.add_argument('out')
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    with open(a.manifest) as f:
        rows = list(csv.DictReader(f))
    fold_of = build(rows, a.root, a.out, a.folds, a.seed)
    for k in range(a.folds):
        labels = sorted({r['label'] for r in rows})
        counts = {lab: sum(1 for r, f in zip(rows, fold_of) if f == k and r['label'] == lab) for lab in labels}
        print(f'fold {k}: ' + ', '.join(f'{lab} {n}' for lab, n in counts.items()))


if __name__ == '__main__':
    main()
