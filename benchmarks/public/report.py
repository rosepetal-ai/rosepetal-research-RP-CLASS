# RP-CLASS - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""The public benchmark's tables (docs/BENCHMARK.md, Metrics), from benchmarks/public/benchmark.py's runs.

    python3 benchmarks/public/report.py <prepared> <out>

Every (task, method, seed) is scored once, from its predictions.csv: macro-F1 (the primary metric), balanced
accuracy, top-1 and, for a binary inspection task, the AUROC of P(positive), the positive class its info.json names.
A dataset's score for a seed is the mean over its tasks (VisA's 12 objects, DAGM's 10 textures count once); the
tables give the mean ± standard deviation over the seeds. The headline is the unweighted mean over the datasets, with
the general (G1-G7) and the industrial (I1-I7) means apart. A method enters a mean only where it has every seed of
every task: a partial benchmark says so instead of averaging what it has.
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))
from rpclass.utils.metrics import evaluate  # noqa: E402

from summary import auc  # noqa: E402

GENERAL = ('eurosat', 'caltech101', 'nct_crc_he', 'bloodmnist', 'deepweeds', 'beans', 'clevr_count')
INDUSTRIAL = ('visa', 'dagm', 'xsdd', 'rawfabrid', 'wood', 'tyres', 'sdnet')
BENCHMARK = GENERAL + INDUSTRIAL      # any other prepared dataset (the development suite): its tables, never a mean


def score(path, positive):
    """{metric: value} of one run's predictions.csv."""
    with open(path) as f:
        rows = list(csv.reader(f))
    names = rows[0][2:]
    labels = np.array([names.index(r[1]) for r in rows[1:]])
    probs = np.array([[float(x) for x in r[2:]] for r in rows[1:]])
    m = evaluate(np.log(np.clip(probs, 1e-12, 1)), labels, {i: n for i, n in enumerate(names)})
    out = {'macro_f1': m.macro_f1, 'balanced_accuracy': m.balanced_accuracy, 'top1': m.top1}
    if positive in names and len(names) == 2:
        k = names.index(positive)
        out['auroc'] = auc(probs[labels == k, k], probs[labels != k, k])
    return out


def collect(prepared, out):
    """{(dataset, task): {model: {seed: metrics}}} and the tasks prepared per dataset."""
    tasks = defaultdict(list)
    for info in sorted(Path(prepared).glob('*/*/info.json')):
        tasks[info.parts[-3]].append(info.parent.name)
    results = defaultdict(lambda: defaultdict(dict))
    for pred in sorted(Path(out).glob('*/*/*/s*/predictions.csv')):
        dataset, task, model, seed = pred.parts[-5:-1]
        info = json.loads((Path(prepared) / dataset / task / 'info.json').read_text())
        results[dataset, task][model][int(seed[1:])] = score(pred, info.get('positive'))
    return results, tasks


def per_dataset(results, tasks, metric):
    """{dataset: {model: {seed: mean over its tasks}}}, complete datasets only."""
    out = defaultdict(dict)
    models = {m for r in results.values() for m in r}
    for dataset, names in tasks.items():
        for model in models:
            seeds = [set(results[dataset, t].get(model, {})) for t in names]
            common = set.intersection(*seeds) if seeds else set()
            if not common or any(metric not in results[dataset, t][model][s] for t in names for s in common):
                continue
            out[dataset][model] = {s: float(np.mean([results[dataset, t][model][s][metric] for t in names]))
                                   for s in sorted(common)}
    return out


def fmt(values):
    v = np.array(list(values)) * 100
    return f'{v.mean():.1f} ± {v.std():.1f}' if len(v) > 1 else f'{v.mean():.1f}'


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('prepared')
    p.add_argument('out')
    a = p.parse_args()
    results, tasks = collect(a.prepared, a.out)
    by_dataset = per_dataset(results, tasks, 'macro_f1')
    models = sorted({m for r in results.values() for m in r})
    benchmark = [d for d in tasks if d in BENCHMARK]
    if benchmark:
        headline(by_dataset, tasks, models, benchmark)
    for metric in ('macro_f1', 'auroc'):
        table = per_dataset(results, tasks, metric)
        if not table:
            continue
        shown = [d for d in BENCHMARK if d in table] + sorted(d for d in table if d not in BENCHMARK)
        print(f'\n## {metric} per dataset\n')
        print('| method | ' + ' | '.join(shown) + ' |\n|---|' + '---|' * len(shown))
        for model in models:
            print(f'| {model} | ' + ' | '.join(fmt(table[d][model].values()) if model in table[d] else '-'
                                                for d in shown) + ' |')
    multi = [d for d in tasks if len(tasks[d]) > 1]
    for dataset in multi:
        print(f'\n## {dataset}: macro-F1 per task\n')
        print('| method | ' + ' | '.join(tasks[dataset]) + ' |\n|---|' + '---|' * len(tasks[dataset]))
        for model in models:
            cells = [fmt(m['macro_f1'] for m in results[dataset, t][model].values()) if model in results[dataset, t]
                     else '-' for t in tasks[dataset]]
            print(f'| {model} | ' + ' | '.join(cells) + ' |')


def headline(by_dataset, tasks, models, benchmark):
    print('## Headline: macro-F1, the unweighted mean over the datasets (mean ± std over seeds)\n')
    print('| method | all | general | industrial | datasets |\n|---|---|---|---|---|')
    for model in models:
        cells = []
        for group in (GENERAL + INDUSTRIAL, GENERAL, INDUSTRIAL):
            have = [d for d in group if model in by_dataset.get(d, {})]
            seeds = set.intersection(*(set(by_dataset[d][model]) for d in have)) if have else set()
            if len(have) < len([d for d in group if d in tasks]) or not seeds:
                cells.append('incomplete')
                continue
            cells.append(fmt(np.mean([by_dataset[d][model][s] for d in have]) for s in sorted(seeds)))
        print(f"| {model} | {' | '.join(cells)} | {sum(model in by_dataset.get(d, {}) for d in benchmark)} |")


if __name__ == '__main__':
    main()
