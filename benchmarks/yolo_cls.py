# RP-CLASS - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""The baseline: Ultralytics YOLO classification on the same k-fold datasets as benchmarks/run.py, with its own
defaults (recipe, augmentation, resize-and-center-crop pre-processing), at the same input size, epochs, patience and
batch. Writes the same predictions.csv and runs.csv lines, so benchmarks/summary.py ranks both together.

    python3 benchmarks/yolo_cls.py <folds root> <out> --datasets <name> [<name> ...] --models yolo11s-cls \
        --seeds 0 1 2 [--imgsz 384] [--epochs 100] [--patience 20] [--batch 32] [--folds 0 1 2 3 4]
"""
import argparse
import csv
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rpclass.data import check_dataset  # noqa: E402
from rpclass.utils.metrics import evaluate  # noqa: E402

from run import RUN_FIELDS  # noqa: E402


ALIAS = 'c_'


def _plain_names(data, run):
    """Ultralytics reads class names starting with 'n0' as ImageNet synset codes (n01440764 ...) and fails on any other
    (CLEVR's n03 ... n10: KeyError 'n03'). For such a dataset, a copy of its train and val splits (hard links; the test
    is predicted from the dataset itself) with every class folder named ALIAS + name, beside the run; None for any
    other dataset. Folder links would not do: Ultralytics does not follow them when it counts the images. The names
    are mapped back for the predictions."""
    classes = sorted(p.name for p in (data / 'train').iterdir() if p.is_dir())
    if not any(c.startswith('n0') for c in classes):
        return None
    view = run.parent / f'.{run.name}-data'
    for split in ('train', 'val'):
        for c in classes:
            folder = view / split / f'{ALIAS}{c}'
            folder.mkdir(parents=True, exist_ok=True)
            for f in (data / split / c).iterdir():
                if f.is_file() and not (folder / f.name).exists():
                    try:
                        os.link(f, folder / f.name)
                    except OSError:
                        shutil.copy2(f, folder / f.name)
    return view


def train_and_predict(model_name, data, run, imgsz, epochs, patience, batch, seed, workers=8):
    """An Ultralytics classifier trained on data (train/, val/, test/) into the folder run, with its own defaults at
    this input size, epochs, patience and batch; run/predictions.csv for the test (as benchmarks/run.py writes it).
    Returns (test metrics, training seconds)."""
    t0 = time.time()
    view = _plain_names(Path(data), run)
    model = YOLO(f'{model_name}.pt')
    try:
        model.train(data=str(view or data), imgsz=imgsz, epochs=epochs, patience=patience, batch=batch, seed=seed,
                    deterministic=True, workers=workers, project=str(run.parent), name=run.name, exist_ok=True,
                    verbose=False, plots=False)
    finally:
        if view:
            shutil.rmtree(view)
    seconds = time.time() - t0
    best = YOLO(str(run / 'weights' / 'best.pt'))
    ds = check_dataset(data)
    names = [best.names[i] for i in range(len(best.names))]
    if view:
        names = [n[len(ALIAS):] for n in names]
    assert names == [ds['names'][i] for i in range(len(ds['names']))], (names, ds['names'])
    labels, probs = [], []
    with open(run / 'predictions.csv.tmp', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['path', 'label'] + names)
        records = ds['records']['test']
        for i in range(0, len(records), 64):
            chunk = records[i:i + 64]
            for r, res in zip(chunk, best.predict([r['path'] for r in chunk], imgsz=imgsz, verbose=False)):
                pr = res.probs.data.float().cpu().numpy()
                labels.append(r['label'])
                probs.append(pr)
                w.writerow([Path(r['path']).name, ds['names'][r['label']]] + [f'{x:.6f}' for x in pr])
    (run / 'predictions.csv.tmp').rename(run / 'predictions.csv')
    (run / 'weights' / 'last.pt').unlink(missing_ok=True)
    return evaluate(np.log(np.clip(np.array(probs), 1e-12, 1)), np.array(labels), ds['names']), seconds


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('root')
    p.add_argument('out')
    p.add_argument('--datasets', nargs='+', required=True)
    p.add_argument('--models', nargs='+', default=['yolo11s-cls'])
    p.add_argument('--seeds', nargs='+', type=int, default=[0])
    p.add_argument('--folds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    p.add_argument('--imgsz', type=int, default=384)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    root, out = Path(a.root), Path(a.out)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
    for dataset in a.datasets:
        for model_name in a.models:
            for seed in a.seeds:
                for k in a.folds:
                    run = out / dataset / model_name / f's{seed}' / f'fold{k}'
                    if (run / 'predictions.csv').exists():
                        continue
                    if run.exists():
                        shutil.rmtree(run)
                    print(f'\n=== {dataset} · {model_name} · seed {seed} · fold {k}', flush=True)
                    t, seconds = train_and_predict(model_name, root / dataset / f'fold{k}', run, a.imgsz, a.epochs,
                                                   a.patience, a.batch, seed, a.workers)
                    runs = out / 'runs.csv'
                    new = not runs.exists()
                    with open(runs, 'a', newline='') as f:
                        w = csv.DictWriter(f, fieldnames=RUN_FIELDS)
                        if new:
                            w.writeheader()
                        w.writerow({'dataset': dataset, 'model': model_name, 'seed': seed, 'fold': k,
                                    'best_epoch': '', 'planned_epochs': '', 'stop_reason': '',
                                    'test_macro_f1': round(t.macro_f1, 5), 'test_top1': round(t.top1, 5),
                                    'test_balanced_accuracy': round(t.balanced_accuracy, 5),
                                    'test_loss': round(t.loss, 5), 'val_macro_f1': '', 'seconds': round(seconds, 1),
                                    'imgsz': a.imgsz, 'epochs': a.epochs, 'batch': a.batch, 'gpu': gpu})


if __name__ == '__main__':
    main()
