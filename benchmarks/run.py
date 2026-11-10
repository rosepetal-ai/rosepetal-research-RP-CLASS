# RP-CLASS - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""Benchmark on k-fold datasets (benchmarks/folds.py): every (dataset, backbone, seed) trains once per fold, and the
test predictions of all the folds give out-of-fold metrics over every image (benchmarks/summary.py).

    python3 benchmarks/run.py <folds root> <out> --datasets <name> [<name> ...] --models hgnetv2-b0 vit-s16-dinov3 \
        --seeds 0 1 2 [--imgsz 384] [--epochs 100] [--patience 20] [--batch 32] [--folds 0 1 2 3 4] [--augment d4]
        [--recipe backbone_lr=2e-4 --tag lr2x] [--shrink-factor 3 --tag shrink3]

--models are backbones of rpclass.cfg.schema.BACKBONES; --augment one of benchmarks/augments.py (default: the
library's, resolved to its name), the run folder then being <model>+<augment> (<model> alone for hflip, the
provisional default); --recipe overrides entries of the recipe under a --tag
(benchmarks/recipe.py), the folder then ending in @<tag>. Writes <out>/<dataset>/<model>/s<seed>/fold<k>/ (the run:
weights/best.pt, results.csv, plots; last.pt is deleted, nothing reads it) with predictions.csv (every test image: its
path, true class and probabilities) and appends a line to <out>/runs.csv. A run with its predictions.csv is not
trained again, so the same command resumes an interrupted benchmark; a run without them (it stopped half way) starts
over.
"""
import argparse
import csv
import shutil
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rpclass import RPCLASS  # noqa: E402
from rpclass.cfg.schema import BACKBONES, SIZES, Architecture  # noqa: E402
from rpclass.data import check_dataset  # noqa: E402
from rpclass.data.sources import read_image  # noqa: E402
from rpclass.engine import trainer as T  # noqa: E402

from augments import AUGMENTS, resolve  # noqa: E402
import recipe  # noqa: E402

RUN_FIELDS = ['dataset', 'model', 'seed', 'fold', 'best_epoch', 'planned_epochs', 'stop_reason', 'test_macro_f1',
              'test_top1', 'test_balanced_accuracy', 'test_loss', 'val_macro_f1', 'seconds', 'imgsz', 'epochs', 'batch',
              'gpu']


class _Images(Dataset):
    """Test images read in DataLoader workers: predict() on the arrays is predict() on the files (it reads them with
    the same read_image), without decoding thousands of them one by one in this process (CLEVR's test: 15,000)."""

    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return read_image(self.paths[i])


def predictions(model, fold_dir, out, workers=8):
    """Probabilities of every test image, in the order of the dataset check."""
    ds = check_dataset(fold_dir)
    records = ds['records']['test']
    names = [model.names[i] for i in range(len(model.names))]
    loader = DataLoader(_Images([r['path'] for r in records]), batch_size=64, num_workers=workers,
                        collate_fn=list)
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['path', 'label'] + names)
        start = 0
        for images in loader:
            chunk = records[start:start + len(images)]
            start += len(images)
            for r, res in zip(chunk, model.predict(images, color='RGB')):
                w.writerow([Path(r['path']).name, ds['names'][r['label']]] + [f'{p:.6f}' for p in res.probs])


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('root')
    p.add_argument('out')
    p.add_argument('--datasets', nargs='+', required=True)
    p.add_argument('--models', nargs='+', required=True)
    p.add_argument('--seeds', nargs='+', type=int, default=[0])
    p.add_argument('--folds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    p.add_argument('--imgsz', type=int, default=384)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--workers', type=int, default=None)
    p.add_argument('--augment', default='default', choices=sorted(AUGMENTS))
    p.add_argument('--recipe', nargs='*', default=[], help='key=value overrides of the recipe (needs --tag)')
    p.add_argument('--tag', default='', help='names the runs of --recipe in their folder')
    p.add_argument('--shrink-factor', type=int, default=None,
                   help='RP-CLASS: read large training images at this odd multiple of their letterbox, as the '
                        "library's own augmentations do (rpclass.data.dataset.SHRINK_FACTOR); needs --tag")
    a = p.parse_args()
    recipe.apply(a.recipe, a.tag)
    augment = AUGMENTS[resolve(a.augment)]
    if a.shrink_factor is not None:
        augment = recipe.shrunk(augment, a.shrink_factor, a.tag)
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    unknown = [m for m in a.models if m not in BACKBONES]
    if unknown:
        raise SystemExit(f"unknown models {unknown}; the backbones are {', '.join(BACKBONES)}")
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
    runs = out / 'runs.csv'
    for dataset in a.datasets:
        for model_name in a.models:
            SIZES[model_name] = Architecture(model_name)
            for seed in a.seeds:
                for k in a.folds:
                    folder = recipe.folder(model_name, a.augment, a.tag)
                    run = out / dataset / folder / f's{seed}' / f'fold{k}'
                    if (run / 'predictions.csv').exists():
                        continue
                    if run.exists():
                        shutil.rmtree(run)
                    print(f'\n=== {dataset} · {folder} · seed {seed} · fold {k}', flush=True)
                    T.RECIPE['seed'] = seed
                    model = RPCLASS(model_name)
                    t0 = time.time()
                    res = model.train(root / dataset / f'fold{k}', imgsz=a.imgsz, epochs=a.epochs, patience=a.patience,
                                      batch=a.batch, name=f'fold{k}', project=str(run.parent), workers=a.workers,
                                      augment=augment)
                    seconds = time.time() - t0
                    predictions(model, root / dataset / f'fold{k}', run / 'predictions.csv.tmp')
                    (run / 'predictions.csv.tmp').rename(run / 'predictions.csv')
                    (run / 'weights' / 'last.pt').unlink(missing_ok=True)     # best.pt is kept; last.pt only costs disk
                    t, v = res['test'], res['val']
                    new = not runs.exists()
                    with open(runs, 'a', newline='') as f:
                        w = csv.DictWriter(f, fieldnames=RUN_FIELDS)
                        if new:
                            w.writeheader()
                        w.writerow({'dataset': dataset, 'model': folder, 'seed': seed, 'fold': k,
                                    'best_epoch': res['best_epoch'], 'planned_epochs': res['planned_epochs'],
                                    'stop_reason': res['stop_reason'], 'test_macro_f1': round(t.macro_f1, 5),
                                    'test_top1': round(t.top1, 5),
                                    'test_balanced_accuracy': round(t.balanced_accuracy, 5),
                                    'test_loss': round(t.loss, 5), 'val_macro_f1': round(v.macro_f1, 5),
                                    'seconds': round(seconds, 1), 'imgsz': a.imgsz, 'epochs': a.epochs,
                                    'batch': a.batch, 'gpu': gpu})


if __name__ == '__main__':
    main()
