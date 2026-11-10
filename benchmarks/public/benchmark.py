# RP-CLASS - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""The public benchmark (docs/BENCHMARK.md): every task, method and seed trains once on that seed's training draw,
chooses its epoch on val, and is scored once on the test.

    python3 benchmarks/public/benchmark.py <prepared> <out> --datasets eurosat visa ... \
        --models hgnetv2-b2 vit-s16-dinov3 yolo11s-cls --seeds 0 1 2 [--epochs 100] [--patience 20] [--batch 32]
        [--tasks candle pcb1 ...] [--augment default] [--recipe key=value ... --tag name]
        [--shrink-factor 3 --tag name] [--final]

<prepared> is what benchmarks/public/prepare.py wrote. --models are RP-CLASS backbones (rpclass.cfg.schema.BACKBONES)
or Ultralytics classifiers (yolo*-cls), each with its own documented default recipe; every method gets the task's
input size (its info.json: 224, or 448 for VisA, DAGM and the tyres), the same epoch cap, patience and batch. Writes
<out>/<dataset>/<task>/<model>/s<seed>/ (the run and predictions.csv: every test image, its class and probabilities;
the weights are deleted once the predictions are written, unless --keep-weights: the benchmark publishes metrics,
never weights) and one line per run in <out>/runs.csv; a run with its predictions.csv is not trained again, so the
same command resumes. DAGM's classes 7-10 are refused without --final: they are not looked at during development,
as the dataset's authors ask. benchmarks/public/report.py makes the tables.
"""
import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))
from rpclass import RPCLASS  # noqa: E402
from rpclass.cfg.schema import BACKBONES, SIZES, Architecture  # noqa: E402
from rpclass.engine import trainer as T  # noqa: E402

from augments import AUGMENTS, resolve  # noqa: E402
import recipe  # noqa: E402
from run import RUN_FIELDS, predictions  # noqa: E402

HELD_OUT = {'dagm': ('class07', 'class08', 'class09', 'class10')}


def tasks_of(prepared, dataset, only=None):
    """The prepared tasks of a dataset, with their info.json."""
    out = []
    for info in sorted((Path(prepared) / dataset).glob('*/info.json')):
        if only is None or info.parent.name in only:
            out.append((info.parent.name, json.loads(info.read_text())))
    if not out:
        raise SystemExit(f'{dataset}: no prepared task in {prepared}; run benchmarks/public/prepare.py first')
    return out


def train_rpclass(model_name, data, run, imgsz, a, seed):
    SIZES[model_name] = Architecture(model_name)
    T.RECIPE['seed'] = seed
    model = RPCLASS(model_name)
    res = model.train(data, imgsz=imgsz, epochs=a.epochs, patience=a.patience, batch=a.batch, name=run.name,
                      project=str(run.parent), workers=a.workers, augment=a.augment_fn)
    predictions(model, data, run / 'predictions.csv.tmp')
    (run / 'predictions.csv.tmp').rename(run / 'predictions.csv')
    (run / 'weights' / 'last.pt').unlink(missing_ok=True)
    return res['test'], res['val'].macro_f1, res['best_epoch'], res['planned_epochs'], res['stop_reason']


def train_yolo(model_name, data, run, imgsz, a, seed):
    import yolo_cls
    t, _ = yolo_cls.train_and_predict(model_name, data, run, imgsz, a.epochs, a.patience, a.batch, seed,
                                      8 if a.workers is None else a.workers)
    return t, '', '', '', ''


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('prepared')
    p.add_argument('out')
    p.add_argument('--datasets', nargs='+', required=True)
    p.add_argument('--models', nargs='+', required=True)
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--tasks', nargs='+', help='only these tasks')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--workers', type=int, default=None)
    p.add_argument('--augment', default='default', choices=sorted(AUGMENTS), help='RP-CLASS only')
    p.add_argument('--recipe', nargs='*', default=[], help='RP-CLASS only: key=value overrides (needs --tag)')
    p.add_argument('--tag', default='', help='names the runs of --recipe in their folder')
    p.add_argument('--shrink-factor', type=int, default=None,
                   help='RP-CLASS: read large training images at this odd multiple of their letterbox, as the '
                        "library's own augmentations do (rpclass.data.dataset.SHRINK_FACTOR); needs --tag")
    p.add_argument('--final', action='store_true', help="the final run: DAGM's classes 7-10 included")
    p.add_argument('--keep-weights', action='store_true', help='keep each run\'s best.pt (deleted by default)')
    a = p.parse_args()
    recipe.apply(a.recipe, a.tag)
    a.augment_fn = AUGMENTS[resolve(a.augment)]
    if a.shrink_factor is not None:
        a.augment_fn = recipe.shrunk(a.augment_fn, a.shrink_factor, a.tag)
    unknown = [m for m in a.models if m not in BACKBONES and not m.endswith('-cls')]
    if unknown:
        raise SystemExit(f"unknown models {unknown}; RP-CLASS backbones are {', '.join(BACKBONES)}, Ultralytics "
                         f"classifiers end in -cls")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
    for dataset in a.datasets:
        for task, info in tasks_of(a.prepared, dataset, a.tasks):
            if task in HELD_OUT.get(dataset, ()) and not a.final:
                print(f'{dataset}/{task}: held out until the final run (--final)', flush=True)
                continue
            for model_name in a.models:
                folder = model_name if model_name.endswith('-cls') else recipe.folder(model_name, a.augment, a.tag)
                for seed in a.seeds:
                    run = out / dataset / task / folder / f's{seed}'
                    if (run / 'predictions.csv').exists():
                        continue
                    if run.exists():
                        shutil.rmtree(run)
                    print(f'\n=== {dataset}/{task} · {folder} · seed {seed}', flush=True)
                    data = Path(a.prepared) / dataset / task / f's{seed}'
                    train = train_yolo if model_name.endswith('-cls') else train_rpclass
                    t0 = time.time()
                    t, val_f1, best, planned, stop = train(model_name, data, run, info['imgsz'], a, seed)
                    runs = out / 'runs.csv'
                    new = not runs.exists()
                    with open(runs, 'a', newline='') as f:
                        w = csv.DictWriter(f, fieldnames=RUN_FIELDS)
                        if new:
                            w.writeheader()
                        w.writerow({'dataset': f'{dataset}/{task}', 'model': folder, 'seed': seed, 'fold': 0,
                                    'best_epoch': best, 'planned_epochs': planned, 'stop_reason': stop,
                                    'test_macro_f1': round(t.macro_f1, 5), 'test_top1': round(t.top1, 5),
                                    'test_balanced_accuracy': round(t.balanced_accuracy, 5),
                                    'test_loss': round(t.loss, 5),
                                    'val_macro_f1': round(val_f1, 5) if val_f1 != '' else '',
                                    'seconds': round(time.time() - t0, 1), 'imgsz': info['imgsz'],
                                    'epochs': a.epochs, 'batch': a.batch, 'gpu': gpu})
                    if not a.keep_weights:
                        (run / 'weights' / 'best.pt').unlink(missing_ok=True)


if __name__ == '__main__':
    main()
