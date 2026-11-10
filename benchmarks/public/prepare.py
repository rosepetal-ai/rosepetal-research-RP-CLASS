# RP-CLASS - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""Download and prepare public benchmark datasets (docs/BENCHMARK.md).

    python3 -I benchmarks/public/prepare.py <name> [<name> ...] --raw <downloads> --out <prepared>
            [--manifests benchmarks/public/manifests] [--tasks <task> ...] [--keep-raw]

Each <name> is a module of this folder (the benchmark) or of benchmarks/dev (the development suite) with:
    NAME      the dataset's folder name
    TASKS     its task names: one for most datasets, one per product or texture for VisA and DAGM
    IMGSZ     the benchmark's input size for it (224, or 448 for VisA, DAGM and the tyres)
    LICENCE   the licence, as read at SOURCE; SOURCE: the page that states it
    download(raw) -> folder   fetches (benchmarks/public/fetch.py) into raw = <downloads>/<NAME>, idempotent; a
              download(raw, tasks) gets only what those tasks need
    rows(folder, task) -> rows, or (rows, share)   (benchmarks/public/common.py)
and optionally POSITIVE (the defect class of a binary inspection task, for AUROC), KEEP (files in raw that survive
its deletion: a download by hand) and NOTES / notes(folder, task) (what the module dropped or changed, and why).
The splits are written to <prepared>/<NAME>/<task>/, and their manifest and info, which are what gets published, to
<manifests>/<NAME>/<task>.csv and .json. A dataset whose tasks are all prepared is not downloaded again. The raw
download is deleted once every task is prepared (the prepared images are hard links and survive), unless --keep-raw.
Run with python -I (so numpy, Pillow, requests and, for beans, pyarrow, for pcam, h5py must be installed outside
~/.local): nothing downloaded can shadow a module. The development suite's manifests never go among the benchmark's
(benchmarks/public/manifests), and a module name may exist in only one of the two folders.
"""
import argparse
import importlib
import inspect
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import PIL
from PIL import features

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[1] / 'dev'))     # the development suite's modules
from common import finalize  # noqa: E402


def environment():
    """What wrote the files our scripts make (PNG conversions, crops, re-encoded JPEGs): their bytes, hence their
    SHA-256 in the manifests, depend on it."""
    return {'python': sys.version.split()[0], 'numpy': np.__version__, 'pillow': PIL.__version__,
            'zlib': features.version('zlib'), 'libjpeg': features.version('jpg')}


def prepare(name, raw_root, out_root, manifests=None, keep_raw=False, tasks=None):
    mod = importlib.import_module(name)
    if manifests and Path(mod.__file__).parent.name == 'dev' and 'public' in Path(manifests).resolve().parts:
        raise SystemExit(f'{name}: a development-suite dataset; its manifests go to benchmarks/dev/manifests, never '
                         f'among the benchmark\'s ({manifests})')
    raw, out = Path(raw_root) / mod.NAME, Path(out_root) / mod.NAME
    todo = [t for t in mod.TASKS if tasks is None or t in tasks]
    print(f'=== {mod.NAME}', flush=True)
    if any(not (out / t / 'info.json').exists() for t in todo):
        takes_tasks = 'tasks' in inspect.signature(mod.download).parameters      # DAGM: one zip per task
        folder = mod.download(raw, tasks=todo) if takes_tasks else mod.download(raw)
        ledger = raw / 'downloads.json'
        meta = {'dataset': mod.NAME, 'licence': mod.LICENCE, 'source': mod.SOURCE,
                'positive': getattr(mod, 'POSITIVE', None), 'downloads': json.loads(ledger.read_text()) if
                ledger.exists() else [], 'environment': environment()}
        for task in todo:
            if (out / task / 'info.json').exists():
                continue
            notes = dict(getattr(mod, 'NOTES', {}))
            notes.update(mod.notes(folder, task) if hasattr(mod, 'notes') else {})
            got = mod.rows(folder, task)
            rows, share = got if isinstance(got, tuple) else (got, None)
            finalize(task, rows, out, mod.IMGSZ, share=share, root=raw, meta={**meta, 'notes': notes})
        if not keep_raw and all((out / t / 'info.json').exists() for t in mod.TASKS):
            _clean(raw, getattr(mod, 'KEEP', ()))      # kept while some task (--tasks) is still to prepare
    else:
        print(f'{mod.NAME}: every task already prepared in {out}', flush=True)
    if manifests:
        dst = Path(manifests) / mod.NAME
        dst.mkdir(parents=True, exist_ok=True)
        for task in todo:
            shutil.copy2(out / task / 'manifest.csv', dst / f'{task}.csv')
            shutil.copy2(out / task / 'info.json', dst / f'{task}.json')


def _clean(raw, keep):
    """Delete the raw download, except the files named in keep (and the ledger that records them)."""
    if not raw.exists():
        return
    kept = {raw / k for k in keep if (raw / k).exists()}
    if kept:
        kept.add(raw / 'downloads.json')
    for p in raw.iterdir():
        if p in kept:
            continue
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    if not kept:
        raw.rmdir()
    print(f'{raw.name}: raw download deleted' + (f', kept {", ".join(sorted(k.name for k in kept))}' if kept else ''),
          flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('names', nargs='+')
    p.add_argument('--raw', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--manifests')
    p.add_argument('--tasks', nargs='+', help='only these tasks (e.g. DAGM development classes)')
    p.add_argument('--keep-raw', action='store_true')
    a = p.parse_args()
    here = Path(__file__).resolve().parent
    both = [n for n in a.names if (here / f'{n}.py').exists() and (here.parent / 'dev' / f'{n}.py').exists()]
    if both:
        raise SystemExit(f"{', '.join(both)}: a module of that name is in both benchmarks/public and benchmarks/dev; "
                         f"rename one")
    for name in a.names:
        prepare(name, a.raw, a.out, a.manifests, a.keep_raw, a.tasks)


if __name__ == '__main__':
    main()
