# RP-CLASS - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""TensorRT engines of trained models, as trtexec builds them: latency, and whether float16 still gives the right
answer.

    python3 benchmarks/tensorrt.py prepare <benchmark out> <folds root> <work> --dataset <name> [--seed 0] [--fold 0]
        exports every trained model of that fold to <work>/<model>/model.onnx, with one real test image letterboxed as
        the engine expects (input.bin, float32 [1, 3, H, W]) and what PyTorch answers for it (expected.json)
    (then, where trtexec is, for each model and precision; GPU time, no host-device copies, median of 500 runs)
        trtexec --onnx=model.onnx --shapes=images:1x3xHxW --minShapes=... --optShapes=... --maxShapes=images:16x3xHxW
                [--fp16] --loadInputs=images:input.bin --exportOutput=out_<precision>.json --noDataTransfers
                --iterations=500 --warmUp=500 --duration=0 > trtexec_<precision>.log
    python3 benchmarks/tensorrt.py report <work>
        the median GPU time per precision, and how far each engine's answer is from PyTorch's (NaN included)
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np


def prepare(a):
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from rpclass import RPCLASS
    from rpclass.cfg.schema import BACKBONES, SIZES, Architecture, size_of
    from rpclass.data import check_dataset, letterbox, load_images
    from rpclass.engine.exporter import export_onnx
    for b in BACKBONES:
        SIZES.setdefault(b, Architecture(b))
    record = check_dataset(Path(a.folds) / a.dataset / f'fold{a.fold}')['records']['test'][0]
    for ckpt in sorted(Path(a.out).glob(f'{a.dataset}/*/s{a.seed}/fold{a.fold}/weights/best.pt')):
        name = ckpt.parts[-5]
        if name not in BACKBONES:
            continue
        m = RPCLASS(ckpt)
        work = Path(a.work) / name
        export_onnx(m.net, size_of(m.cfg) or m.cfg.backbone, m.names, m.imgsz, work, check=False)
        x = letterbox(load_images(record['path'])[0][0], *m.imgsz)[0][None]
        x.numpy().astype(np.float32).tofile(work / 'input.bin')
        with torch.no_grad():
            z = m.net.cpu().eval()(x).float()
        (work / 'expected.json').write_text(json.dumps({'imgsz': m.imgsz, 'logits': (z - z.amax(1, keepdim=True))
                                                        .numpy()[0].tolist()}))
        print(f'{name}: {work}', flush=True)


def _engine_output(path):
    """trtexec --exportOutput: a list of {name, dimensions, values}."""
    data = json.loads(Path(path).read_text())
    out = next(o for o in data if o['name'] == 'logits')
    return np.array(out['values'], np.float64)


def report(a):
    rows = []
    for work in sorted(p for p in Path(a.work).iterdir() if (p / 'expected.json').exists()):
        ref = np.array(json.loads((work / 'expected.json').read_text())['logits'])
        cells = []
        for precision in ('fp16', 'fp32'):
            log = work / f'trtexec_{precision}.log'
            if not log.exists():
                cells += ['-', '-']
                continue
            m = re.search(r'GPU Compute Time: min = [\d.]+ ms, max = [\d.]+ ms, mean = [\d.]+ ms, median = ([\d.]+) ms',
                          log.read_text())
            ms = f'{float(m.group(1)):.2f} ms' if m else 'failed'
            out = work / f'out_{precision}.json'
            if not out.exists():
                cells += [ms, 'no output']
                continue
            z = _engine_output(out)
            if not np.isfinite(z).all():
                cells += [ms, 'NaN / inf']
                continue
            z = z - z.max()
            p, q = np.exp(z) / np.exp(z).sum(), np.exp(ref) / np.exp(ref).sum()
            same = int(z.argmax()) == int(ref.argmax())
            verdict = 'same class' if same else 'OTHER class'
            cells += [ms, f'probabilities {np.abs(p - q).max():.1e} apart, {verdict}']
        rows.append((work.name, cells))
    print('| model | float16 | float16 answer | float32 | float32 answer |\n|---|---|---|---|---|')
    for name, c in rows:
        print(f'| {name} | ' + ' | '.join(c) + ' |')


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    pp = sub.add_parser('prepare')
    pp.add_argument('out')
    pp.add_argument('folds')
    pp.add_argument('work')
    pp.add_argument('--dataset', required=True)
    pp.add_argument('--seed', type=int, default=0)
    pp.add_argument('--fold', type=int, default=0)
    pr = sub.add_parser('report')
    pr.add_argument('work')
    a = p.parse_args()
    prepare(a) if a.cmd == 'prepare' else report(a)


if __name__ == '__main__':
    main()
