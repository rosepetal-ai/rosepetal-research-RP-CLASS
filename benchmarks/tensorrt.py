# RP-CLASS - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""TensorRT engines of trained models, as trtexec builds them: latency, and whether float16 still gives the right
answer.

    python3 benchmarks/tensorrt.py prepare <benchmark out> <folds root> <work> --dataset <name> [--seed 0] [--fold 0]
        exports every trained model of that fold to <work>/<model>/model.onnx, with one real test image letterboxed as
        the engine expects (input.bin, float32 [1, 3, H, W]) and what PyTorch answers for it (expected.json)
    (then, where trtexec is, for each model and precision; GPU time, no host-device copies, median of 1000 runs)
        trtexec --onnx=model.onnx --shapes=images:1x3xHxW --minShapes=... --optShapes=... --maxShapes=images:16x3xHxW
                [--fp16] --loadInputs=images:input.bin --exportOutput=out_<precision>.json --noDataTransfers
                --iterations=1000 --warmUp=500 --duration=0 > trtexec_<precision>.log
    python3 benchmarks/tensorrt.py sizes <work> --image <path> [--checkpoint m=<best.pt> ...]
                                         [--yolo yolo11s-cls.pt --yolo-imgsz 224 384]
        the same for every size of the library at its default input size (s, m, l at 384, lt at 448): a size given a
        checkpoint exports its trained network, any other its pretrained one with a fresh head, which costs the same
        time. With --yolo, an Ultralytics classifier too (needs ultralytics), for its time only
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


def sizes(a):
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from rpclass import RPCLASS
    from rpclass.cfg.schema import SIZES, default_imgsz, derive
    from rpclass.data import letterbox, load_images
    from rpclass.engine.exporter import export_onnx
    from rpclass.nn import Classifier
    checkpoints = dict(c.split('=', 1) for c in a.checkpoint)
    image = load_images(a.image)[0][0]
    for size in SIZES:
        if size in checkpoints:
            m = RPCLASS(checkpoints[size])
            net, names, imgsz = m.net, m.names, m.imgsz
        else:
            torch.manual_seed(0)
            imgsz = default_imgsz(SIZES[size])
            net, names = Classifier(derive(SIZES[size], 2, imgsz), pretrained=True), {0: 'a', 1: 'b'}
        work = Path(a.work) / f'rpclass-{size}'
        export_onnx(net, size, names, imgsz, work, check=False)
        x = letterbox(image, *imgsz)[0][None]
        x.numpy().astype(np.float32).tofile(work / 'input.bin')
        with torch.no_grad():
            z = net.cpu().eval()(x).float()
        (work / 'expected.json').write_text(json.dumps({'imgsz': list(imgsz), 'logits': (z - z.amax(1, keepdim=True))
                                                        .numpy()[0].tolist()}))
        print(f'rpclass-{size} ({imgsz[0]} px): {work}', flush=True)
    for side in a.yolo_imgsz if a.yolo else ():
        from ultralytics import YOLO
        work = Path(a.work) / f'{Path(a.yolo).stem}-{side}'
        work.mkdir(parents=True, exist_ok=True)
        onnx = YOLO(a.yolo).export(format='onnx', imgsz=side, dynamic=True, simplify=False)
        Path(onnx).replace(work / 'model.onnx')
        np.random.default_rng(0).random((1, 3, side, side), np.float32).tofile(work / 'input.bin')
        (work / 'time_only.json').write_text(json.dumps({'imgsz': [side, side]}))
        print(f'{work.name}: {work}', flush=True)


def _engine_output(path):
    """trtexec --exportOutput: a list of {name, dimensions, values}."""
    data = json.loads(Path(path).read_text())
    out = next(o for o in data if o['name'] == 'logits')
    return np.array(out['values'], np.float64)


def report(a):
    rows = []
    gpus = set()
    for work in sorted(p for p in Path(a.work).iterdir() if (p / 'expected.json').exists()
                       or (p / 'time_only.json').exists()):
        expected = work / 'expected.json'
        ref = np.array(json.loads(expected.read_text())['logits']) if expected.exists() else None
        cells = []
        for precision in ('fp16', 'fp32'):
            log = work / f'trtexec_{precision}.log'
            if not log.exists():
                cells += ['-', '-']
                continue
            text = log.read_text()
            m = re.search(r'GPU Compute Time: min = [\d.]+ ms, max = [\d.]+ ms, mean = [\d.]+ ms, median = ([\d.]+) ms',
                          text)
            gpus.update(re.findall(r'Selected Device: (.+)', text))
            ms = f'{float(m.group(1)):.2f} ms' if m else 'failed'
            out = work / f'out_{precision}.json'
            if ref is None:
                cells += [ms, 'time only']
                continue
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
    print(f"GPU: {', '.join(sorted(gpus)) or 'unknown'}\n")
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
    ps = sub.add_parser('sizes')
    ps.add_argument('work')
    ps.add_argument('--image', required=True, help='a real image, letterboxed as each engine expects')
    ps.add_argument('--checkpoint', action='append', default=[], help='size=best.pt: that size trained')
    ps.add_argument('--yolo', help='an Ultralytics classifier, e.g. yolo11s-cls.pt (time only)')
    ps.add_argument('--yolo-imgsz', type=int, nargs='+', default=[224])
    pr = sub.add_parser('report')
    pr.add_argument('work')
    a = p.parse_args()
    {'prepare': prepare, 'sizes': sizes, 'report': report}[a.cmd](a)


if __name__ == '__main__':
    main()
