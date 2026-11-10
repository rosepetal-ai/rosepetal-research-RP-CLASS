# RP-CLASS - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""Can a trained model be served in float16? Float16 engines are the usual choice for serving, and a value above
65504 anywhere inside the model becomes inf there (the DINOv3 ViT's attention did this). For every trained model of
a benchmark (one fold), this exports it and runs ONNX Runtime once in float32 on real test images with every
intermediate tensor exposed, and reports the tensors that float16 cannot hold.

    python3 benchmarks/fp16.py <benchmark out> <folds root> --dataset <name> [--seed 0] [--fold 0] [--images 8]
"""
import argparse
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rpclass import RPCLASS  # noqa: E402
from rpclass.cfg.schema import BACKBONES, SIZES, Architecture, size_of  # noqa: E402
from rpclass.data import check_dataset, letterbox, load_images  # noqa: E402
from rpclass.engine.exporter import export_onnx, fp16_overflows  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('out')
    p.add_argument('folds')
    p.add_argument('--dataset', required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--images', type=int, default=8)
    a = p.parse_args()
    for b in BACKBONES:
        SIZES.setdefault(b, Architecture(b))
    records = check_dataset(Path(a.folds) / a.dataset / f'fold{a.fold}')['records']['test'][:a.images]
    print(f'| model | values over 65504 | largest | where |\n|---|---|---|---|')
    for ckpt in sorted(Path(a.out).glob(f'{a.dataset}/*/s{a.seed}/fold{a.fold}/weights/best.pt')):
        model_name = ckpt.parts[-5]
        if model_name not in BACKBONES:
            continue
        m = RPCLASS(ckpt)
        x = torch.stack([letterbox(img, *m.imgsz)[0] for img, _ in load_images([r['path'] for r in records])])
        with tempfile.TemporaryDirectory() as tmp:
            export_onnx(m.net, size_of(m.cfg) or m.cfg.backbone, m.names, m.imgsz, tmp, check=False)
            big = []
            for i in range(len(x)):
                big += fp16_overflows(Path(tmp) / 'model.onnx', x[i:i + 1])
        names = sorted({n for n, _ in big})
        top = max((v for _, v in big), default=0.0)
        where = ', '.join(names[:3]) + (' ...' if len(names) > 3 else '')
        print(f'| {model_name} | {len(names)} tensors | {top:.3g} | {where} |', flush=True)


if __name__ == '__main__':
    main()
