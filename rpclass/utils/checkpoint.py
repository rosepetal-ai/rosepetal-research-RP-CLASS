# RP-CLASS - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""Checkpoints and partial weight loading.

A checkpoint carries everything needed to use the model: weights, architecture, class names and input size.

Partial loading matches weights by name and shape, so a trained model initializes the same size for a dataset with
other classes: only the classification head keeps its fresh initialization.
"""
import datetime

import torch

from ..cfg.schema import derive, from_dict, to_dict

FORMAT = 1


def save_checkpoint(path, net, cfg, names, imgsz, metrics=None, extra=None):
    ckpt = {'rpclass_format': FORMAT, 'config': to_dict(cfg), 'names': {int(k): str(v) for k, v in names.items()},
            'imgsz': [int(imgsz[0]), int(imgsz[1])], 'metrics': metrics or {},
            'date': datetime.datetime.now().isoformat(timespec='seconds'),
            'state_dict': {k: v.detach().cpu() for k, v in net.state_dict().items()}}
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def load_checkpoint(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    if ckpt.get('rpclass_format') != FORMAT:
        raise ValueError(f'{path} is not an RP-CLASS checkpoint (or comes from an incompatible version)')
    ckpt['config'] = from_dict(ckpt['config'])
    ckpt['names'] = {int(k): v for k, v in ckpt['names'].items()}
    return ckpt


def build_from_checkpoint(ckpt):
    from ..nn import Classifier
    net = Classifier(derive(ckpt['config'], len(ckpt['names']), ckpt['imgsz']), pretrained=False)
    net.load_state_dict(ckpt['state_dict'])
    return net


def load_weights(net, state_dict):
    """Load whatever fits into `net`; returns a report of what was loaded and what kept its fresh initialization."""
    own = net.state_dict()
    new, reinit = {}, []
    for k, v in own.items():
        s = state_dict.get(k)
        if s is not None and s.shape == v.shape:
            new[k] = s
        else:
            reinit.append(k)
    net.load_state_dict({**own, **new})
    total = sum(v.numel() for v in own.values())
    loaded = sum(own[k].numel() for k in new)
    groups = sorted({k.rsplit('.', 1)[0] for k in reinit})
    return {'loaded_fraction': loaded / total, 'reinitialized': groups,
            'summary': f"loaded {100 * loaded / total:.1f}% of the weights"
                       + (f" · kept fresh: {', '.join(groups)}" if groups else '')}
