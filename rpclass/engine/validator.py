# RP-CLASS - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""Validation: run the model on a dataset split and compute the metrics (optionally the plots and metrics.json)."""
import contextlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data import collate
from ..utils.metrics import evaluate, softmax
from ..utils.plots import GRID_IMAGES, pick_examples, save_plots


@torch.no_grad()
def validate(net, dataset, names, device, batch=32, plots_dir=None, workers=0, amp_dtype=None):
    loader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=workers, collate_fn=collate)
    was_training = net.training
    net.eval()
    logits, labels = [], []
    for images, targets, _ in loader:
        amp = torch.autocast(str(device).split(':')[0], dtype=amp_dtype) if amp_dtype else contextlib.nullcontext()
        with amp:
            out = net(images.to(device, non_blocking=True))
        logits.append(out.float().cpu().numpy())          # one copy per batch
        labels.append(targets.numpy())
    net.train(was_training)
    logits = np.concatenate(logits) if logits else np.zeros((0, len(names)), np.float32)
    labels = np.concatenate(labels) if labels else np.zeros(0, np.int64)
    metrics = evaluate(logits, labels, names)
    if plots_dir is not None:
        probs = softmax(logits.astype(np.float64))
        picks = pick_examples(probs, labels, GRID_IMAGES)
        items = [(dataset.read(dataset.records[i]).image, int(labels[i]), int(probs[i].argmax()), float(probs[i].max()))
                 for i in picks]
        save_plots(metrics, names, items, plots_dir)
        (Path(plots_dir) / 'metrics.json').write_text(json.dumps(
            {**metrics.to_dict(), 'images': metrics.images, 'per_class': metrics.per_class,
             'confusion_matrix': metrics.confusion_matrix.tolist()}, indent=2))
    return metrics
