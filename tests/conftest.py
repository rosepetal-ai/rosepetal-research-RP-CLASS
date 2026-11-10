# RP-CLASS - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from rpclass.cfg.schema import SIZES, derive
from rpclass.nn import Classifier
from rpclass.utils.checkpoint import save_checkpoint

NAMES = {0: 'nok', 1: 'ok', 2: 'scratch'}
COLORS = {'nok': (200, 40, 40), 'ok': (40, 200, 40), 'scratch': (40, 40, 200)}


@pytest.fixture(autouse=True)
def hermetic(monkeypatch, tmp_path_factory):
    """Tests never download pretrained weights (every network starts from random ones) and never write the dataset
    check cache to the real home."""
    import rpclass.data.check as check
    import rpclass.nn.classifier as classifier
    real = classifier.timm.create_model
    monkeypatch.setattr(classifier, '_create_model',
                        lambda name, pretrained=False, **kw: real(name, pretrained=False, **kw))
    monkeypatch.setattr(check, 'CACHE_DIR', tmp_path_factory.getbasetemp() / 'rpclass-cache')


def make_net(num_classes=3, size='s', imgsz=(64, 64), arch=None):
    """A network of a size (or of any architecture), random weights."""
    torch.manual_seed(0)
    return Classifier(derive(arch or SIZES[size], num_classes, imgsz), pretrained=False).eval()


def make_dataset(root, splits=None, classes=tuple(NAMES.values()), seed=0, size=(60, 90)):
    """A dataset folder: per split, per class, `n` images of the class color plus noise (so a model can learn them)."""
    splits = splits or {'train': 4, 'val': 2, 'test': 2}
    rng = np.random.default_rng(seed)
    root = Path(root)
    for split, n in splits.items():
        for c in classes:
            folder = root / split / c
            folder.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                img = np.clip(np.array(COLORS.get(c, (128, 128, 128))) + rng.integers(-30, 30, (*size, 3)), 0, 255)
                Image.fromarray(img.astype(np.uint8)).save(folder / f'{c}_{i}.png')
    return root


@pytest.fixture
def trained_ckpt(tmp_path):
    """A checkpoint with random weights: enough to exercise predict and export."""
    path = tmp_path / 'best.pt'
    save_checkpoint(path, make_net(imgsz=(64, 96)), SIZES['s'], NAMES, [64, 96])
    return path
