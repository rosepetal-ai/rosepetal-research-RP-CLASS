# RP-CLASS - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""How a new head starts: the linear-probe start (RECIPE head_start), lt's default."""
import math

import torch

from rpclass import RPCLASS
from rpclass.engine.trainer import RECIPE, _fit_probe

from .conftest import make_dataset


def test_only_lt_starts_from_a_probe():
    from rpclass.cfg.schema import SIZES
    starts = {size: RECIPE['head_start'].get(SIZES[size].backbone) for size in ('s', 'm', 'l', 'lt')}
    assert starts == {'s': None, 'm': None, 'l': None, 'lt': 'probe'}


def test_the_probe_separates_separable_features_with_bounded_logits():
    g = torch.Generator().manual_seed(0)
    n, d, c, eps = 300, 64, 2, 0.1
    y = (torch.rand(n, generator=g) < 0.09).long()                    # 9 % minority, as in the failing sets
    x = torch.randn(n, d, generator=g, dtype=torch.float64) * 0.05 + 3.0
    x[:, 0] += y.double() * 0.4                                      # a small shift on a large shared mean
    w, b = _fit_probe(x, y, c, eps, RECIPE['probe_l2'])
    logits = x @ w.T + b
    assert (logits.argmax(1) == y).all()
    gap = logits.max(1).values - logits.min(1).values
    cap = math.log((1 - eps + eps / c) / (eps / c))                  # the gap label smoothing aims at: 2.94
    assert gap.mean() <= cap + 0.5 and gap.max() <= 2 * cap          # bounded, not driven to certainty


def test_a_probe_start_fits_the_head_once_and_only_for_a_new_model(tmp_path, monkeypatch, capsys):
    from rpclass.engine import trainer as T
    monkeypatch.setenv('RPCLASS_WORKERS', '0')
    monkeypatch.chdir(tmp_path)
    data = make_dataset(tmp_path / 'data')
    m = RPCLASS('s')
    monkeypatch.setitem(T.RECIPE, 'head_start', {m.cfg.backbone: 'probe'})
    m.train(data, imgsz=64, epochs=1, patience=None, batch=4, device='cpu', name='a')
    assert 'head: fitted on the frozen features of' in capsys.readouterr().out
    m.train(data, imgsz=64, epochs=1, patience=None, batch=4, device='cpu', name='b')   # from its own weights
    assert 'head: fitted' not in capsys.readouterr().out
