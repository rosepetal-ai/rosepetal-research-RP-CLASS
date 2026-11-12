# RP-CLASS - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""How a new head starts: its learning-rate factor and the linear-probe start (RECIPE head_lr_factor, head_start)."""
import math

import torch

from rpclass import RPCLASS
from rpclass.engine.trainer import RECIPE, _fit_probe, _param_groups

from .conftest import make_dataset, make_net


def test_a_head_factor_multiplies_only_the_head_lr():
    net = make_net()
    head = {id(p) for p in net.head_parameters()}
    for factor in (1.0, 15.0):
        for g in _param_groups(net, RECIPE, factor):
            for p in g['params']:
                want = RECIPE['lr'] * factor if id(p) in head else RECIPE['backbone_lr']
                assert g['base_lr'] == want
    assert all(g['base_lr'] in (RECIPE['lr'], RECIPE['backbone_lr']) for g in _param_groups(net, RECIPE, 1.0))


def test_the_recipe_starts_every_head_as_before():
    assert RECIPE['head_lr_factor'] == {} and RECIPE['head_start'] == {}


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
