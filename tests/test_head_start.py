# RP-CLASS - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""How a new head starts (RECIPE head_start): lt's 'restart' (timm's head; if validation still scores what answering
one class does after the warmup, training starts over with the head as a scaled linear probe), the CNNs' timm head."""
import csv
import math

import numpy as np
import torch
import yaml

from rpclass import RPCLASS
from rpclass.engine import trainer as T
from rpclass.engine.trainer import RECIPE, _fit_probe, _scaled_probe, _single_class_f1, _stratified_order

from .conftest import make_dataset

TIMINGS = {'seconds', 'train_s', 'val_s'}


def test_only_lt_can_start_over():
    from rpclass.cfg.schema import SIZES
    starts = {size: RECIPE['head_start'].get(SIZES[size].backbone) for size in ('s', 'm', 'l', 'lt')}
    assert starts == {'s': None, 'm': None, 'l': None, 'lt': 'restart'}


def test_the_single_class_score():
    assert math.isclose(_single_class_f1([0] * 91 + [1] * 9, 2), 91 / 191)             # 0.476: a stuck binary run
    assert math.isclose(_single_class_f1(np.repeat(np.arange(8), 10), 8), 20 / 90 / 8)
    assert math.isclose(_single_class_f1([1, 1, 1], 3), 1.0)        # one class present: answering it is all right


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


def test_the_scaled_probe_keeps_its_direction_and_drops_its_confidence():
    g = torch.Generator().manual_seed(0)
    n, d, c, eps = 300, 32, 3, RECIPE['label_smoothing']
    y = torch.randint(0, c, (n,), generator=g)
    x = torch.randn(n, d, generator=g, dtype=torch.float64) * 0.05 + 1.0
    x[torch.arange(n), y] += 0.3
    w0, b0 = _fit_probe(x, y, c, eps, RECIPE['probe_l2'])
    w1, b1 = _scaled_probe(x, y, c, dict(RECIPE, probe_scale=1.0))
    assert torch.allclose(w1, w0) and torch.allclose(b1, b0)                          # scale 1: the probe itself
    w, b = _scaled_probe(x, y, c, dict(RECIPE, probe_scale=0.1))
    prior = torch.log((1 - eps) * torch.bincount(y, minlength=c).double() / n + eps / c)
    assert torch.allclose(x @ w.T + b - prior, 0.1 * (x @ w0.T + b0 - prior))        # a tenth of its departure


def test_a_stratified_order_keeps_the_class_shares_in_every_prefix():
    labels = np.array([0] * 91 + [1] * 9)
    order = _stratified_order(labels, np.random.default_rng(0))
    assert sorted(order.tolist()) == list(range(100))
    for k in (12, 23, 50):
        assert abs(int((labels[order[:k]] == 1).sum()) - 0.09 * k) <= 1


def _train(tmp_path, monkeypatch, name, epochs=4, **recipe):
    monkeypatch.setenv('RPCLASS_WORKERS', '0')
    monkeypatch.chdir(tmp_path)
    data = tmp_path / 'data'
    if not data.exists():
        make_dataset(data, splits={'train': 6, 'val': 3})
    m = RPCLASS('s')
    for key, value in recipe.items():
        monkeypatch.setitem(T.RECIPE, key, value)
    res = m.train(data, imgsz=64, epochs=epochs, patience=None, batch=4, device='cpu', name=name, augment=None)
    return m, res


def _rows(run):
    return [{k: v for k, v in r.items() if k not in TIMINGS} for r in csv.DictReader(open(run / 'results.csv'))]


def test_a_run_still_answering_one_class_after_the_warmup_starts_over_once(tmp_path, monkeypatch, capsys):
    """restart_margin 1: every validation counts as 'one class'. The run looks once, at the first validation after the
    warmup (epoch 2 here), starts over in its own folder with the head as a probe, and does not look again."""
    m, res = _train(tmp_path, monkeypatch, 'r', head_start={RPCLASS('s').cfg.backbone: 'restart'}, restart_margin=1.0,
                    restart_min_epochs=2, warmup_steps=2)
    out = capsys.readouterr().out
    assert out.count('training starts over from the pretrained weights') == 1
    assert out.count('head: a linear probe of the pretrained features') == 1
    run = res['dir']
    assert run.resolve() == (tmp_path / 'runs' / 'train' / 'r').resolve()
    assert not (tmp_path / 'runs' / 'train' / 'r2').exists()
    args = yaml.safe_load((run / 'args.yaml').read_text())
    assert args['started_over'] is True and args['head_start'] == 'probe'
    assert [int(r['epoch']) for r in _rows(run)] == [1, 2, 3, 4]       # the second attempt's epochs only
    assert (run / 'weights' / 'best.pt').exists()


def test_a_run_that_learns_is_the_run_without_the_look(tmp_path, monkeypatch):
    """restart_margin -1: no validation counts as 'one class'. The look changes nothing, number for number."""
    backbone = RPCLASS('s').cfg.backbone
    _, looked = _train(tmp_path, monkeypatch, 'a', head_start={backbone: 'restart'}, restart_margin=-1.0,
                       restart_min_epochs=1, warmup_steps=1)
    _, plain = _train(tmp_path, monkeypatch, 'b', head_start={}, warmup_steps=1)
    assert _rows(looked['dir']) == _rows(plain['dir'])
    a = torch.load(looked['dir'] / 'weights' / 'best.pt', weights_only=True)['state_dict']
    b = torch.load(plain['dir'] / 'weights' / 'best.pt', weights_only=True)['state_dict']
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_the_look_waits_for_the_warmup_and_needs_an_epoch_left(tmp_path, monkeypatch, capsys):
    """A warmup that lasts the whole run: the first validation after it is the last one, with no epoch left to start
    over for, so the run never starts over."""
    _train(tmp_path, monkeypatch, 'w', head_start={RPCLASS('s').cfg.backbone: 'restart'}, restart_margin=1.0,
           restart_min_epochs=1, warmup_steps=2000, warmup_max_fraction=1.0)
    assert 'starts over' not in capsys.readouterr().out


def test_a_probe_start_fits_the_head_once_and_only_for_a_new_model(tmp_path, monkeypatch, capsys):
    backbone = RPCLASS('s').cfg.backbone
    m, _ = _train(tmp_path, monkeypatch, 'p', epochs=1, head_start={backbone: 'probe'})
    assert 'head: a linear probe of the pretrained features of 18 training images, scaled by 0.1' in \
        capsys.readouterr().out
    m.train(tmp_path / 'data', imgsz=64, epochs=1, patience=None, batch=4, device='cpu', name='q')   # its weights
    assert 'head:' not in capsys.readouterr().out


def test_a_probe_start_reads_only_its_share_of_the_time(tmp_path, monkeypatch, capsys):
    """With a time budget the probe reads at most probe_time_share of the time left; 0 leaves one batch (4 images),
    too few for a probe, and the head starts as timm makes it."""
    monkeypatch.setenv('RPCLASS_WORKERS', '0')
    monkeypatch.chdir(tmp_path)
    data = make_dataset(tmp_path / 'data', splits={'train': 6, 'val': 3})
    m = RPCLASS('s')
    monkeypatch.setitem(T.RECIPE, 'head_start', {m.cfg.backbone: 'probe'})
    monkeypatch.setitem(T.RECIPE, 'probe_time_share', 0.0)
    m.train(data, imgsz=64, epochs=2, patience=None, batch=4, device='cpu', name='t', time=600, final_eval=False,
            augment=None)
    assert 'head: no linear probe: its time share read 4 of 18 training images, too few' in capsys.readouterr().out
