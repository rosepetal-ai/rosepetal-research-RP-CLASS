# RP-CLASS - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""Training against a time budget (RP-OD's, without the mosaic): costs by kind of epoch, the phases as shares of the
time, the rules that keep finished phases finished, and a real run on a fake clock. The log lines that tools parse are
pinned here."""
import csv
import math
import re
from types import SimpleNamespace

import numpy as np
import pytest

import rpclass.data.augment as A
from rpclass import RPCLASS
from rpclass.engine.trainer import RECIPE, Budget, Schedule, _plan_epoch, train

from .conftest import make_dataset


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def budget(seconds, plain=(), clean=(), reserve=None):
    """A Budget on a fake clock that has run these epochs back to back, in this order."""
    clock = Clock()
    b = Budget(seconds, 0.0, reserve, clock=clock)
    for kind, times in (('plain', plain), ('clean', clean)):
        for s in times:
            clock.now += s
            b.record(s, kind, 0.0, 1, 0.0)
    return b


def schedule(epochs=100, spe=10):
    return Schedule(epochs, spe, dict(RECIPE, warmup_steps=2000))


def plan(b, s, done, clean_from=None, early=False, cap=100):
    return _plan_epoch(b, s, done, cap, clean_from, early, RECIPE)


def test_each_kind_of_epoch_costs_its_own_and_borrows_from_the_other():
    b = budget(1000, plain=[9, 7, 7, 7, 7])
    assert b.cost('plain') == 7 and b.cost('clean') == 7          # the last three; clean not measured: plain's
    assert budget(1000, plain=[5], clean=[4]).cost('clean') == 4
    assert budget(1000, clean=[4]).cost('plain') == 4
    assert budget(10).cost('plain') == 0


def test_the_next_epoch_is_clean_and_last_when_two_do_not_fit():
    s = schedule()
    assert plan(budget(65, plain=[10] * 5), s, 5) == 5           # 15 s left: one epoch, not two
    assert s.end == 6 and s.final_start == 5
    s = schedule()
    plan(budget(55, plain=[10] * 5), s, 5)                       # 5 s left: not even the clean one
    assert s.end == 5


def test_decay_and_clean_stretch_start_at_their_share_of_the_time():
    s = schedule()
    clean_from = plan(budget(100, plain=[1] * 79), s, 79)
    assert s.final_start > 79 and clean_from is None             # the next one's middle at 79.5 %: not yet
    s = schedule()
    clean_from = plan(budget(100, plain=[1] * 80), s, 80)
    assert s.final_start == 80 and s.end == 100 and clean_from is None   # 80.5 %: the decay starts
    s = schedule()
    assert plan(budget(100, plain=[1] * 96), s, 96) == 96        # 96.5 %: clean from here to the end


def test_the_end_follows_the_measured_cost():
    s = schedule()
    plan(budget(1000, plain=[5.5] * 4), s, 4, cap=500)
    assert s.end == 4 + 2 + int((1000 - 5.5 * 4 - 11) // 5.5)


def test_a_running_decay_follows_the_time_left_without_a_jump_in_the_learning_rate():
    s = schedule()
    s.start_decay(80, 10)
    before = s.factor(84, 9)
    plan(budget(1000, plain=[1] * 85), s, 85)                    # far more time than planned: it ends later
    assert s.end > 90 and s.factor(85, 0) == pytest.approx(before, abs=0.02) and s.factor(85, 0) <= before
    assert s.factor(s.end - 1, 9) == pytest.approx(RECIPE['final_lr_ratio'], abs=1e-3)
    end = s.end
    plan(budget(87, plain=[1] * 85), s, 85)                      # and earlier with less
    assert s.end == 87 < end


def test_after_patience_time_only_brings_the_end_forward():
    s = schedule()
    s.start_decay(30, 6)
    plan(budget(1000, plain=[1] * 31), s, 31, early=True)
    assert s.end == 36
    plan(budget(34, plain=[1] * 31), s, 31, early=True)
    assert s.end == 34


def test_the_warmup_shrinks_to_the_budget_from_where_it_is():
    """RP-OD's test_rp_od_6: a warmup of 2000 steps in a run of 126 trained at 6 % of the learning rate at most.
    Resized after 20 steps to what the budget holds, it goes on from its current factor to 1, never in one jump."""
    s = schedule(500, 126)                                       # 5 % of 63000 steps: capped at 2000
    assert s.warmup == 2000
    s.resize_warmup(12, 20)
    assert s.warmup == 40
    assert s.factor(0, 19) == pytest.approx(20 / 2000)
    assert s.factor(0, 20) - s.factor(0, 19) < 0.06 and s.factor(0, 39) == 1 and s.factor(0, 40) == 1
    s = schedule(500, 126)
    s.resize_warmup(300, 20)
    assert s.warmup == 300 and s.factor(1, 9) < 1


def test_a_time_budget_tells_the_augmentation_when_images_are_clean():
    sample = A.Sample(np.zeros((40, 40, 3), np.uint8), 0, 'x')
    ctx = SimpleNamespace(epoch=99, epochs=100, imgsz=[32, 32], rng=np.random.default_rng(0), clean=None)
    assert A.default_augment(sample, ctx) is sample               # epoch 100 of 100: the clean stretch
    ctx.epoch, ctx.clean = 99, False
    A.P_PHOTOMETRIC, A.P_FLIP = 1.0, 1.0
    try:
        assert A.default_augment(sample, ctx) is not sample       # the budget says not clean yet
    finally:
        A.P_PHOTOMETRIC, A.P_FLIP = 0.5, 0.5
    ctx.epoch, ctx.clean = 2, True
    assert A.default_augment(sample, ctx) is sample


class SteppedBudget(Budget):
    """Every epoch takes `step` seconds of a fake clock, whatever it really took."""

    def __init__(self, seconds, step, reserve=None):
        self.fake = Clock()
        super().__init__(seconds, 0.0, reserve, clock=self.fake)
        self.step = step

    def record(self, seconds, kind, val_s, n_val, save_s):
        self.fake.now += self.step
        super().record(self.step, kind, val_s, n_val, save_s)


EPOCH_LINE = re.compile(r'^epoch (\d+)/(\d+) \[(flat|final)\]  lr \d\.\d{2}e[+-]\d{2}  loss \d+\.\d{4}  \d+s '
                        r'\(train \d+\.\ds, val \d+\.\ds\)$', re.M)
VAL_LINE = re.compile(r'^val  top1 \d\.\d{3}  top5 (\d\.\d{3}|-)  macro-F1 \d\.\d{3}  bal-acc \d\.\d{3}  '
                      r'loss \d+\.\d{4}(  \* best)?$', re.M)


def test_training_to_a_budget_end_to_end(tmp_path, monkeypatch, capsys):
    """45 s at 10 s per epoch: 4 epochs, the last one clean. The log lines keep their exact format, the whole line
    (RP-OD pinned only their start)."""
    monkeypatch.setenv('RPCLASS_WORKERS', '0')
    data = make_dataset(tmp_path / 'data', {'train': 2, 'val': 1})
    seen = {}

    def record(sample, ctx):
        seen.setdefault(ctx.epoch, set()).add(ctx.clean)
        return sample

    m = RPCLASS('s')
    b = SteppedBudget(45, 10)
    res = train(m, str(data), [64, 64], 100, None, 2, 'cpu', record, 't', project=tmp_path / 'runs', budget=b,
                final_eval=False, pretrained=False)
    assert res['planned_epochs'] == 4 and res['stop_reason'] == 'TIME_BUDGET_REACHED'
    assert res['budget_overrun_s'] == pytest.approx(-5)
    assert res['val'] is None and res['test'] is None
    assert m.ckpt_path == res['dir'] / 'weights' / 'best.pt'
    assert seen[0] == seen[1] == seen[2] == {False} and seen[3] == {True}
    out = capsys.readouterr().out
    epochs = EPOCH_LINE.findall(out)
    assert [(int(e), int(n)) for e, n, _ in epochs] == [(1, 4), (2, 4), (3, 4), (4, 4)]
    assert len(VAL_LINE.findall(out)) == 4
    # 3 steps an epoch, too few to time the first steps: the warmup is sized after the first epoch to its share of
    # the 4 planned epochs, so the run reaches the full learning rate and ends below it
    lrs = [float(r['lr']) for r in csv.DictReader(open(res['dir'] / 'results.csv'))]
    assert max(lrs) == pytest.approx(RECIPE['backbone_lr']) and lrs[-1] < max(lrs)


def test_the_warmup_never_outlasts_a_run_that_ends_on_its_epochs(tmp_path, monkeypatch):
    """time=10 min for 30 epochs that take seconds: the warmup was sized from the steps the time holds (thousands),
    longer than the whole run, which then never reached its learning rate. Now the epochs cap it: 5 % of 120 steps."""
    monkeypatch.setenv('RPCLASS_WORKERS', '0')
    monkeypatch.chdir(tmp_path)
    data = make_dataset(tmp_path / 'data', {'train': 4, 'val': 1})            # 12 images, batch 3: 4 steps an epoch
    res = RPCLASS('s').train(data, imgsz=32, epochs=30, patience=None, batch=3, device='cpu', pretrained=False,
                             time=600, final_eval=False, augment=None)
    lrs = [float(r['lr']) for r in csv.DictReader(open(res['dir'] / 'results.csv'))]
    assert res['planned_epochs'] == 30 and res['stop_reason'] == 'EPOCHS_COMPLETED'
    assert lrs[2] == pytest.approx(RECIPE['backbone_lr']) and lrs[-1] < lrs[2]


def test_a_decay_that_starts_inside_the_warmup_still_reaches_its_floor_without_a_jump():
    """The warmup alone used to rule while it lasted: a decay started inside it climbed instead, and one entirely
    inside it never came down. Now the lower of the two rules, so the decay always ends on its floor."""
    s = schedule(100, 10)                                         # warmup: 50 steps, 5 epochs
    s.start_decay(2, 3)                                           # patience or the budget: decay over epochs 2-4
    factors = [s.factor(1, 9)] + [s.factor(e, k) for e in range(2, 5) for k in range(10)]
    steepest = max(1 / 50, (1 - RECIPE['final_lr_ratio']) * math.pi / (2 * 30))   # a warmup step, the decay's steepest
    assert max(abs(b - a) for a, b in zip(factors, factors[1:])) <= steepest + 1e-9
    assert factors[-1] == pytest.approx(RECIPE['final_lr_ratio'], abs=5e-3)     # the last of 30 steps
    assert max(factors) < 1                                       # it never reached the top: the decay met it first


def test_a_time_budget_call_with_a_reserve_function(tmp_path, monkeypatch):
    """A training service's call: time=, reserve as a function of the measured stats, patience as a function of the
    planned epochs, final_eval=False, its own project folder. On a fake clock of 10 s per epoch, 45 s with 20 reserved
    leave room for 2 epochs (4 without the reserve)."""
    import rpclass.model as M
    monkeypatch.setenv('RPCLASS_WORKERS', '0')
    monkeypatch.setattr(M, 'Budget', lambda seconds, started, reserve: SteppedBudget(seconds, 10, reserve))
    data = make_dataset(tmp_path / 'data', {'train': 2, 'val': 1})
    seen, asked = [], []
    res = RPCLASS('s').train(data, imgsz=64, epochs=500, time=45, reserve=lambda s: seen.append(dict(s)) or 20,
                             patience=lambda e: asked.append(e) or 50, batch=2, device='cpu', pretrained=False,
                             final_eval=False, project=tmp_path / 'service', name='run', workers=None)
    assert set(res) == {'dir', 'best_epoch', 'val', 'test', 'planned_epochs', 'stop_reason', 'budget_overrun_s'}
    assert res['planned_epochs'] == 2 and res['stop_reason'] == 'TIME_BUDGET_REACHED' and asked
    assert set(seen[-1]) == {'epoch_s', 'val_s', 'n_val', 'val_s_per_image'} and seen[-1]['n_val'] == 3
    assert res['dir'] == tmp_path / 'service' / 'run' and (res['dir'] / 'weights' / 'best.pt').exists()


def test_patience_none_or_a_function_of_the_planned_epochs(tmp_path, monkeypatch):
    monkeypatch.setenv('RPCLASS_WORKERS', '0')
    monkeypatch.chdir(tmp_path)
    data = make_dataset(tmp_path / 'data', {'train': 2, 'val': 1})
    res = RPCLASS('s').train(data, imgsz=64, epochs=3, patience=None, batch=2, device='cpu', pretrained=False,
                             final_eval=False)
    assert res['planned_epochs'] == 3 and res['stop_reason'] == 'EPOCHS_COMPLETED'
    asked = []
    res = RPCLASS('s').train(data, imgsz=64, epochs=10, patience=lambda e: asked.append(e) or 0, batch=2,
                             device='cpu', pretrained=False, final_eval=False)
    assert asked[0] == 10 and res['stop_reason'] == 'EARLY_STOPPED' and res['planned_epochs'] == 3
