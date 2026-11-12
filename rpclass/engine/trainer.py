# RP-CLASS - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""Training with one fixed recipe. The user chooses data, size, epochs, patience, batch, device and an optional
augmentation function; everything else is decided here.

The schedule is RP-OD's: warmup, then a flat learning rate, then a cosine decay over the last 20 % of the epochs. If
validation does not improve for `patience` epochs during the flat phase, the final decay starts right away, from the
weights of the best epoch so far, and training ends when it is done. The last 4 % of the epochs (at least one: the
clean images of the default augmentation) also start again from the best epoch's weights.

With a time budget the same phases are shares of the time instead of the epochs (Budget, _plan_epoch): training stops
between epochs, and the warmup is sized to the steps the budget holds.

The best epoch has the highest macro-F1 on val; among equal ones, the lowest val loss; on an exact tie, the latest. Its
EMA weights go to best.pt. Patience counts the epochs without a higher macro-F1, so a saturated validation set (every
image right) stops improving even if its loss still falls. The test split, if any, is evaluated once at the end and
never used for a decision.

Training runs in reduced precision (bfloat16 where the GPU has it natively, else float16); validation in float32, as
model.val() and predict() compute it. A classifier's scores hang on a few borderline images, and reduced precision
moved 1 or 2 of 53 test images in 78 of 135 runs of a k-fold benchmark on private sets (docs/EXPERIMENTS.md): enough
to change the best epoch, and to report other numbers than a later model.val(). Validation is a small share of an
epoch's time.

RECIPE: RP-OD's structure with the usual values for fine-tuning a classifier (a new head at 10x the learning rate of
the pretrained backbone), measured in the recipe experiments (docs/EXPERIMENTS.md, 2026-11-06): both learning rates
doubled lose 1.1 points of macro-F1 on average and halved change nothing consistent, so they stay; label smoothing 0.1
adds 0.3 points on average and, on one private OK / NOK set, cuts the OK images a threshold must reject to catch every
NOK from 81 and 95 to 23 and 28 of 160. It caps a confidence at 1 - 0.1 + 0.1 / C (0.95 for two classes); the validation
loss is the plain negative log-likelihood, without it. The other values are still RP-OD's or the usual ones.
"""
import contextlib
import copy
import csv
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from ..cfg.schema import derive, from_dict, to_dict
from ..data import ClassificationDataset, cached, check_dataset
from ..data.loader import MicroBatches
from ..data.check import format_problems
from ..nn import Classifier
from ..utils.checkpoint import load_weights, save_checkpoint
from .validator import validate

RECIPE = dict(lr=1e-3, backbone_lr=1e-4, weight_decay=0.05, betas=(0.9, 0.999), label_smoothing=0.1,
              warmup_steps=2000, warmup_max_fraction=0.05, final_fraction=0.2, final_lr_ratio=0.05, ema_decay=0.9999,
              ema_tau=2000, clip_grad=1.0, freeze_backbone_bn_below=8, clean_fraction=0.04, restore_best=True, seed=0,
              head_lr_factor={}, head_start={}, probe_l2=0.03, probe_images=4096)
# head_lr_factor {backbone: factor}: the new head's lr times this for that backbone. head_start {backbone: 'probe'}: the
# new head starts as a linear probe on the frozen features of the training images (_probe_head). Both empty: every
# backbone trains as before, number for number. Under test for lt (docs/EXPERIMENTS.md).
NO_DECAY = ('pos_embed', 'token')  # besides biases and norms (1-D): position embeddings and class tokens of ViTs
PROGRESS_SECONDS = 120           # one progress line this often inside an epoch (none in epochs shorter than this)
FINAL_EVAL_MARGIN = 1.5          # the final evaluation of val and test, with its plots, in time per validated image
TABLE_CLASSES = 20               # the per-class table goes to the log every epoch up to this many classes
MAIN_THREADS = 4                 # CPU threads of the main process while it trains on a GPU with workers (train())
CSV_FIELDS = ['epoch', 'phase', 'lr', 'loss', 'top1', 'top5', 'macro_f1', 'balanced_accuracy', 'val_loss', 'fitness',
              'best', 'seconds', 'train_s', 'val_s']


def log(msg=''):
    print(msg, flush=True)


class ModelEMA:
    """Exponential moving average of the weights; it is what gets validated and saved."""

    def __init__(self, model, decay, tau):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay, self.tau, self.updates = decay, tau, 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / self.tau))
        src, own = model.state_dict(), self.ema.state_dict()
        floats = [k for k, v in own.items() if v.dtype.is_floating_point]
        torch._foreach_lerp_([own[k] for k in floats], [src[k].detach() for k in floats], 1 - d)   # one call for all
        for k, v in own.items():
            if not v.dtype.is_floating_point:
                v.copy_(src[k])


class Schedule:
    """Learning-rate factor per optimizer step: warmup, flat, final cosine decay (which patience can bring forward)."""

    def __init__(self, epochs, steps_per_epoch, r):
        self.spe, self.r = steps_per_epoch, r
        self.final_epochs = max(1, round(r['final_fraction'] * epochs))
        self.final_start = epochs - self.final_epochs
        self.warmup = self._warmup(epochs)
        self.low = r['final_lr_ratio']
        self.decay_from, self.decay_p0 = None, 0.0     # set when a running decay is re-planned (time budget)
        self.warmup_from, self.warmup_f0 = 0, 0.0      # set when a running warmup is resized (time budget)

    def _warmup(self, epochs):
        return max(1, min(self.r['warmup_steps'], round(self.r['warmup_max_fraction'] * epochs * self.spe)))

    @property
    def end(self):
        return self.final_start + self.final_epochs

    def resize_warmup(self, steps, done):
        """Time budget: the warmup now lasts `steps` (at most warmup_steps), counted from the start, after `done` steps
        of it. The factor goes on from where it is to 1 over what is left, at least `done` more steps: a budget that
        holds fewer steps than the warmup it started with must not jump to the full learning rate."""
        steps = max(1, min(self.r['warmup_steps'], steps))
        if done >= self.warmup:
            return
        f0 = done / self.warmup if self.warmup_from == 0 else self._warmup_factor(done - 1)
        self.warmup_from, self.warmup_f0, self.warmup = done, f0, max(steps, 2 * done)

    def _warmup_factor(self, g):
        if self.warmup_from == 0:
            return (g + 1) / self.warmup
        return self.warmup_f0 + (1 - self.warmup_f0) * (g + 1 - self.warmup_from) / (self.warmup - self.warmup_from)

    def start_decay(self, epoch, epochs):
        """The final decay starts at `epoch` and lasts `epochs` (patience, or the time budget reaching it)."""
        self.final_start, self.final_epochs = epoch, epochs
        self.decay_from, self.decay_p0 = None, 0.0

    def plan_decay(self, end, done):
        """Time budget, the decay running: training now ends at `end`. What is left of the decay is spread over the
        epochs from `done` to `end`, so the learning rate goes on from where it is, without a jump, and still reaches
        its floor at the end."""
        if end == self.end:
            return
        self.decay_p0, self.decay_from = self._progress(done, 0), done
        self.final_epochs = end - self.final_start

    def plan_flat(self, end, done):
        """Time budget, before the decay: training is projected to end at `end`; the decay keeps its share of it but
        never starts before the next epoch (the budget starts it, when the time comes)."""
        final = max(1, round(self.r['final_fraction'] * end))
        self.final_start = max(done + 1, end - final)
        self.final_epochs = max(1, end - self.final_start)

    def phase(self, epoch):
        return 'final' if epoch >= self.final_start else 'flat'

    def _progress(self, epoch, step):
        if self.decay_from is None:
            p = ((epoch - self.final_start) * self.spe + step) / (self.final_epochs * self.spe)
        else:
            p = self.decay_p0 + (1 - self.decay_p0) * (((epoch - self.decay_from) * self.spe + step)
                                                       / ((self.end - self.decay_from) * self.spe))
        return min(max(p, 0.0), 1.0)

    def factor(self, epoch, step):
        """During the decay the lower of the warmup and the decay: a decay that starts while the warmup still rises (a
        run of one or two epochs, a short budget) climbs with the warmup until the decay meets it, then follows the
        decay to its floor, without a jump (before, the warmup alone ruled and could keep the floor from ever
        coming)."""
        g = epoch * self.spe + step
        decay = None
        if epoch >= self.final_start:
            decay = self.low + (1 - self.low) * 0.5 * (1 + math.cos(math.pi * self._progress(epoch, step)))
        if g < self.warmup:
            warm = self._warmup_factor(g)
            return warm if decay is None else min(warm, decay)
        return 1.0 if decay is None else decay


def _clean_start(end, r):
    """First epoch of the clean last stretch: the last clean_fraction of the epochs up to where training really ends (at
    least one), the same rule as rpclass.data.augment, so patience moves both together."""
    return end - max(1, round(r['clean_fraction'] * end))


class Budget:
    """Training against a time budget (RPCLASS.train(time=...)): the phases are fractions of the time, not of the
    epochs.

    `seconds` counts from `started` (the call to train, so the setup is included) to the end of train. `reserve` is
    time the caller needs afterwards (seconds, or reserve(stats) -> seconds with the stats below), taken off the budget
    after every epoch; `extra` is time rpclass itself needs at the end (its final evaluation). Epochs are measured by
    kind, since they can cost differently: plain (augmented) and clean; a kind not measured yet costs what the other
    one did. An epoch costs the mean of the last WINDOW of its kind; the first epoch of the run is measured without its
    first steps, which carry the CUDA and worker start-up, when it has enough of them to tell (4 or more)."""

    WINDOW = 3
    NEAREST = {'plain': ('plain', 'clean'), 'clean': ('clean', 'plain')}

    def __init__(self, seconds, started, reserve=None, clock=time.time):
        self.seconds, self.started, self.clock = float(seconds), started, clock
        self.reserve = reserve if callable(reserve) else (lambda stats, s=float(reserve or 0): s)
        self.extra = lambda stats: 0.0
        self.costs = {kind: [] for kind in self.NEAREST}
        self.val_s = self.n_val = self.save_s = 0

    def record(self, seconds, kind, val_s, n_val, save_s):
        self.costs[kind].append(seconds)
        self.val_s, self.n_val, self.save_s = val_s, n_val, save_s

    def cost(self, kind):
        """Seconds an epoch of this kind costs: measured, else what the other kind measured (0 before the first)."""
        for k in self.NEAREST[kind]:
            if self.costs[k]:
                last = self.costs[k][-self.WINDOW:]
                return sum(last) / len(last)
        return 0.0

    @property
    def epoch_s(self):
        return self.cost('plain')

    @property
    def stats(self):
        return {'epoch_s': self.epoch_s, 'val_s': self.val_s, 'n_val': self.n_val,
                'val_s_per_image': self.val_s / max(self.n_val, 1)}

    def elapsed(self):
        return self.clock() - self.started

    def remaining(self):
        """Seconds left for epochs: the budget minus what has run, the caller's reserve and rpclass's own close (reload
        the best weights, adopt them, and the final evaluation when there is one)."""
        return self.seconds - self.elapsed() - self.reserve(self.stats) - self.extra(self.stats) - 2 * self.save_s

    def fraction(self, seconds=0.0):
        """Share of the time for epochs used once `seconds` more have run."""
        total = self.elapsed() + self.remaining()
        return (self.elapsed() + seconds) / total if total > 0 else 1.0

    def overrun(self):
        """Seconds past the budget the caller left to rpclass (negative: time to spare)."""
        return self.elapsed() - (self.seconds - self.reserve(self.stats))

    def state(self):
        return {'seconds': self.seconds, 'elapsed': self.elapsed(), 'costs': self.costs, 'val_s': self.val_s,
                'n_val': self.n_val, 'save_s': self.save_s}

    def load(self, state):
        """Continue a resumed run with what was left of its budget."""
        self.started = self.clock() - state['elapsed']
        self.costs = {k: list(v) for k, v in state['costs'].items()}
        self.val_s, self.n_val, self.save_s = state['val_s'], state['n_val'], state['save_s']


def _plan_epoch(budget, schedule, done, cap, clean_from, early, r):
    """After `done` epochs, in time mode: what the next epoch is, from the time left and what each kind of epoch costs.
    Returns clean_from and moves the end of `schedule` (equal to `done`: training stops here).

    Training only stops between epochs, and the last epoch is always clean, so the next one is the last when it and a
    clean one do not both fit. The phases go by the share of the time at the middle of the next epoch: the final decay
    from 1 - final_fraction, clean from 1 - clean_fraction. A phase never comes back once over: the decay and the clean
    stretch only end sooner or later with the time left (the decay without a jump in the learning rate). After
    patience (`early`) time can only bring the end forward."""
    left, clean_s = budget.remaining(), budget.cost('clean')
    in_clean = clean_from is not None and clean_from < done
    next_s = clean_s if in_clean else budget.cost('plain')
    if done >= cap or left < clean_s:
        n = 0
    elif in_clean:
        n = math.floor(left / clean_s) if clean_s > 0 else cap - done
    elif left < next_s + clean_s:
        n = 1
    else:
        plain_s = budget.cost('plain')
        n = 2 + (math.floor((left - next_s - clean_s) / plain_s) if plain_s > 0 else cap)
    end = min(cap, done + n)
    if early:
        end = min(end, schedule.end)
    if end <= done:
        start = min(schedule.final_start, done)
        schedule.start_decay(start, done - start)
        return clean_from
    clean = (in_clean or end == done + 1 or budget.fraction(clean_s / 2) >= 1 - r['clean_fraction']
             or early and done >= _clean_start(end, r))        # after patience the end is known in epochs
    if clean and clean_from is None:
        clean_from = done
    if schedule.final_start <= done:
        schedule.plan_decay(end, done)
    elif clean or budget.fraction(next_s / 2) >= 1 - r['final_fraction']:
        schedule.start_decay(done, end - done)
    else:
        schedule.plan_flat(end, done)
    return clean_from


def _restore_best(path, net, ema):
    """The model and its EMA back to the best epoch's weights (the EMA saved in best.pt). The EMA restarts its warmup,
    so that it follows what gets trained from there instead of staying on the restored weights; the optimizer keeps its
    moments."""
    state_dict = torch.load(path, map_location='cpu', weights_only=True)['state_dict']
    net.load_state_dict(state_dict)
    ema.ema.load_state_dict(state_dict)
    ema.updates = 0


def _param_groups(net, r, head_factor=1.0):
    """The new head at lr (times head_factor), the pretrained backbone at backbone_lr; no weight decay on biases, norms,
    embeddings and tokens."""
    head = {id(p) for p in net.head_parameters()}
    groups = {}
    for n, p in net.named_parameters():
        if not p.requires_grad:
            continue
        part = 'head' if id(p) in head else 'backbone'
        decay = p.ndim > 1 and not any(k in n for k in NO_DECAY)
        groups.setdefault((part, decay), []).append(p)
    out = []
    for (part, decay), ps in groups.items():
        lr = (r['lr'] if head_factor == 1 else r['lr'] * head_factor) if part == 'head' else r['backbone_lr']
        out.append({'params': ps, 'lr': lr, 'base_lr': lr, 'weight_decay': r['weight_decay'] if decay else 0.0})
    return out


def _fit_probe(x, y, num_classes, eps, l2):
    """A multinomial logistic regression on features x [N, D] (float64) for labels y [N]: the label-smoothed
    cross-entropy of the recipe plus l2 / 2 * |W|^2, on features centred and divided by their global RMS, solved by
    L-BFGS. The bias starts at the log of the smoothed class prior. Returns (W [C, D], b [C]) on the raw features."""
    mu = x.mean(0)
    rms = (x - mu).pow(2).mean().sqrt().clamp_min(1e-12)
    z = (x - mu) / rms
    p = torch.bincount(y, minlength=num_classes).double() / len(y)
    w = torch.zeros(num_classes, x.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.log((1 - eps) * p + eps / num_classes).requires_grad_(True)
    ce = nn.CrossEntropyLoss(label_smoothing=eps)
    opt = torch.optim.LBFGS([w, b], lr=1, max_iter=500, history_size=20, line_search_fn='strong_wolfe',
                            tolerance_grad=1e-10, tolerance_change=1e-14)

    def closure():
        opt.zero_grad()
        loss = ce(z @ w.T + b, y) + 0.5 * l2 * w.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        w_raw = w / rms
        return w_raw, b - w_raw @ mu


def _probe_head(net, records, imgsz, num_classes, r, device, workers):
    """head_start 'probe': the new head becomes a linear probe of the frozen pretrained features of the training images,
    letterboxed as validation reads them (no augmentation), at most r['probe_images'] of them (a class-stratified draw
    with its own generator: the global random state is untouched). In eval mode and float32, before the optimizer and
    the EMA exist. A ViT whose head learns at AdamW's step size from a small pooled feature otherwise spends its first
    epochs at the class prior, and on a rare, small defect may never leave it (docs/EXPERIMENTS.md)."""
    from torch.utils.data import DataLoader, Subset
    from ..data import collate
    dataset = ClassificationDataset(records, imgsz)
    picks = np.arange(len(dataset))
    if len(picks) > r['probe_images']:
        rng = np.random.default_rng(r['seed'])
        labels = np.array([rec['label'] for rec in records])
        share = r['probe_images'] / len(picks)
        picks = np.sort(np.concatenate([rng.choice(np.flatnonzero(labels == c), round(share * (labels == c).sum()),
                                                   replace=False) for c in np.unique(labels)]))
    loader = DataLoader(Subset(dataset, picks.tolist()), batch_size=32, shuffle=False, num_workers=workers,
                        collate_fn=collate)
    was_training = net.training
    net.eval()
    feats, ys = [], []
    with torch.no_grad():
        for images, targets, _ in loader:
            x = (images.to(device, non_blocking=True) - net.mean) / net.std
            feats.append(net.body.forward_head(net.body.forward_features(x), pre_logits=True).float().cpu())
            ys.append(targets)
    net.train(was_training)
    x, y = torch.cat(feats).double(), torch.cat(ys)
    w, b = _fit_probe(x, y, num_classes, r['label_smoothing'], r['probe_l2'])
    head = net.body.get_classifier()
    with torch.no_grad():
        head.weight.copy_(w.to(head.weight))
        head.bias.copy_(b.to(head.bias))
        right = float(((x @ w.T + b).argmax(1) == y).double().mean())
    log(f'head: fitted on the frozen features of {len(y)} training images (train accuracy {right:.3f})')


def _set_train_mode(net, freeze_bn):
    net.train()
    if freeze_bn:
        for m in net.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()


def _freeze_bn(net):
    """Too few images per step for batch statistics: BatchNorm keeps its pretrained statistics and affine weights."""
    for m in net.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            for p in m.parameters():
                p.requires_grad_(False)


def _amp(device):
    if device.type != 'cuda':
        return None
    # without including_emulation=False a T4 (sm75) answers True and trains in
    # emulated bfloat16, several times slower than its float16 tensor cores
    return torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16


def _autocast(device, dtype):
    return torch.autocast(device.type, dtype=dtype) if dtype else contextlib.nullcontext()


def _raise_open_file_limit(target=65536):
    """DataLoader workers hand every tensor to the main process as an open file: with many workers a limit of 1024
    open files (common in containers) runs out ('received 0 items of ancdata'). Raise the soft limit towards the hard
    one; nothing happens where that is not possible."""
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        want = target if hard == resource.RLIM_INFINITY else min(target, hard)
        if soft != resource.RLIM_INFINITY and soft < want:
            resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
    except (ImportError, ValueError, OSError):
        pass


def _micro_batch(net, criterion, batch, imgsz, num_classes, device, amp_dtype, headroom=0.8):
    """Largest divisor of `batch` whose forward + backward peaks under `headroom` of the GPU memory: the margin is
    for the optimizer, the EMA and fragmentation. The weights are left untouched."""
    if device.type != 'cuda':
        return batch
    backup = {k: v.clone() for k, v in net.state_dict().items()}
    limit = headroom * torch.cuda.get_device_properties(device).total_memory
    for micro in [d for d in range(batch, 0, -1) if batch % d == 0]:
        images = labels = out = None
        torch.cuda.reset_peak_memory_stats(device)
        try:
            images = torch.rand(micro, 3, *imgsz, device=device)
            labels = torch.randint(0, num_classes, (micro,), device=device)
            with _autocast(device, amp_dtype):
                out = net(images)
            criterion(out.float(), labels).backward()
            torch.cuda.synchronize(device)
            ok = torch.cuda.max_memory_allocated(device) <= limit or micro == 1
        except torch.cuda.OutOfMemoryError:
            ok = False
        net.zero_grad(set_to_none=True)
        images = labels = out = None
        torch.cuda.empty_cache()
        if ok:
            net.load_state_dict(backup)
            return micro
    # 'out of memory' in the message: a job runner can recognise it, for example to retry on a larger GPU
    raise RuntimeError(f'out of memory: the model does not fit in GPU memory even with 1 image per step at '
                       f'{imgsz[0]}x{imgsz[1]}')


def _progress(epoch, epochs, step, steps, loss, images, seconds, width=20):
    done = step / steps
    bar = '█' * round(width * done) + '░' * (width - round(width * done))
    left = seconds / step * (steps - step) / 60
    return (f'  epoch {epoch + 1}/{epochs} [{bar}] {100 * done:3.0f}% · {step}/{steps} steps · loss {loss:.3f} · '
            f'{images / seconds:.0f} img/s · ~{left:.0f} min left')


def _run_dir(project, name):
    base = Path(project) / name
    path, k = base, 2
    while path.exists():
        path, k = Path(f'{base}{k}'), k + 1
    path.mkdir(parents=True)
    return path


def epoch_line(epoch, end, phase, lr, loss, seconds, train_s, val_s):
    """The end of an epoch, one line. Its format is a contract: tools that follow a training from its log parse it
    (pinned by the tests)."""
    return (f'epoch {epoch}/{end} [{phase}]  lr {lr:.2e}  loss {loss:.4f}  {seconds:.0f}s '
            f'(train {train_s:.1f}s, val {val_s:.1f}s)')


def val_line(m, best):
    """The validation of an epoch, one line; a contract too. top5 is '-' with 5 classes or fewer."""
    top5 = f'{m.top5:.3f}' if m.top5 >= 0 else '-'
    return (f'val  top1 {m.top1:.3f}  top5 {top5}  macro-F1 {m.macro_f1:.3f}  bal-acc {m.balanced_accuracy:.3f}  '
            f'loss {m.loss:.4f}' + ('  * best' if best else ''))


def train(model, data, imgsz, epochs, patience, batch, device, augment, name, project='runs/train', workers=None,
          save_state=False, resume=None, budget=None, final_eval=True, pretrained=True, keep_orientation=False):
    """Trains `model` (an RPCLASS) in place and returns {'dir', 'best_epoch', 'val', 'test', 'planned_epochs',
    'stop_reason', 'budget_overrun_s'}.

    patience: epochs without a higher macro-F1 before the final decay starts early; None never; a function
    patience(planned_epochs) -> int is evaluated again whenever the end of training moves.

    budget: a Budget, to train for a time instead of a number of epochs (`epochs` is then the most it may train). The
    phases are shares of the time: the decay and the clean stretch start at the epoch whose middle falls past their
    share, and after every epoch _plan_epoch decides what the next one is from what each kind of epoch cost so far. It
    stops between epochs, never inside one.

    final_eval: evaluate the best weights on val and test (with plots) at the end. False: only reload and adopt them
    ('val' and 'test' are None), for a caller that evaluates on its own.

    pretrained: a new model starts from the pretrained weights of its backbone (False: random weights). A model loaded
    from a checkpoint starts from it.

    save_state: also write weights/state.pt after every epoch (optimizer, EMA, schedule, best so far), so a long run
    that gets interrupted can continue with resume=<that file> and the same arguments. Not part of RPCLASS.train.

    On a GPU with workers, the main process keeps MAIN_THREADS CPU threads while training. All it does on the CPU is
    copies and bookkeeping, and its default pool of one thread per core stalled with the cores busy preparing images.
    On a 24-core machine running another training, a validation of 100 images held in memory took 0.3-2.4 s with 24
    threads and 0.2-0.4 s with 4. The pool is restored afterwards.
    """
    device = torch.device(device)
    workers = int(os.environ.get('RPCLASS_WORKERS', min(8, os.cpu_count() or 1))) if workers is None else workers
    threads = torch.get_num_threads()
    if device.type != 'cpu' and workers > 0:
        torch.set_num_threads(min(threads, MAIN_THREADS))
    try:
        return _train(model, data, imgsz, epochs, patience, batch, device, augment, name, project, workers, save_state,
                      resume, budget, final_eval, pretrained, keep_orientation)
    finally:
        torch.set_num_threads(threads)


def _train(model, data, imgsz, epochs, patience, batch, device, augment, name, project, workers, save_state, resume,
           budget, final_eval, pretrained, keep_orientation):
    r = RECIPE
    state = torch.load(resume, map_location='cpu', weights_only=True) if resume else None
    if state and from_dict(state['args']['model']) != model.cfg:
        raise ValueError(f'{resume} is a run of another model configuration')
    torch.manual_seed(r['seed'])
    np.random.seed(r['seed'])
    random.seed(r['seed'])
    if workers > 0:
        _raise_open_file_limit()

    # ---- data
    ds = check_dataset(data)
    names, stats = ds['names'], ds['stats']
    if ds['warnings']:
        log(format_problems(ds['warnings'], f"{len(ds['warnings'])} warning(s) about the dataset"))
    split_counts = ' · '.join(f"{s} {stats[s]['images']} images" for s in ('train', 'val', 'test') if s in stats)
    log(f'data: {len(names)} classes · {split_counts}')
    log('classes (train/val): ' + ', '.join(f"{n} {stats['train']['per_class'][i]}/{stats['val']['per_class'][i]}"
                                            for i, n in names.items()))

    # ---- model
    d = derive(model.cfg, len(names), imgsz)
    net = Classifier(d, pretrained=pretrained and state is None and model.net is None)
    if state:
        net.load_state_dict(state['net'])
        log(f"resuming {state['args']['name']} at epoch {state['epoch'] + 1}")
    elif model.net is not None:
        report = load_weights(net, model.net.state_dict())
        log(f"starting from the current weights: {report['summary']}")
    net.to(device)
    amp_dtype = _amp(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=r['label_smoothing'], reduction='sum')

    train_set = ClassificationDataset(ds['records']['train'], imgsz, augment, epochs=epochs, seed=r['seed'])
    val_set = ClassificationDataset(ds['records']['val'], imgsz)
    t_read = time.time()
    val_eval = cached(val_set, workers)        # read and letterboxed once, if it fits (rpclass/data/dataset.py)
    val_workers = 0 if val_eval is not val_set else workers
    read_per_image = (time.time() - t_read) / max(len(val_set), 1) if val_eval is not val_set else 0.0
    micro = state['micro'] if state else _micro_batch(net, criterion, batch, imgsz, len(names), device, amp_dtype)
    accumulate = math.ceil(batch / micro)
    per_step = min(micro, len(train_set))     # a training split smaller than the micro-batch is one smaller step
    freeze_bn = per_step < r['freeze_backbone_bn_below']
    if freeze_bn:
        _freeze_bn(net)
    log(f'batch {batch} = {micro} images x {accumulate} accumulation step(s)'
        + (f' · BatchNorm frozen ({per_step} images per step < {r["freeze_backbone_bn_below"]})' if freeze_bn else '')
        + f" · precision {str(amp_dtype).replace('torch.', '') if amp_dtype else 'float32'} · device {device}")
    # the order of an epoch depends on (seed, epoch) only: the same with any number of workers, and after a resume;
    # every micro-batch is prepared by all the workers at once (rpclass/data/loader.py)
    order = torch.Generator()
    loader = MicroBatches(train_set, micro, order, workers, pin=device.type == 'cuda', seed=r['seed'])
    steps_per_epoch = max(1, math.ceil(len(loader) / accumulate))
    backbone = model.cfg.backbone
    if state is None and model.net is None and r['head_start'].get(backbone) == 'probe':
        _probe_head(net, ds['records']['train'], imgsz, len(names), r, device, workers)
    optimizer = torch.optim.AdamW(_param_groups(net, r, r['head_lr_factor'].get(backbone, 1.0)), betas=r['betas'])
    scaler = torch.amp.GradScaler('cuda', enabled=amp_dtype == torch.float16)
    ema = ModelEMA(net, r['ema_decay'], r['ema_tau'])
    schedule = Schedule(epochs, steps_per_epoch, r)
    params = [p for p in net.parameters() if p.requires_grad]
    if budget is not None and final_eval:     # the final evaluation of val and test is part of the time too
        n_test = stats.get('test', {}).get('images', 0)
        n_eval = stats['val']['images'] + n_test
        # validations of a cached val skip reading; the test is read and letterboxed: that is priced apart
        budget.extra = lambda s: FINAL_EVAL_MARGIN * (s['val_s_per_image'] * n_eval + read_per_image * n_test)

    best_key, best_f1, best_epoch, stale, epoch = None, -1.0, -1, 0, 0
    if state:
        ema.ema.load_state_dict(state['ema'])
        ema.updates = state['ema_updates']
        optimizer.load_state_dict(state['optimizer'])
        if state['scaler']:                   # empty: the run did not use float16; a fresh scaler calibrates itself
            scaler.load_state_dict(state['scaler'])
        schedule.final_start = state['final_start']
        best_key = tuple(state['best_key']) if state['best_key'] is not None else None
        best_f1, best_epoch, stale, epoch = state['best_f1'], state['best_epoch'], state['stale'], state['epoch']
    clean_from, early = None, False           # clean_from: first clean epoch, fixed once it has started
    if budget is not None and epochs == 1:
        clean_from = 0                        # the only epoch is the last one, clean as in epoch mode
    if state:
        clean_from, early = state['clean_from'], state['early']
    if state and budget is not None:
        schedule.final_epochs, schedule.warmup = state['final_epochs'], state['warmup']
        schedule.decay_from, schedule.decay_p0 = state['decay_from'], state['decay_p0']
        schedule.warmup_from, schedule.warmup_f0 = state['warmup_from'], state['warmup_f0']
        budget.load(state['budget'])
    train_set.set_epochs(schedule.end)        # the augmentation's phases follow the real end of training
    first_epoch = epoch                       # its first steps carry the start-up: left out of its cost

    # ---- run folder
    run = Path(state['run']) if state else _run_dir(project, name)
    if not state:
        (run / 'weights').mkdir()
    args = {'data': str(Path(data).resolve()), 'imgsz': list(imgsz), 'epochs': epochs,
            'patience': getattr(patience, '__name__', 'function') if callable(patience) else patience,
            'time': budget.seconds if budget is not None else None, 'batch': batch, 'device': str(device),
            'augment': getattr(augment, '__name__', None) if augment else None,
            'shrink_factor': train_set.shrink_factor,
            'keep_orientation': bool(keep_orientation), 'name': run.name,
            'model': to_dict(model.cfg), 'classes': names, 'micro_batch': micro, 'accumulate': accumulate,
            'bn_frozen': freeze_bn, 'pretrained': bool(pretrained),
            'recipe': {k: list(v) if isinstance(v, tuple) else v for k, v in r.items()}}
    if state:
        args['name'] = state['args']['name']
    else:
        (run / 'args.yaml').write_text(yaml.safe_dump(args, sort_keys=False, allow_unicode=True))
    csv_file = open(run / 'results.csv', 'a' if state else 'w', newline='')
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    if not state:
        writer.writeheader()
    csv_file.flush()
    extra = {'data': args['data'], 'keep_orientation': args['keep_orientation']}
    log(f'results: {run}')

    try:
        while epoch < schedule.end:
            t0 = time.time()
            if budget is None and clean_from is None and epoch >= _clean_start(schedule.end, r):
                clean_from = epoch                   # time mode: _plan_epoch decides it
            clean = clean_from is not None and epoch >= clean_from
            if r['restore_best'] and epoch == clean_from and 0 <= best_epoch < epoch - 1:
                _restore_best(run / 'weights' / 'best.pt', net, ema)
                log(f'\nepochs {clean_from + 1}-{schedule.end}: back to the weights of the best epoch '
                    f'({best_epoch + 1})')
            if budget is not None:
                train_set.set_clean(clean)
            train_set.set_epoch(epoch)
            _set_train_mode(net, freeze_bn)
            loss_sum, done, seen, next_report = torch.zeros((), device=device), 0, 0, t0 + PROGRESS_SECONDS
            # the first epoch of the run is timed from step `warm`, and in time mode its steps `warm`-2*`warm` size the
            # warmup to the steps the budget holds, not to `epochs` (only a cap there); sized again after the epoch
            warm = min(10, steps_per_epoch // 4) if epoch == first_epoch else 0
            t_warm = None
            order.manual_seed(r['seed'] * 1_000_003 + epoch)
            it = iter(loader)
            for step in range(steps_per_epoch):
                factor = schedule.factor(epoch, step)
                for g in optimizer.param_groups:
                    g['lr'] = g['base_lr'] * factor
                group = []
                for _ in range(accumulate):
                    try:
                        group.append(next(it))
                    except StopIteration:
                        break
                if not group:
                    break
                total = sum(len(labels) for _, labels, _ in group)
                for images, labels, _ in group:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    with _autocast(device, amp_dtype):
                        logits = net(images)
                    loss = criterion(logits.float(), labels) / total      # the mean over the effective batch
                    scaler.scale(loss).backward()
                    loss_sum += loss.detach()                             # on the device: read only for the log
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, r['clip_grad'])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(net)
                done += 1
                seen += total
                if warm and done in (warm, 2 * warm):
                    if device.type == 'cuda':
                        torch.cuda.synchronize(device)
                    if done == warm:
                        t_warm = time.time()
                    elif budget is not None and not state:
                        step_s = (time.time() - t_warm) / warm
                        if step_s > 0:          # never more steps than `epochs` holds
                            steps = min(epochs * steps_per_epoch, (budget.elapsed() + budget.remaining()) / step_s)
                            schedule.resize_warmup(round(r['warmup_max_fraction'] * steps), done)
                if time.time() >= next_report:
                    next_report = time.time() + PROGRESS_SECONDS
                    log(_progress(epoch, schedule.end, step + 1, steps_per_epoch, float(loss_sum) / done, seen,
                                  time.time() - t0))
            train_loss = float(loss_sum) / max(done, 1)                   # per optimizer step (effective batch)

            t_val = time.time()
            metrics = validate(ema.ema, val_eval, names, device, micro, workers=val_workers)
            val_s = time.time() - t_val
            phase = schedule.phase(epoch)
            if metrics.nonfinite:
                log(f'warning: {metrics.nonfinite} validation images gave non-finite scores (has training diverged?); '
                    f'they count as wrong')
            # the best is the highest (macro-F1, -val loss), the latest on an exact tie; only a higher macro-F1 resets
            # patience: a saturated validation set (every image right) keeps lowering its loss without improving
            key = metrics.key()
            improved = best_key is None or key >= best_key
            if improved:
                if metrics.macro_f1 > best_f1:
                    stale = 0
                elif phase == 'flat':
                    stale += 1
                best_key, best_f1, best_epoch = key, metrics.macro_f1, epoch
                save_checkpoint(run / 'weights' / 'best.pt', ema.ema, model.cfg, names, imgsz,
                                {**metrics.to_dict(), 'epoch': epoch + 1}, extra)
            elif phase == 'flat':
                stale += 1
            t_save = time.time()
            save_checkpoint(run / 'weights' / 'last.pt', ema.ema, model.cfg, names, imgsz,
                            {**metrics.to_dict(), 'epoch': epoch + 1}, extra)
            save_s = time.time() - t_save
            seconds = time.time() - t0
            wait = patience(schedule.end) if callable(patience) else patience
            stopped = phase == 'flat' and wait is not None and stale >= wait and epoch + 1 < schedule.final_start
            back = False
            if stopped:
                schedule.start_decay(epoch + 1, schedule.final_epochs)
                early = True
                back = r['restore_best'] and best_epoch < epoch
                if back:
                    _restore_best(run / 'weights' / 'best.pt', net, ema)
            if budget is not None:     # plan before the log, so that 'epoch N/E' shows the current projection
                cost = seconds
                if t_warm is not None and done > warm:       # the steps before `warm`, at the pace of the rest
                    cost = (t_val - t_warm) * done / (done - warm) + time.time() - t_val
                budget.record(cost, 'clean' if clean else 'plain', val_s, len(val_set), save_s)
                before, had_clean = schedule.end, clean_from
                clean_from = _plan_epoch(budget, schedule, epoch + 1, epochs, clean_from, early, r)
                if epoch == first_epoch and not state:
                    # the warmup to its share of the run as now projected: a whole measured epoch (validation and
                    # saves included) and the epochs cap, also for epochs too short to be timed by their steps
                    schedule.resize_warmup(round(r['warmup_max_fraction'] * schedule.end * steps_per_epoch),
                                           (epoch + 1) * steps_per_epoch)
            train_set.set_epochs(schedule.end)
            row = {k: v for k, v in metrics.to_dict().items() if k != 'loss'}
            writer.writerow({'epoch': epoch + 1, 'phase': phase, 'lr': float(f"{factor * r['backbone_lr']:.4g}"),
                             'loss': round(train_loss, 5), **{k: round(v, 5) for k, v in row.items()},
                             'val_loss': round(metrics.loss, 5), 'best': int(improved), 'seconds': round(seconds, 1),
                             'train_s': round(t_val - t0, 1), 'val_s': round(val_s, 1)})
            csv_file.flush()
            log('\n' + epoch_line(epoch + 1, schedule.end, phase, factor * r['backbone_lr'], train_loss, seconds,
                                  t_val - t0, val_s))
            log(val_line(metrics, improved))
            if len(names) <= TABLE_CLASSES:
                log(metrics.table(names))
            if stopped:
                log(f'no improvement for {wait} epochs: '
                    + (f'back to the weights of the best epoch ({best_epoch + 1}), ' if back else '')
                    + f'starting the final learning-rate decay ({schedule.final_epochs} epochs, ends at epoch '
                      f'{schedule.end})')
            if budget is not None:
                log(f'time: {budget.remaining() / 60:.1f} min left for epochs ({100 * budget.fraction():.0f} % used), '
                    f'~{budget.epoch_s:.0f}s each · training ends at epoch {schedule.end}'
                    + (f' (was {before})' if schedule.end != before else ''))
                if had_clean is None and clean_from is not None and epoch + 1 < schedule.end:
                    log(f'time: clean images from epoch {clean_from + 1}')
                if epoch + 1 == 1 and schedule.end == 1 and not clean:
                    log('warning: the first epoch used up the time budget; training ends without a clean epoch')
            if save_state:
                tmp = run / 'weights' / 'state.pt.tmp'
                torch.save({'run': str(run.resolve()), 'epoch': epoch + 1, 'net': net.state_dict(),
                            'ema': ema.ema.state_dict(), 'ema_updates': ema.updates,
                            'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                            'final_start': schedule.final_start, 'best_key': list(best_key), 'best_f1': best_f1,
                            'best_epoch': best_epoch, 'stale': stale, 'micro': micro, 'args': args,
                            'final_epochs': schedule.final_epochs, 'warmup': schedule.warmup,
                            'decay_from': schedule.decay_from, 'decay_p0': schedule.decay_p0,
                            'warmup_from': schedule.warmup_from, 'warmup_f0': schedule.warmup_f0,
                            'clean_from': clean_from, 'early': early,
                            'budget': budget.state() if budget is not None else None}, tmp)
                os.replace(tmp, run / 'weights' / 'state.pt')
            epoch += 1
    finally:
        csv_file.close()

    # ---- final: best weights, plots, test
    best = torch.load(run / 'weights' / 'best.pt', map_location='cpu', weights_only=True)
    net = Classifier(derive(model.cfg, len(names), imgsz), pretrained=False)
    net.load_state_dict(best['state_dict'])
    net.to(device).eval()
    log(f'\nbest epoch {best_epoch + 1} (macro-F1 {best_f1:.4f})')
    model._adopt(net, names, imgsz, run / 'weights' / 'best.pt', args['data'], args['keep_orientation'])
    result = {'dir': run, 'best_epoch': best_epoch + 1, 'val': None, 'test': None, 'planned_epochs': schedule.end,
              'stop_reason': ('EARLY_STOPPED' if early
                              else 'TIME_BUDGET_REACHED' if budget is not None and schedule.end < epochs
                              else 'EPOCHS_COMPLETED'),
              'budget_overrun_s': None}
    if not final_eval:
        if budget is not None:
            result['budget_overrun_s'] = budget.overrun()
        return result
    val_metrics = validate(net, val_eval, names, device, micro, plots_dir=run / 'val', workers=val_workers)
    test_metrics = None
    if ds['records'].get('test'):
        test_set = ClassificationDataset(ds['records']['test'], imgsz)
        test_metrics = validate(net, test_set, names, device, micro, plots_dir=run / 'test', workers=workers)
        log('\ntest (never used for any decision)')
        log(val_line(test_metrics, False).replace('val ', 'test', 1))
        log(test_metrics.table(names))
    result.update(val=val_metrics, test=test_metrics)
    if budget is not None:
        result['budget_overrun_s'] = budget.overrun()
    return result
