# RP-CLASS - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""Training: the schedule, the parameter groups, accumulation, a whole run, resume, and the rules that pick the best
epoch and bring the end forward."""
import csv

import numpy as np
import pytest
import torch
import yaml

from rpclass import RPCLASS
from rpclass.engine.trainer import RECIPE, ModelEMA, Schedule, _clean_start, _param_groups, _restore_best

from .conftest import NAMES, make_dataset, make_net


def test_schedule_warmup_flat_final_and_patience():
    s = Schedule(epochs=10, steps_per_epoch=100, r=dict(RECIPE, warmup_steps=50))
    assert s.final_start == 8 and s.end == 10
    assert s.factor(0, 0) == pytest.approx(1 / 50) and s.factor(0, 49) == 1 and s.factor(5, 0) == 1
    assert s.factor(8, 0) == pytest.approx(1) and s.factor(9, 99) == pytest.approx(RECIPE['final_lr_ratio'], abs=1e-3)
    s.final_start = 4   # patience triggered after epoch 4
    assert s.end == 6 and s.phase(4) == 'final' and s.factor(5, 99) < 0.1


def test_the_head_and_the_backbone_have_their_own_learning_rate_and_no_decay_on_1d():
    net = make_net()
    names = {id(p): n for n, p in net.named_parameters()}
    head = {id(p) for p in net.head_parameters()}
    groups = _param_groups(net, RECIPE)
    assert sum(len(g['params']) for g in groups) == len(names)
    for g in groups:
        for p in g['params']:
            assert g['base_lr'] == (RECIPE['lr'] if id(p) in head else RECIPE['backbone_lr'])
            assert (g['weight_decay'] == 0) == (p.ndim == 1)


def test_a_vit_decays_its_patch_embedding_but_not_its_tokens():
    """The patch embedding is a convolution like any other; only position embeddings and tokens go without decay."""
    from rpclass.cfg.schema import Architecture
    net = make_net(arch=Architecture('vit-s16-dinov3'))
    decay = {id(p): g['weight_decay'] > 0 for g in _param_groups(net, RECIPE) for p in g['params']}
    named = dict(net.named_parameters())
    assert decay[id(named['body.patch_embed.proj.weight'])]
    tokens = [n for n in named if 'token' in n or 'pos_embed' in n]
    assert tokens and not any(decay[id(named[n])] for n in tokens)


def test_accumulation_equals_one_big_batch():
    """Two micro-batches of 2, each loss divided by the 4 images of the effective batch, give the gradient of one
    batch of 4 (BatchNorm in eval: batch statistics differ by construction)."""
    net = make_net()
    net.train()
    for m in net.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()
    crit = torch.nn.CrossEntropyLoss(reduction='sum')
    x, y = torch.rand(4, 3, 64, 64), torch.tensor([0, 1, 2, 1])
    (crit(net(x), y) / 4).backward()
    big = [p.grad.clone() for p in net.parameters() if p.grad is not None]
    net.zero_grad()
    for i in (0, 2):
        (crit(net(x[i:i + 2]), y[i:i + 2]) / 4).backward()
    small = [p.grad.clone() for p in net.parameters() if p.grad is not None]
    worst = max(float((a - b).abs().max() / (a.abs().max() + 1e-8)) for a, b in zip(big, small))
    assert worst < 1e-4


def test_train_end_to_end(tmp_path, monkeypatch, capsys):
    from rpclass.engine import trainer as T
    monkeypatch.setattr(T, 'PROGRESS_SECONDS', 0)                    # a progress line after every step
    monkeypatch.setenv('RPCLASS_WORKERS', '0')
    monkeypatch.chdir(tmp_path)
    data = make_dataset(tmp_path / 'data')
    m = RPCLASS('s')
    res = m.train(data, imgsz=64, epochs=3, patience=1, batch=4, device='cpu', name='t')
    run = res['dir']
    assert '  epoch 1/3 [' in capsys.readouterr().out
    rows = list(csv.DictReader(open(run / 'results.csv')))
    assert 1 <= len(rows) <= 3 and set(T.CSV_FIELDS) == set(rows[0])
    args = yaml.safe_load(open(run / 'args.yaml'))
    assert args['micro_batch'] == 4 and args['accumulate'] == 1 and args['classes'] == NAMES
    assert args['augment'] == 'default_augment' and args['keep_orientation'] is False
    for f in ('weights/best.pt', 'weights/last.pt', 'val/confusion_matrix.png', 'val/predictions.jpg',
              'val/metrics.json', 'test/metrics.json'):
        assert (run / f).exists(), f
    assert m.names == NAMES and m.imgsz == [64, 64] and m.ckpt_path == run / 'weights' / 'best.pt'
    assert m.keep_orientation is False and RPCLASS(run / 'weights' / 'best.pt').keep_orientation is False
    assert res['val'].images == 6 and res['test'].images == 6
    assert len(m.predict(np.zeros((50, 70, 3), np.uint8), device='cpu')) == 1
    assert m.val(split='test', device='cpu').images == 6
    reloaded = RPCLASS(run / 'weights' / 'best.pt')
    assert reloaded.data == str(data.resolve()) and reloaded.names == NAMES


def test_a_model_learns_the_synthetic_classes(tmp_path, monkeypatch):
    """Not only plumbing: three classes of distinct colors are learned in a few epochs. The tests have no ImageNet
    weights, and the recipe's learning rates are for a pretrained backbone: from random weights the backbone needs
    the head's. Batch 4 keeps BatchNorm frozen; training it from random weights on 3 steps an epoch makes the running
    statistics lag behind the weights, and validation collapses. One thread, so that the result does not hang on how
    the machine splits its sums: with 8 images a class and 10 epochs, a 4-thread CI runner once failed to learn what
    the same code learned on the next run; with 16 and 20, five seeds out of five reach 1.0 (chance is 0.33)."""
    from rpclass.engine import trainer as T
    monkeypatch.setitem(T.RECIPE, 'backbone_lr', 1e-3)
    monkeypatch.setenv('RPCLASS_WORKERS', '0')
    monkeypatch.chdir(tmp_path)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        data = make_dataset(tmp_path / 'data', {'train': 16, 'val': 6})
        res = RPCLASS('s').train(data, imgsz=64, epochs=20, patience=None, batch=4, device='cpu', name='t',
                                 augment=None)
    finally:
        torch.set_num_threads(threads)
    assert res['val'].macro_f1 >= 0.9


def test_keep_orientation_and_the_default_input_size(monkeypatch, trained_ckpt, tmp_path):
    """keep_orientation picks the augmentation without flips or turns, and a checkpoint keeps it; imgsz=None is the
    size's default (384 for the CNNs, 448 for lt) or a checkpoint's own size."""
    import rpclass.model as M
    from rpclass.utils.checkpoint import load_checkpoint, save_checkpoint
    got = {}

    def fake_train(model, data, imgsz, epochs, patience, batch, device, augment, *args, **kw):
        got.update(imgsz=imgsz, augment=augment, keep=kw['keep_orientation'])

    monkeypatch.setattr(M, '_train', fake_train)
    M.RPCLASS('m').train('data', keep_orientation=True)
    assert got == {'imgsz': [384, 384], 'augment': M.upright_augment, 'keep': True}
    M.RPCLASS('lt').train('data')
    assert got == {'imgsz': [448, 448], 'augment': M.default_augment, 'keep': False}
    M.RPCLASS('s').train('data', augment=None, keep_orientation=True)
    assert got['augment'] is None and got['keep'] is True
    M.RPCLASS('s').train('data', augment=M.upright_augment, keep_orientation=True)
    assert got['augment'] is M.upright_augment
    with pytest.raises(ValueError, match='keep_orientation'):
        M.RPCLASS('s').train('data', augment=lambda s, c: s, keep_orientation=True)
    M.RPCLASS(trained_ckpt).train('data')
    assert got['imgsz'] == [64, 96] and got['augment'] is M.default_augment
    # a checkpoint trained with the switch keeps it, and training it again inherits it
    ckpt = load_checkpoint(trained_ckpt)
    upright = tmp_path / 'upright.pt'
    save_checkpoint(upright, M.Classifier(M.derive(ckpt['config'], 3, [64, 96]), pretrained=False), ckpt['config'],
                    ckpt['names'], [64, 96], extra={'keep_orientation': True})
    model = M.RPCLASS(upright)
    assert model.keep_orientation is True
    model.train('data')
    assert got['augment'] is M.upright_augment and got['keep'] is True
    custom = lambda s, c: s  # noqa: E731
    model.train('data', augment=custom)                     # an inherited switch leaves a custom function alone
    assert got['augment'] is custom
    model.train('data', keep_orientation=False)
    assert got['augment'] is M.default_augment and got['keep'] is False


def test_the_val_cache_changes_no_number(tmp_path, monkeypatch):
    """Validation reads its split once and keeps it in memory (rpclass/data/dataset.py, CachedSplit): every epoch's
    metrics and the final ones are the same to the last digit as when it reads the images every time."""
    import rpclass.data.dataset as D
    monkeypatch.setenv('RPCLASS_WORKERS', '0')
    monkeypatch.chdir(tmp_path)
    data = make_dataset(tmp_path / 'data', {'train': 4, 'val': 3, 'test': 2})
    rows = {}
    for name, budget in (('cached', D.CACHE_BYTES), ('read', 0)):
        monkeypatch.setattr(D, 'CACHE_BYTES', budget)
        res = RPCLASS('s').train(data, imgsz=64, epochs=3, patience=None, batch=4, device='cpu', name=name)
        rows[name] = [{k: v for k, v in r.items() if k not in ('seconds', 'train_s', 'val_s')}
                      for r in csv.DictReader(open(res['dir'] / 'results.csv'))]
        rows[name].append((res['val'].macro_f1, res['val'].loss, res['test'].macro_f1, res['test'].loss))
    assert rows['cached'] == rows['read']


def test_micro_batches_from_chunks_change_no_number(tmp_path, monkeypatch):
    """The training loader joins every micro-batch from chunks prepared by several workers (rpclass/data/loader.py):
    every epoch's numbers and the final ones are those of whole micro-batches prepared in this process."""
    import rpclass.data.loader as L
    monkeypatch.chdir(tmp_path)
    data = make_dataset(tmp_path / 'data', {'train': 5, 'val': 3, 'test': 2})
    rows = {}
    for name, chunk, workers in (('whole', None, 0), ('chunks', 1, 2)):
        monkeypatch.setattr(L, 'CHUNK', chunk)
        res = RPCLASS('s').train(data, imgsz=64, epochs=3, patience=None, batch=4, device='cpu', name=name,
                                 workers=workers)
        rows[name] = [{k: v for k, v in r.items() if k not in ('seconds', 'train_s', 'val_s')}
                      for r in csv.DictReader(open(res['dir'] / 'results.csv'))]
        rows[name].append((res['val'].macro_f1, res['val'].loss, res['test'].macro_f1, res['test'].loss))
    assert rows['chunks'] == rows['whole']


def test_the_main_process_keeps_few_threads_while_training_on_a_gpu(tmp_path, monkeypatch):
    """train() on a GPU with workers runs with MAIN_THREADS CPU threads and gives the pool back, also after an error;
    on the CPU, or without workers, it keeps them all."""
    import rpclass.engine.trainer as T
    seen = []

    def fake(*args, **kw):
        seen.append(torch.get_num_threads())
        raise RuntimeError('stop')

    monkeypatch.setattr(T, '_train', fake)
    monkeypatch.setattr(T, 'MAIN_THREADS', 1)
    before = torch.get_num_threads()
    for device, workers in (('cuda', 2), ('cpu', 2), ('cuda', 0)):
        with pytest.raises(RuntimeError, match='stop'):
            T.train(None, tmp_path, [32, 32], 1, None, 4, device, None, 'x', workers=workers)
        assert torch.get_num_threads() == before
    assert seen == [1, before, before]


def test_pre_ampere_cards_train_in_float16(monkeypatch):
    # torch answers bf16 "supported" on a T4 by emulating it; only sm80+ has it natively
    from rpclass.engine import trainer as T
    cuda = torch.device('cuda')
    native = {'T4': False, 'L4': True}
    for card, has_bf16 in native.items():
        monkeypatch.setattr(torch.cuda, 'is_bf16_supported',
                            lambda including_emulation=True, n=has_bf16: n or including_emulation)
        assert T._amp(cuda) == (torch.bfloat16 if has_bf16 else torch.float16), card
    assert T._amp(torch.device('cpu')) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a GPU')
def test_float16_training_runs(tmp_path, monkeypatch):
    from rpclass.engine import trainer as T
    monkeypatch.setattr(T, '_amp', lambda device: torch.float16)
    monkeypatch.setenv('RPCLASS_WORKERS', '0')
    monkeypatch.chdir(tmp_path)
    data = make_dataset(tmp_path / 'data')
    res = RPCLASS('s').train(data, imgsz=64, epochs=2, batch=4, device='cuda', name='t')
    rows = list(csv.DictReader(open(res['dir'] / 'results.csv')))
    assert rows and all(np.isfinite(float(r['val_loss'])) for r in rows)


def test_interrupted_run_resumes(tmp_path, monkeypatch):
    """save_state writes the whole training state every epoch; resume continues the same run where it stopped."""
    from rpclass.engine import trainer as T
    monkeypatch.setenv('RPCLASS_WORKERS', '0')
    monkeypatch.chdir(tmp_path)
    data = make_dataset(tmp_path / 'data', {'train': 2, 'val': 1})

    def crash_in_third_epoch(sample, ctx):
        if ctx.epoch == 2:
            raise KeyboardInterrupt
        return sample

    with pytest.raises(KeyboardInterrupt):
        T.train(RPCLASS('s'), data, [64, 64], 4, 4, 2, 'cpu', crash_in_third_epoch, 't', save_state=True,
                pretrained=False)
    run = tmp_path / 'runs' / 'train' / 't'
    assert torch.load(run / 'weights' / 'state.pt', weights_only=True)['epoch'] == 2
    with pytest.raises(ValueError, match='another model configuration'):
        T.train(RPCLASS('m'), data, [64, 64], 4, 4, 2, 'cpu', None, 't', resume=run / 'weights' / 'state.pt')
    res = T.train(RPCLASS('s'), data, [64, 64], 4, 4, 2, 'cpu', None, 't', save_state=True,
                  resume=run / 'weights' / 'state.pt')
    assert res['dir'] == run and not (tmp_path / 'runs' / 'train' / 't2').exists()
    assert [int(r['epoch']) for r in csv.DictReader(open(run / 'results.csv'))] == [1, 2, 3, 4]
    assert torch.load(run / 'weights' / 'state.pt', weights_only=True)['epoch'] == 4


def test_clean_stretch_follows_the_real_end():
    assert [_clean_start(end, RECIPE) for end in (120, 150, 60, 10, 3)] == [115, 144, 58, 9, 2]


def test_restore_best_puts_back_model_and_ema(tmp_path):
    best, net = make_net(), make_net()
    with torch.no_grad():
        for p in net.parameters():
            p.add_(1.0)
    ema = ModelEMA(net, 0.9999, 2000)
    ema.updates = 500
    torch.save({'state_dict': best.state_dict()}, tmp_path / 'best.pt')
    _restore_best(tmp_path / 'best.pt', net, ema)
    for m in (net, ema.ema):
        assert all(torch.equal(v, best.state_dict()[k]) for k, v in m.state_dict().items())
    assert ema.updates == 0                     # the EMA follows what gets trained from the best weights


def scripted(monkeypatch, scores):
    """Training validation as usual but with these (macro-F1, val loss) values, epoch after epoch (the last one
    repeats); every return to the best weights is recorded as (epochs finished, the model now holds best.pt)."""
    from rpclass.engine import trainer as T
    real_validate, real_restore, done, restores = T.validate, T._restore_best, [], []

    def validate(net, dataset, names, device, batch=32, plots_dir=None, **kwargs):
        m = real_validate(net, dataset, names, device, batch, plots_dir, **kwargs)
        if plots_dir is None:
            m.macro_f1, m.loss = scores[min(len(done), len(scores) - 1)]
            m.fitness = m.macro_f1
            done.append(m.macro_f1)
        return m

    def restore(path, net, ema):
        real_restore(path, net, ema)
        saved = torch.load(path, map_location='cpu', weights_only=True)['state_dict']
        restores.append((len(done), all(torch.equal(v.cpu(), saved[k]) for k, v in net.state_dict().items())))

    monkeypatch.setattr(T, 'validate', validate)
    monkeypatch.setattr(T, '_restore_best', restore)
    return restores


def test_the_clean_last_epochs_start_from_the_best(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('RPCLASS_WORKERS', '0')
    monkeypatch.chdir(tmp_path)
    data = make_dataset(tmp_path / 'data', {'train': 2, 'val': 1})
    restores = scripted(monkeypatch, [(0.5, 1.0), (0.9, 0.5), (0.1, 2.0)])
    res = RPCLASS('s').train(data, imgsz=64, epochs=10, patience=100, batch=2, device='cpu', pretrained=False)
    assert restores == [(9, True)]                      # before the 10th epoch, the last 4 % (at least one)
    assert 'epochs 10-10: back to the weights of the best epoch (2)' in capsys.readouterr().out
    assert res['best_epoch'] == 2


def test_patience_starts_the_decay_from_the_best(tmp_path, monkeypatch, capsys):
    """patience 2: no improvement in epochs 3 and 4, so the final decay (2 epochs) starts from epoch 2's weights,
    and the clean last epoch (the 6th) starts from them again."""
    monkeypatch.setenv('RPCLASS_WORKERS', '0')
    monkeypatch.chdir(tmp_path)
    data = make_dataset(tmp_path / 'data', {'train': 2, 'val': 1})
    restores = scripted(monkeypatch, [(0.5, 1.0), (0.9, 0.5), (0.1, 2.0)])
    res = RPCLASS('s').train(data, imgsz=64, epochs=10, patience=2, batch=2, device='cpu', pretrained=False)
    assert restores == [(4, True), (5, True)]
    out = capsys.readouterr().out
    assert 'no improvement for 2 epochs: back to the weights of the best epoch (2), starting the final' in out
    assert len(list(csv.DictReader(open(res['dir'] / 'results.csv')))) == 6


def test_a_saturated_validation_keeps_the_latest_lower_loss_but_runs_out_of_patience(tmp_path, monkeypatch, capsys):
    """Every validation image right from epoch 2 on while the loss keeps falling: each epoch takes over as the best
    (equal macro-F1, lower loss), but none is an improvement for patience, which then starts the decay from the latest
    weights instead of going back to the first epoch at the top."""
    monkeypatch.setenv('RPCLASS_WORKERS', '0')
    monkeypatch.chdir(tmp_path)
    data = make_dataset(tmp_path / 'data', {'train': 2, 'val': 1})
    restores = scripted(monkeypatch, [(0.5, 1.0)] + [(1.0, 1.0 / k) for k in range(2, 20)])
    res = RPCLASS('s').train(data, imgsz=64, epochs=10, patience=2, batch=2, device='cpu', pretrained=False)
    out = capsys.readouterr().out
    rows = list(csv.DictReader(open(res['dir'] / 'results.csv')))
    assert 'no improvement for 2 epochs: starting the final' in out      # after epochs 3 and 4
    assert restores == [] and len(rows) < 10 and all(r['best'] == '1' for r in rows)
    assert res['best_epoch'] == len(rows)


def test_a_worse_loss_at_equal_macro_f1_does_not_take_over(tmp_path, monkeypatch):
    monkeypatch.setenv('RPCLASS_WORKERS', '0')
    monkeypatch.chdir(tmp_path)
    data = make_dataset(tmp_path / 'data', {'train': 2, 'val': 1})
    scripted(monkeypatch, [(0.8, 0.5), (0.8, 0.9), (0.8, 0.5)])
    res = RPCLASS('s').train(data, imgsz=64, epochs=3, patience=None, batch=2, device='cpu', pretrained=False)
    assert res['best_epoch'] == 3            # epoch 2 is worse (higher loss); epoch 3 ties epoch 1: the latest wins


def test_open_file_limit_is_raised_for_dataloader_workers():
    """Workers pass each tensor as an open file: a limit of 1024 broke a COCO validation with 16 workers (RP-OD)."""
    resource = pytest.importorskip('resource')
    from rpclass.engine.trainer import _raise_open_file_limit
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if hard != resource.RLIM_INFINITY and hard < 4096:
        pytest.skip('hard limit too low to test')
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (1024, hard))
        _raise_open_file_limit()
        expected = 65536 if hard == resource.RLIM_INFINITY else min(65536, hard)
        assert resource.getrlimit(resource.RLIMIT_NOFILE)[0] == expected
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a GPU')
def test_the_micro_batch_probe_always_accepts_one_image(monkeypatch):
    from rpclass.engine import trainer as T
    net = make_net().cuda()
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda device=None: float('inf'))
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda device: type('P', (), {'total_memory': 1})())
    crit = torch.nn.CrossEntropyLoss(reduction='sum')
    assert T._micro_batch(net, crit, 4, [64, 64], 3, torch.device('cuda'), None) == 1


def test_the_micro_batch_probe_says_out_of_memory_when_nothing_fits(monkeypatch):
    """The error says 'out of memory', so a job runner can recognise it (to retry on a larger GPU, say); RP-OD's
    message lacks the words. Runs on the CPU: the CUDA calls are stubbed and every forward runs out of memory."""
    from rpclass.engine import trainer as T

    class Hungry(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x):
            raise torch.cuda.OutOfMemoryError('CUDA out of memory')

    real_rand, real_randint = torch.rand, torch.randint
    monkeypatch.setattr(torch, 'rand', lambda *a, device=None, **k: real_rand(*a, **k))
    monkeypatch.setattr(torch, 'randint', lambda *a, device=None, **k: real_randint(*a, **k))
    for name in ('reset_peak_memory_stats', 'synchronize', 'empty_cache'):
        monkeypatch.setattr(torch.cuda, name, lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda device: type('P', (), {'total_memory': 1})())
    crit = torch.nn.CrossEntropyLoss(reduction='sum')
    with pytest.raises(RuntimeError, match='out of memory'):
        T._micro_batch(Hungry(), crit, 4, [64, 64], 3, torch.device('cuda'), None)


def test_accumulation_in_the_trainer_equals_one_big_batch(tmp_path, monkeypatch):
    """The same run with batch 4 in one step or in two micro-batches of 2: the same weights (BatchNorm frozen in both:
    fewer than 8 images a step)."""
    from rpclass.engine import trainer as T
    monkeypatch.setenv('RPCLASS_WORKERS', '0')
    monkeypatch.chdir(tmp_path)
    data = make_dataset(tmp_path / 'data', {'train': 4, 'val': 1})
    runs = []
    for micro in (4, 2):
        monkeypatch.setattr(T, '_micro_batch', lambda *a, m=micro, **k: m)
        res = RPCLASS('s').train(data, imgsz=32, epochs=2, patience=None, batch=4, device='cpu', pretrained=False,
                                 final_eval=False, augment=None, name=f'm{micro}')
        assert yaml.safe_load(open(res['dir'] / 'args.yaml'))['accumulate'] == 4 // micro
        runs.append(torch.load(res['dir'] / 'weights' / 'last.pt', weights_only=True)['state_dict'])
    worst = max(float((runs[0][k].float() - runs[1][k].float()).abs().max() / (runs[0][k].float().abs().max() + 1e-8))
                for k in runs[0] if runs[0][k].is_floating_point())
    assert worst < 1e-4


def test_batchnorm_is_frozen_when_the_training_split_is_smaller_than_8(tmp_path, monkeypatch, capsys):
    """Batch 32 but 3 training images: every step has 3 images, too few for batch statistics."""
    monkeypatch.setenv('RPCLASS_WORKERS', '0')
    monkeypatch.chdir(tmp_path)
    data = make_dataset(tmp_path / 'data', {'train': 1, 'val': 1})
    RPCLASS('s').train(data, imgsz=32, epochs=1, batch=32, device='cpu', pretrained=False, final_eval=False)
    assert 'BatchNorm frozen (3 images per step < 8)' in capsys.readouterr().out


def test_a_resumed_run_keeps_its_early_stop(tmp_path, monkeypatch):
    """Interrupted in the decay that patience started: the resumed run still ends 'EARLY_STOPPED'."""
    from rpclass.engine import trainer as T
    monkeypatch.setenv('RPCLASS_WORKERS', '0')
    monkeypatch.chdir(tmp_path)
    data = make_dataset(tmp_path / 'data', {'train': 2, 'val': 1})
    restores = scripted(monkeypatch, [(0.5, 1.0), (0.9, 0.5), (0.1, 2.0)])

    def crash_in_fifth_epoch(sample, ctx):
        if ctx.epoch == 4:
            raise KeyboardInterrupt
        return sample

    with pytest.raises(KeyboardInterrupt):
        T.train(RPCLASS('s'), data, [64, 64], 10, 2, 2, 'cpu', crash_in_fifth_epoch, 't', save_state=True,
                pretrained=False)
    run = tmp_path / 'runs' / 'train' / 't'
    res = T.train(RPCLASS('s'), data, [64, 64], 10, 2, 2, 'cpu', None, 't', save_state=True,
                  resume=run / 'weights' / 'state.pt')
    assert res['stop_reason'] == 'EARLY_STOPPED' and res['planned_epochs'] == 6
    assert restores == [(4, True), (5, True)]           # patience after epoch 4, then the clean epoch 6


def test_a_run_resumed_inside_its_clean_stretch_does_not_go_back_to_the_best_again(tmp_path, monkeypatch):
    """A clean stretch of 3 epochs (8-10) starts from the best weights once; interrupted in epoch 9 and resumed, it
    goes on from where it was instead of throwing epoch 8 away."""
    from rpclass.engine import trainer as T
    monkeypatch.setitem(T.RECIPE, 'clean_fraction', 0.3)
    monkeypatch.setenv('RPCLASS_WORKERS', '0')
    monkeypatch.chdir(tmp_path)
    data = make_dataset(tmp_path / 'data', {'train': 2, 'val': 1})
    restores = scripted(monkeypatch, [(0.5, 1.0), (0.9, 0.5), (0.1, 2.0)])

    def crash_in_ninth_epoch(sample, ctx):
        if ctx.epoch == 8:
            raise KeyboardInterrupt
        return sample

    with pytest.raises(KeyboardInterrupt):
        T.train(RPCLASS('s'), data, [64, 64], 10, None, 2, 'cpu', crash_in_ninth_epoch, 't', save_state=True,
                pretrained=False)
    run = tmp_path / 'runs' / 'train' / 't'
    T.train(RPCLASS('s'), data, [64, 64], 10, None, 2, 'cpu', None, 't', save_state=True,
            resume=run / 'weights' / 'state.pt')
    assert restores == [(7, True)]


def test_training_on_an_exported_layout(tmp_path, monkeypatch):
    """validation/ instead of val/, an unlabeled/ split, every class folder in every split, an empty test split; and
    val() takes 'validation'."""
    monkeypatch.setenv('RPCLASS_WORKERS', '0')
    monkeypatch.chdir(tmp_path)
    data = make_dataset(tmp_path / 'data', {'train': 2, 'validation': 1, 'test': 0, 'unlabeled': 1})
    (data / 'train' / 'rare').mkdir()
    (data / 'validation' / 'rare').mkdir()
    m = RPCLASS('s')
    res = m.train(data, imgsz=32, epochs=1, batch=2, device='cpu', pretrained=False)
    assert m.names == {0: 'nok', 1: 'ok', 2: 'rare', 3: 'scratch'} and res['test'] is None
    assert m.val(split='validation', device='cpu').images == 3
