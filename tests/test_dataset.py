# RP-CLASS - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""Dataset folders: the layout, every error listed with its file, the warnings, the cache, the augmentation contract."""
import shutil

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from rpclass.data import (AugmentError, ClassificationDataset, DatasetError, Sample, check_dataset, class_map,
                          collate)

from .conftest import make_dataset


def test_the_classes_are_the_sorted_folder_names(tmp_path):
    root = make_dataset(tmp_path / 'd', classes=('scratch', 'OK', 'dent'))
    ds = check_dataset(root)
    assert ds['names'] == {0: 'OK', 1: 'dent', 2: 'scratch'}
    assert ds['stats']['train'] == {'images': 12, 'per_class': [4, 4, 4]}
    assert {r['label'] for r in ds['records']['val'] if 'scratch_' in r['path']} == {2}


def test_validation_may_be_called_val_validation_or_valid(tmp_path):
    root = make_dataset(tmp_path / 'd', {'train': 2, 'validation': 1})
    ds = check_dataset(root)
    assert ds['stats']['val']['images'] == 3 and 'test' not in ds['records']
    assert any('no test images' in w[3] for w in ds['warnings'])
    shutil.move(root / 'validation', root / 'valid')
    assert check_dataset(root)['stats']['val']['images'] == 3


def test_a_missing_or_empty_split_is_an_error(tmp_path):
    with pytest.raises(DatasetError, match="a 'val' folder is required"):
        check_dataset(make_dataset(tmp_path / 'a', {'train': 2}))
    root = make_dataset(tmp_path / 'b', {'train': 2, 'val': 0})        # a 0 % validation split
    with pytest.raises(DatasetError, match="'val' contains no images"):
        check_dataset(root)
    with pytest.raises(DatasetError, match='at least 2 class folders'):
        check_dataset(make_dataset(tmp_path / 'c', {'train': 2, 'val': 1}, classes=('ok',)))
    with pytest.raises(DatasetError, match='not found'):
        check_dataset(tmp_path / 'nothing')


def test_every_bad_file_is_listed_with_its_reason(tmp_path):
    root = make_dataset(tmp_path / 'd', {'train': 2, 'val': 1})
    (root / 'train' / 'ok' / 'broken.jpg').write_bytes(b'not an image')
    Image.fromarray(np.zeros((8, 8), np.uint16)).save(root / 'train' / 'nok' / 'deep.png')
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(root / 'val' / 'loose.png')
    rng = np.random.default_rng(0)
    for name in ('cut.jpg', 'cut.bmp'):        # interrupted copies: verify() alone reads no pixel of these formats
        Image.fromarray(rng.integers(0, 256, (200, 300, 3), dtype=np.uint8)).save(root / 'val' / 'ok' / name)
        data = (root / 'val' / 'ok' / name).read_bytes()
        (root / 'val' / 'ok' / name).write_bytes(data[:len(data) // 2])
    with pytest.raises(DatasetError) as e:
        check_dataset(root)
    msg = str(e.value)
    assert '5 error(s)' in msg
    assert 'broken.jpg: unreadable or truncated image' in msg and 'deep.png: not an 8-bit image' in msg
    assert 'cut.jpg: unreadable or truncated image' in msg and 'cut.bmp: unreadable or truncated image' in msg
    assert 'loose.png: image outside any class folder' in msg


def test_unlabeled_empty_class_folders_and_classes_without_training_images(tmp_path):
    """A common export layout: every class folder in every split, often empty, and an unlabeled/ split."""
    root = make_dataset(tmp_path / 'd', {'train': 2, 'val': 1})
    (root / 'unlabeled' / 'ok').mkdir(parents=True)
    (root / 'train' / 'rare').mkdir()
    (root / 'val' / 'rare').mkdir()
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(root / 'val' / 'rare' / 'x.png')
    ds = check_dataset(root)
    assert ds['names'] == {0: 'nok', 1: 'ok', 2: 'rare', 3: 'scratch'}
    whys = [w[3] for w in ds['warnings']]
    assert 'folder ignored (not train, val or test)' in whys
    assert 'class without training images: the model cannot learn it' in whys


def test_identical_images_across_splits_are_reported(tmp_path):
    root = make_dataset(tmp_path / 'd', {'train': 2, 'val': 1})
    shutil.copy(root / 'train' / 'ok' / 'ok_0.png', root / 'val' / 'ok' / 'copy.png')
    ds = check_dataset(root)
    assert any('identical image across train and val' in w[3] for w in ds['warnings'])


def test_the_check_is_cached_until_a_file_changes(tmp_path, monkeypatch):
    import rpclass.data.check as check
    root = make_dataset(tmp_path / 'd', {'train': 2, 'val': 1})
    check_dataset(root)
    calls = []
    real = check._check_one
    monkeypatch.setattr(check, '_check_one', lambda p: calls.append(p) or real(p))
    check_dataset(root)
    assert calls == []                                              # nothing changed: nothing read again
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(root / 'train' / 'ok' / 'new.png')
    check_dataset(root)
    assert len(calls) == 7                                           # the train split, read again


def test_class_map_matches_by_name():
    m = class_map({0: 'ok', 1: 'nok'}, {0: 'nok', 1: 'ok'})
    assert m.tolist() == [1, 0]
    with pytest.raises(DatasetError, match="missing in the dataset: \\['dent'\\]"):
        class_map({0: 'ok', 1: 'nok'}, {0: 'nok', 1: 'ok', 2: 'dent'})


def test_the_augmentation_contract(tmp_path):
    ds = check_dataset(make_dataset(tmp_path / 'd', {'train': 1, 'val': 1}))
    records = ds['records']['train']
    for bad, match in ((lambda s, c: s.image, 'must return the Sample'),
                       (lambda s, c: Sample(s.image.astype(np.float32), s.label, s.path), 'uint8 RGB'),
                       (lambda s, c: Sample(s.image, (s.label + 1) % 3, s.path), 'must not change the class'),
                       (lambda s, c: Sample(s.image[:0], s.label, s.path), 'no pixels left')):
        with pytest.raises(AugmentError, match=match):
            ClassificationDataset(records, [64, 64], augment=bad)[0]
    image, label, meta = ClassificationDataset(records, [64, 96], augment=lambda s, c: s)[0]
    assert tuple(image.shape) == (3, 64, 96) and 0 <= image.min() and image.max() <= 1
    assert label == records[0]['label'] and meta['orig_shape'] == (60, 90)

    def in_place(sample, ctx):                   # an augmentation may edit the image it gets
        sample.image[:10] = 0
        return sample

    image, _, _ = ClassificationDataset(records, [64, 96], augment=in_place)[0]
    assert float(image[:, 3, 48].max()) == 0.0    # the top rows, letterboxed: black


def _orientation(image):
    """(height, width, mark in the top half, mark in the left half) of an image with one bright mark."""
    h, w = image.shape[:2]
    y, x = np.unravel_index(image.astype(int).sum(2).argmax(), (h, w))
    return h, w, bool(y < h / 2), bool(x < w / 2)


def test_the_default_augmentation_shows_the_8_orientations_and_the_upright_one_none():
    """docs/DESIGN.md, decision 8: flips and turns by 90 degrees by default; keep_orientation never moves the image."""
    import rpclass.data.augment as A
    from types import SimpleNamespace
    image = np.zeros((40, 60, 3), np.uint8)
    image[:4, :6] = 255                                      # a mark in the top-left corner of a wide image
    seen, upright = set(), set()
    for k in range(200):
        ctx = SimpleNamespace(epoch=0, epochs=100, imgsz=[32, 32], rng=np.random.default_rng(k), clean=None)
        seen.add(_orientation(A.default_augment(Sample(image.copy(), 0, 'x'), ctx).image))
        ctx.rng = np.random.default_rng(k)
        out = A.upright_augment(Sample(image.copy(), 0, 'x'), ctx).image
        upright.add(_orientation(out))
        assert out.shape == image.shape
    assert len(seen) == 8                                    # 2 shapes x 4 corners: the dihedral group of the square
    assert upright == {_orientation(image)}


def test_the_default_augmentation_is_the_benchmarks_d4():
    """benchmarks/augments.py writes d4 out on its own (its runs must not move with the library); the default is the
    same, draw for draw, so that the measured runs say what the default does."""
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    import rpclass.data.augment as A
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
    import augments

    def ctx(seed):
        return SimpleNamespace(epoch=0, epochs=100, imgsz=[32, 32], rng=np.random.default_rng(seed), clean=None)

    image = np.random.default_rng(1).integers(0, 256, (30, 50, 3), dtype=np.uint8)
    for k in range(60):
        a = A.default_augment(Sample(image.copy(), 0, 'x'), ctx(k))
        b = augments.AUGMENTS[augments.DEFAULT](Sample(image.copy(), 0, 'x'), ctx(k))
        assert np.array_equal(a.image, b.image)
    assert augments.resolve('default') == 'd4'


def test_the_benchmarks_named_augmentations_never_move():
    """hflip and d4 name the runs of docs/EXPERIMENTS.md: their outputs on fixed seeds are pinned (the same with
    Pillow 11 and 13). If this fails, the library's photometric distortion or the draws changed: the old run folders
    no longer mean what a new run with the same name would do."""
    import hashlib
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
    import augments
    image = np.random.default_rng(7).integers(0, 256, (24, 40, 3), dtype=np.uint8)
    pinned = {'hflip': 'ae015ae63fe9364167e43403889319c75657f3d4', 'd4': '33deaf963c80e75e3dbdcb1cfd932438245eb592'}
    for name, digest in pinned.items():
        h = hashlib.sha1()
        for k in range(20):
            ctx = SimpleNamespace(epoch=0, epochs=100, imgsz=[32, 32], rng=np.random.default_rng(k), clean=None)
            out = augments.AUGMENTS[name](Sample(image.copy(), 0, 'x'), ctx).image
            h.update(str(out.shape).encode())
            h.update(out.tobytes())
        assert h.hexdigest() == digest, name


def test_shrink_size_is_an_exact_odd_multiple_of_the_letterbox():
    """rpclass.data.letterbox.shrink_size: exactly factor x the size the full image is letterboxed to, for every image
    that this shrinks to half its pixels or fewer and every square input size; None otherwise, for a non-square input
    size and for an even factor."""
    from rpclass.data.letterbox import letterbox_geometry, shrink_size
    rng = np.random.default_rng(0)
    for _ in range(3000):
        h, w = (int(v) for v in rng.integers(20, 6000, 2))
        side = int(rng.choice([96, 128, 224, 256, 384, 448, 512, 640, 1024]))
        _, rh, rw, _, _ = letterbox_geometry(h, w, side, side)
        size = shrink_size(h, w, side, side, 3)
        if 3 * rh < h and 3 * rw < w and 2 * 9 * rh * rw <= h * w:
            assert size == (3 * rh, 3 * rw)
        else:
            assert size is None
    assert shrink_size(3000, 4000, 448, 448, 5) == (5 * 336, 5 * 448)
    assert shrink_size(3000, 4000, 384, 640, 3) is None
    assert shrink_size(3000, 4000, 448, 448, 2) is None


def _large_photo(path, size=(1600, 1200), seed=0, **save):
    """A smooth 'photo' (noise blown up), saved at `size` (width, height)."""
    rng = np.random.default_rng(seed)
    image = Image.fromarray(rng.integers(0, 256, (12, 16, 3), dtype=np.uint8)).resize(size, Image.BICUBIC)
    image.save(path, **save)
    return image


def _grating(h, w, cycles, side):
    """A vertical grating at `cycles` cycles per pixel of the image's letterbox at `side`: fine detail, which a softer
    resize loses first."""
    x = np.arange(w)
    row = 127.5 + 100 * np.sin(2 * np.pi * cycles * side / max(h, w) * x)
    return np.repeat(np.repeat(np.round(row)[None, :, None], h, 0), 3, 2).astype(np.uint8)


def _turn(sample, ctx):
    return Sample(np.ascontiguousarray(np.rot90(sample.image)), sample.label, sample.path)


def _same(sample, ctx):
    return sample


def test_a_shrunk_read_gives_the_full_resolution_tensor(tmp_path):
    """A large training image read at 3x its letterbox and letterboxed with the full image's geometry gives the
    full-resolution tensor, as read and turned by 90 degrees: fine detail keeps its contrast (1 % at most, against up
    to 24 % lost at 2x) and the rest differs by the rounding of the intermediate image to 8 bits."""
    from rpclass.data import any_resolution
    from rpclass.data.letterbox import letterbox
    from rpclass.data.sources import read_image
    side = 160
    for cycles in (0.2, 0.45):
        path = tmp_path / f'grating{cycles}.png'
        Image.fromarray(_grating(1800, 2400, cycles, side)).save(path)
        records = [{'path': str(path), 'label': 0}]
        for turn in (_same, _turn):
            shrunk = ClassificationDataset(records, [side, side], any_resolution(lambda s, c, t=turn: t(s, c)))
            full = ClassificationDataset(records, [side, side], lambda s, c, t=turn: t(s, c))
            a, b = shrunk[0], full[0]
            assert shrunk.shrink_factor == 3 and full.shrink_factor is None
            assert a[2]['geom'] == b[2]['geom'] and a[2]['orig_shape'] == b[2]['orig_shape']
            _, rh, rw, top, left = b[2]['geom']
            area = (slice(None), slice(top, top + rh), slice(left, left + rw))
            assert abs(float(a[0][area].std()) / float(b[0][area].std()) - 1) < 0.01, (cycles, turn.__name__)
            assert float((a[0] - b[0]).abs().mean()) * 255 < 0.5
    full = letterbox(read_image(path), side, side)[0]
    assert torch.equal(ClassificationDataset(records, [side, side], _same)[0][0], full)


def test_read_image_reads_a_large_image_at_the_size_asked(tmp_path, monkeypatch):
    """read_image(path, fit): exactly the size fit asks for, for JPEG (decoded at a reduced DCT scale), PNG, grey and
    palette images, close to the full image resized; fit returning None, or no fit, gives the full image. A JPEG whose
    sides are not multiples of the DCT scale keeps its content in place: a thin line near its far edge lands where the
    full image's resize puts it."""
    from PIL import JpegImagePlugin
    from rpclass.data.sources import read_image
    base = _large_photo(tmp_path / 'a.jpg', quality=95)
    base.save(tmp_path / 'a.png')
    base.convert('L').save(tmp_path / 'g.jpg', quality=95)
    base.convert('P').save(tmp_path / 'p.png')
    drafts = []
    real_draft = JpegImagePlugin.JpegImageFile.draft
    monkeypatch.setattr(JpegImagePlugin.JpegImageFile, 'draft',
                        lambda self, mode, size: drafts.append(size) or real_draft(self, mode, size))
    for name in ('a.jpg', 'a.png', 'g.jpg', 'p.png'):
        path = tmp_path / name
        full = read_image(path)
        assert full.shape == (1200, 1600, 3)
        small = read_image(path, lambda h, w: (300, 400))
        assert small.shape == (300, 400, 3) and small.dtype == np.uint8 and small.flags.writeable
        reference = np.asarray(Image.fromarray(full).resize((400, 300), Image.BILINEAR)).astype(int)
        assert np.abs(small.astype(int) - reference).mean() < 3, name
        assert read_image(path, lambda h, w: None).shape == (1200, 1600, 3)
    assert drafts == [(400, 300), (400, 300)]                  # the two JPEGs, decoded at a quarter
    line = np.full((1205, 1603, 3), 40, np.uint8)
    line[:, 1580:1584] = 255
    Image.fromarray(line).save(tmp_path / 'line.jpg', quality=98)
    small = read_image(tmp_path / 'line.jpg', lambda h, w: (301, 400)).astype(float)[..., 0]
    reference = np.asarray(Image.fromarray(read_image(tmp_path / 'line.jpg')).resize((400, 301), Image.BILINEAR))
    columns = np.arange(400)

    def centre(image):
        weight = np.clip(image.astype(float)[..., 0].mean(0) if image.ndim == 3 else image.mean(0), 60, None) - 60
        return float((weight * columns).sum() / weight.sum())
    assert abs(centre(small) - centre(reference)) < 0.1


def test_large_training_images_reach_the_augmentation_shrunk(tmp_path, monkeypatch):
    """Large training images reach a declared augmentation (the library's own, @any_resolution) at 3x their letterbox.
    Small images, undeclared functions (functools.wraps around a declared one included), splits without augmentation
    (val, test) and non-square input sizes are read in full; the factor is fixed when the dataset is built, and
    SHRINK_FACTOR = None turns it off."""
    import functools
    import rpclass.data.dataset as D
    from rpclass.data import any_resolution, default_augment, upright_augment
    assert D._declared(default_augment) and D._declared(upright_augment)
    (tmp_path / 'a').mkdir()
    _large_photo(tmp_path / 'a' / 'big.jpg', quality=95).resize((120, 90)).save(tmp_path / 'a' / 'small.png')
    records = [{'path': str(tmp_path / 'a' / n), 'label': 0} for n in ('big.jpg', 'small.png')]
    seen = {}

    def look(sample, ctx):
        seen[sample.path] = sample.image.shape[:2]
        return sample

    declared = any_resolution(lambda s, c: look(s, c))
    shrunk = ClassificationDataset(records, [128, 128], declared)
    shrunk[0]
    assert seen[records[0]['path']] == (288, 384)                   # 3 x the 96 x 128 its letterbox resizes it to
    shrunk[1]
    assert seen[records[1]['path']] == (90, 120)                    # small: as it is
    for other in (ClassificationDataset(records, [128, 128], look),
                  ClassificationDataset(records, [128, 128], functools.wraps(declared)(lambda s, c: look(s, c))),
                  ClassificationDataset(records, [96, 128], declared)):
        other[0]
        assert seen[records[0]['path']] == (1200, 1600) and other.shrink_factor is None
    val = ClassificationDataset(records, [128, 128])
    assert val[0][2]['orig_shape'] == (1200, 1600) and val.read(records[0]).image.shape[:2] == (1200, 1600)
    monkeypatch.setattr(D, 'SHRINK_FACTOR', None)
    shrunk[0]
    assert seen[records[0]['path']] == (288, 384)                   # fixed when it was built
    ClassificationDataset(records, [128, 128], declared)[0]
    assert seen[records[0]['path']] == (1200, 1600)


def test_the_benchmarks_augmentations_read_full_images(monkeypatch):
    """benchmarks/augments.py's functions, upright included, read every training image in full (their runs never move
    with the library); --shrink-factor wraps one to read large images shrunk, under its own --tag and with its own
    name in args.yaml."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
    import augments
    import recipe
    import rpclass.data.dataset as D
    for name, fn in augments.AUGMENTS.items():
        if fn is not None and name != 'default':
            assert not D._declared(fn), name
    monkeypatch.setattr(D, 'SHRINK_FACTOR', None)
    for factor, tag in ((3, ''), (2, 'shrink2'), (1, 'shrink1')):
        with pytest.raises(SystemExit):
            recipe.shrunk(augments.d4, factor, tag)
    with pytest.raises(SystemExit):
        recipe.shrunk(None, 3, 'shrink3')
    wrapped = recipe.shrunk(augments.d4, 5, 'shrink5')
    assert D._declared(wrapped) and D.SHRINK_FACTOR == 5 and wrapped.fn is augments.d4
    assert wrapped.__name__ == 'd4+shrink5'


def test_the_runners_pass_shrink_factor_to_training(tmp_path, monkeypatch):
    """benchmarks/run.py --shrink-factor 3 --tag shrink3: the run trains with the wrapped augmentation, in its own
    folder."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
    import rpclass.data.dataset as D
    import run
    seen = {}

    class Stop(Exception):
        pass

    class Fake:
        def __init__(self, name):
            pass

        def train(self, data, **kw):
            seen.update(kw)
            raise Stop

    monkeypatch.setattr(run, 'RPCLASS', Fake)
    monkeypatch.setattr(D, 'SHRINK_FACTOR', D.SHRINK_FACTOR)
    monkeypatch.setattr(sys, 'argv', ['run.py', str(tmp_path / 'folds'), str(tmp_path / 'out'), '--datasets', 'x',
                                      '--models', 'hgnetv2-b2', '--folds', '0', '--augment', 'd4',
                                      '--shrink-factor', '3', '--tag', 'shrink3'])
    with pytest.raises(Stop):
        run.main()
    assert seen['augment'].__name__ == 'd4+shrink3' and D._declared(seen['augment'])
    assert seen['project'].endswith('hgnetv2-b2+d4@shrink3/s0')


def test_a_cached_split_gives_the_same_tensors(tmp_path):
    from rpclass.data import cached
    ds = check_dataset(make_dataset(tmp_path / 'd', {'train': 1, 'val': 3}))
    val = ClassificationDataset(ds['records']['val'], [64, 96])
    cache = cached(val, workers=0)
    assert cache is not val and len(cache) == len(val) and cache.records is val.records
    for i in range(len(val)):
        x, y, m = val[i]
        cx, cy, cm = cache[i]
        assert torch.equal(x, cx) and y == cy and m['path'] == cm['path']
    import rpclass.data.dataset as D
    D.CACHE_BYTES, old = 10, D.CACHE_BYTES
    try:
        assert cached(val) is val                                       # too large for the budget: read every time
    finally:
        D.CACHE_BYTES = old


@pytest.mark.parametrize('n, micro, chunk, workers', [(10, 4, 1, 0), (10, 4, 3, 2), (11, 4, 4, 0), (3, 4, 2, 0),
                                                       (8, 4, None, 3)])
def test_micro_batches_are_those_of_whole_batches(tmp_path, monkeypatch, n, micro, chunk, workers):
    """rpclass/data/loader.py: micro-batches joined from chunks prepared by several workers are, image for image, the
    batches of a DataLoader of whole batches with the same sampler, two epochs in a row (the last incomplete one
    dropped when there is a whole one), augmented images included."""
    import rpclass.data.loader as L
    from torch.utils.data import RandomSampler
    from rpclass.data import default_augment
    monkeypatch.setattr(L, 'CHUNK', chunk)
    ds = check_dataset(make_dataset(tmp_path / 'd', {'train': n, 'val': 1}))
    records = ds['records']['train'][:n]
    dataset = ClassificationDataset(records, [40, 56], default_augment, epochs=10, seed=3)
    reference = ClassificationDataset(records, [40, 56], default_augment, epochs=10, seed=3)
    order, ref_order = torch.Generator(), torch.Generator()
    batches = L.MicroBatches(dataset, micro, order, workers)
    plain = DataLoader(reference, batch_size=micro, sampler=RandomSampler(reference, generator=ref_order),
                       collate_fn=collate, drop_last=n >= micro)
    assert len(batches) == len(plain)
    for epoch in (0, 1):
        dataset.set_epoch(epoch)
        reference.set_epoch(epoch)
        order.manual_seed(epoch)
        ref_order.manual_seed(epoch)
        got, want = list(batches), list(plain)
        assert len(got) == len(want) == len(batches)
        for (x, y, m), (rx, ry, rm) in zip(got, want):
            assert torch.equal(x, rx) and torch.equal(y, ry)
            assert [a['path'] for a in m] == [b['path'] for b in rm]


def test_dataloader_workers_see_the_epoch(tmp_path):
    ds = check_dataset(make_dataset(tmp_path / 'd', {'train': 1, 'val': 1}))

    def stamp(sample, ctx):                     # the epoch, written into the image
        return Sample(np.full_like(sample.image, ctx.epoch), sample.label, sample.path)

    dataset = ClassificationDataset(ds['records']['train'], [32, 32], augment=stamp)
    loader = DataLoader(dataset, batch_size=3, num_workers=1, persistent_workers=True, collate_fn=collate)
    seen = []
    for epoch in (0, 3):
        dataset.set_epoch(epoch)
        images, _, _ = next(iter(loader))
        seen.append(round(float(images[0, 0, 16, 16]) * 255))
    assert seen == [0, 3]


def test_file_names_that_are_not_utf8_are_fine(tmp_path, capsys):
    """Archives made on Windows unpack with names in another encoding: the check, its cache and its report cope."""
    import os
    root = make_dataset(tmp_path / 'd', {'train': 2, 'val': 1})
    odd = os.fsdecode(b'pe\xe7a.png')
    shutil.copy(root / 'train' / 'ok' / 'ok_0.png', root / 'train' / 'ok' / odd)    # also a duplicate: a warning
    ds = check_dataset(root)
    assert ds['stats']['train']['images'] == 7
    from rpclass.data.check import format_problems
    print(format_problems(ds['warnings'], 'warnings'))
    assert 'pe\\udce7a.png' in capsys.readouterr().out


def test_an_empty_test_folder_is_no_test_split(tmp_path):
    """Exported datasets often have test/ with its class folders even when no image goes to test."""
    ds = check_dataset(make_dataset(tmp_path / 'd', {'train': 2, 'val': 1, 'test': 0}))
    assert any('no test images' in w[3] for w in ds['warnings']) and not ds['records']['test']
