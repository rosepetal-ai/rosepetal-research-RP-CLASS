# RP-CLASS - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""Augmentations of the recipe experiments (docs/DESIGN.md, decision 8), selected by name in benchmarks/run.py and
benchmarks/public/benchmark.py (--augment). None erases or invents content: geometry only moves the whole image (a
rotation enlarges the canvas, never crops), and the padding is the letterbox grey.

    default   resolved to the named augmentation the library's default equals, draw for draw (DEFAULT: d4 since the
              8 orientations became the default; tests/test_dataset.py checks it), so a run folder never holds two
    upright   the library's augmentation for keep_orientation=True: photometric distortion only (called through a
              function of this module, so that it reads full images like the others here)
    hflip     the provisional default until 2026-11-06: photometric distortion 0.5, horizontal flip 0.5
    none      no augmentation
    d4        hflip, then the rest of the 8 orientations of the square (vertical flip 0.5, then a random multiple of
              90 degrees)
    rotate    hflip, then a free rotation, 0-360 degrees, on a canvas that holds the whole rotated image
    d4-zoom   d4, then a zoom-out 0.5: the image on a grey canvas up to 1.3x its size, at a random spot

All of them leave the last 4 % of the epochs clean, as the library's do. Their geometry and the order of their random
draws (photometric, horizontal flip, vertical flip, turn, the library default's order too) are written out here; the
photometric distortion and the clean stretch are the library's, and a test pins the outputs of hflip and d4 on fixed
seeds. Run folders: <model> for hflip (the provisional default, which every run without a suffix used), and
<model>+<name> for every other augmentation, 'default' resolved first.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rpclass.data import Sample, default_augment, upright_augment  # noqa: E402
from rpclass.data.augment import NO_AUG_FRACTION, photometric  # noqa: E402

FILL = 114
ZOOM_MAX, P_ZOOM = 1.3, 0.5


def _clean(ctx):
    clean = getattr(ctx, 'clean', None)
    return ctx.epoch >= ctx.epochs - max(1, round(NO_AUG_FRACTION * ctx.epochs)) if clean is None else clean


def _with(sample, image):
    return Sample(np.ascontiguousarray(image), sample.label, sample.path)


def hflip(sample, ctx):
    if _clean(ctx):
        return sample
    rng = ctx.rng
    if rng.random() < 0.5:
        sample = Sample(photometric(sample.image, rng), sample.label, sample.path)
    if rng.random() < 0.5:
        sample = _with(sample, sample.image[:, ::-1])
    return sample


def d4(sample, ctx):
    if _clean(ctx):
        return sample
    sample = hflip(sample, ctx)
    image = sample.image[::-1] if ctx.rng.random() < 0.5 else sample.image
    return _with(sample, np.rot90(image, int(ctx.rng.integers(4))))


def rotate(sample, ctx):
    if _clean(ctx):
        return sample
    sample = hflip(sample, ctx)
    angle = float(ctx.rng.uniform(0, 360))
    rotated = Image.fromarray(sample.image).rotate(angle, resample=Image.BILINEAR, expand=True,
                                                     fillcolor=(FILL, FILL, FILL))
    return _with(sample, np.asarray(rotated))


def d4_zoom(sample, ctx):
    if _clean(ctx):
        return sample
    sample = d4(sample, ctx)
    if ctx.rng.random() >= P_ZOOM:
        return sample
    h, w = sample.image.shape[:2]
    r = ctx.rng.uniform(1.0, ZOOM_MAX)
    big_h, big_w = int(h * r), int(w * r)
    top, left = int(ctx.rng.uniform(0, big_h - h)), int(ctx.rng.uniform(0, big_w - w))
    canvas = np.full((big_h, big_w, 3), FILL, np.uint8)
    canvas[top:top + h, left:left + w] = sample.image
    return _with(sample, canvas)


def upright(sample, ctx):
    """upright_augment, draw for draw; not declared any_resolution: it reads every image in full, as the others here."""
    return upright_augment(sample, ctx)


AUGMENTS = {'default': default_augment, 'upright': upright, 'hflip': hflip, 'none': None, 'd4': d4,
            'rotate': rotate, 'd4-zoom': d4_zoom}
DEFAULT = 'd4'


def resolve(name):
    """The concrete name of an augmentation: 'default' becomes DEFAULT."""
    return DEFAULT if name == 'default' else name
