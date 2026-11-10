# RP-CLASS - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""RP-CLASS's training augmentation, applied to training images before the letterbox. RPCLASS.train uses
default_augment, or upright_augment with keep_orientation=True; augment=None trains without augmentation, and any
other function(sample, ctx) replaces them (contract in rpclass.data.dataset).

  default_augment     photometric distortion 0.5, then the 8 orientations of the square: a horizontal flip 0.5, a
                      vertical flip 0.5 and a turn by a random multiple of 90 degrees
  upright_augment     photometric distortion 0.5 only: the image is never flipped or turned
  last NO_AUG_FRACTION (4 %) of the epochs, at least one: clean images, for both

Why the 8 orientations (docs/EXPERIMENTS.md, 2026-11-06; docs/DESIGN.md, decision 8): over the provisional default
(photometric and the horizontal flip) they gained 1.3 to 3.6 points of macro-F1 on two private OK / NOK inspection sets
(parts seen from above) for HGNetv2-B2 and the DINOv3 ViT, 1.1 to 1.8 on histology (Kather-2016, PCam), and changed
nothing on natural photos, half of them upright (flowers: +0.3, -0.1); a zoom-out added nothing and a free rotation lost
on square crops. Their one risk is a task whose class is the orientation (a part mounted upside down, a mirrored label):
the model would learn to ignore it. That is what keep_orientation=True is for; it removes the horizontal flip too.

Never anything that erases or invents content (no random erasing, mixup or cutmix): in inspection that removes the
defect or creates one. The phases are fractions of ctx.epochs, the epoch where training ends (the trainer moves it
when patience brings the end forward); with a time budget the trainer tells every epoch whether it is clean
(ctx.clean). The values are module attributes, for every run: `import rpclass.data.augment as A; A.P_PHOTOMETRIC = 0.0`.
One flip and the four turns already give all 8 orientations, so only P_FLIP = P_VFLIP = 0 together stop mirroring;
TURNS = False removes the quarter turns (the half turn stays, through the two flips). No geometry at all is
keep_orientation=True.
"""
import numpy as np
from PIL import Image, ImageEnhance

from .dataset import Sample, any_resolution

P_PHOTOMETRIC, P_FLIP, P_VFLIP, TURNS = 0.5, 0.5, 0.5, True
NO_AUG_FRACTION = 0.04                               # = the trainer's clean_fraction


@any_resolution
def default_augment(sample, ctx):
    """function(sample, ctx) -> sample for RPCLASS.train: photometric distortion and the 8 orientations."""
    if _clean(ctx):
        return sample
    rng = ctx.rng
    sample = _photometric(sample, rng)
    image = sample.image
    if rng.random() < P_FLIP:
        image = image[:, ::-1]
    if rng.random() < P_VFLIP:
        image = image[::-1]
    if TURNS:
        image = np.rot90(image, int(rng.integers(4)))
    return Sample(np.ascontiguousarray(image), sample.label, sample.path)


@any_resolution
def upright_augment(sample, ctx):
    """function(sample, ctx) -> sample for RPCLASS.train(keep_orientation=True): photometric distortion only."""
    if _clean(ctx):
        return sample
    return _photometric(sample, ctx.rng)


def _clean(ctx):
    clean = getattr(ctx, 'clean', None)
    return ctx.epoch >= ctx.epochs - max(1, round(NO_AUG_FRACTION * ctx.epochs)) if clean is None else clean


def _photometric(sample, rng):
    if rng.random() < P_PHOTOMETRIC:
        return Sample(photometric(sample.image, rng), sample.label, sample.path)
    return sample


def photometric(image, rng):
    """Brightness, contrast, saturation and hue, each with probability 0.5 (torchvision RandomPhotometricDistort)."""
    im = Image.fromarray(image)
    contrast_first = rng.random() < 0.5
    if rng.random() < 0.5:
        im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.875, 1.125))
    if contrast_first and rng.random() < 0.5:
        im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.5, 1.5))
    if rng.random() < 0.5:
        im = ImageEnhance.Color(im).enhance(rng.uniform(0.5, 1.5))
    if rng.random() < 0.5:
        shift = int(round(rng.uniform(-0.05, 0.05) * 255))
        h, s, v = im.convert('HSV').split()
        im = Image.merge('HSV', (h.point(lambda x: (x + shift) % 256), s, v)).convert('RGB')
    if not contrast_first and rng.random() < 0.5:
        im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.5, 1.5))
    return np.asarray(im)
