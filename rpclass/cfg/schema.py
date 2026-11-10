# RP-CLASS - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""Model sizes and everything derived from them.

RP-CLASS has a few sizes and nothing else to choose: classes come from the dataset and the input size is chosen when
training (each backbone proposes one: imgsz in BACKBONES). A size fixes the backbone, a timm network with pretrained
weights (ImageNet for HGNetv2, DINOv3 LVD-1689M for lt) whose classification head is replaced by one for the
dataset's classes.

Two ranges (docs/DESIGN.md, decision 2): the CNN sizes s / m / l, RP-OD's HGNetv2 B0 / B2 / B4, fast and safe in
float16 anywhere; and lt, a transformer (DINOv3 ViT-S/16), the most accurate in the first benchmark, 1.8x slower than
m at 384 px (its default 448 px does 1.36x the work of 384), its attention beyond float16 inside (TensorRT handles
it; a plain float16 runtime does not). Which backbone sits in each size can still change with the recipe experiments and
the public benchmark. Other candidates are registered in BACKBONES, so that a benchmark can build any of them:
`SIZES['x'] = Architecture('convnext-t-dinov3')` before `RPCLASS('x')`.
"""
from dataclasses import asdict, dataclass


class ConfigError(ValueError):
    pass


# ------------------------------------------------------------------------------------------------ fixed choices
# Backbones: the timm model with its pretrained tag, pinned. fixed_size: the network is built for one input size
# (vision transformers: their position encoding); CNNs take any size. imgsz: the input size a new model of it trains at
# unless told otherwise (docs/EXPERIMENTS.md, 2026-11-06: HGNetv2-B2 peaks at 384 px, the ViT still gains at 448).
BACKBONES = {
    'hgnetv2-b0': dict(timm='hgnetv2_b0.ssld_stage1_in22k_in1k', imgsz=384),
    'hgnetv2-b2': dict(timm='hgnetv2_b2.ssld_stage1_in22k_in1k', imgsz=384),
    'hgnetv2-b4': dict(timm='hgnetv2_b4.ssld_stage1_in22k_in1k', imgsz=384),
    'convnext-t-dinov3': dict(timm='convnext_tiny.dinov3_lvd1689m', imgsz=384),
    'vit-s16-dinov3': dict(timm='vit_small_patch16_dinov3.lvd1689m', fixed_size=True, imgsz=448),
}


@dataclass(frozen=True)
class Architecture:
    """What a size fixes. Users choose a size; checkpoints store the architecture, so any checkpoint loads."""
    backbone: str


SIZES = {
    's': Architecture('hgnetv2-b0'),       # CNN range
    'm': Architecture('hgnetv2-b2'),
    'l': Architecture('hgnetv2-b4'),
    'lt': Architecture('vit-s16-dinov3'),  # transformer range
}
DEFAULT_SIZE = 'm'
STRIDE = 32                 # input sides are rounded up to multiples of this (the backbones' coarsest stride)


def model_config(size=DEFAULT_SIZE):
    if size not in SIZES:
        raise ConfigError(f"unknown model size {size!r}; RP-CLASS has {', '.join(map(repr, SIZES))}")
    return SIZES[size]


def default_imgsz(arch):
    """The input size a new model of this architecture trains at by default (BACKBONES), as [height, width]."""
    side = BACKBONES[arch.backbone]['imgsz']
    return [side, side]


def size_of(arch):
    """The name of the size with this architecture, or None."""
    return next((name for name, a in SIZES.items() if a == arch), None)


def describe(arch):
    name = size_of(arch)
    return f'size {name} ({arch.backbone})' if name else f'not a standard size ({arch.backbone})'


def to_dict(arch):
    return {'size': size_of(arch), **asdict(arch)}


def from_dict(data):
    """Checkpoint config -> Architecture."""
    arch = Architecture(data['backbone'])
    if arch.backbone not in BACKBONES:
        raise ConfigError(f'backbone {arch.backbone!r} is not part of this version of RP-CLASS')
    return arch


# ------------------------------------------------------------------------------------------------ derived
def derive(arch, num_classes, imgsz):
    """Everything the network needs, computed from the architecture. Nothing here is user-facing."""
    b = BACKBONES[arch.backbone]
    return dict(timm_name=b['timm'], num_classes=int(num_classes), imgsz=[int(imgsz[0]), int(imgsz[1])],
                fixed_size=bool(b.get('fixed_size', False)))


def input_size(imgsz):
    """[height, width], each rounded up to a multiple of STRIDE (the image is letterboxed into it, never deformed)."""
    if isinstance(imgsz, int):
        imgsz = [imgsz, imgsz]
    if len(imgsz) != 2 or min(imgsz) < 1:
        raise ValueError(f'imgsz must be an int or [height, width], got {imgsz!r}')
    return [-(-int(s) // STRIDE) * STRIDE for s in imgsz]
