# RP-CLASS - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""Letterbox, as in RP-OD, with the arithmetic of a CUDA letterbox kernel with antialias, so that a GPU pre-processing
at serving time gives the same pixels.

Scale = the largest that fits the whole image (upscaling allowed), computed in float32; resized size rounded down;
image centered (odd pixel to the bottom/right); padding 114; bilinear resampling with half-pixel centers and
antialias, in float32 without rounding back to uint8; then x / 255. The model applies its own normalization inside.
"""
import warnings

import numpy as np
import torch
import torch.nn.functional as F

PAD_VALUE = 114
_F = np.float32


def letterbox_geometry(in_h, in_w, out_h, out_w):
    """(scale, resized_h, resized_w, pad_top, pad_left), with the kernel's float32 arithmetic."""
    scale = min(_F(out_h) / _F(in_h), _F(out_w) / _F(in_w))
    rh = max(1, int(np.floor(_F(in_h) * scale)))
    rw = max(1, int(np.floor(_F(in_w) * scale)))
    return float(scale), rh, rw, (out_h - rh) // 2, (out_w - rw) // 2


def shrink_size(in_h, in_w, out_h, out_w, factor):
    """The size to read an in_h x in_w training image at: exactly `factor` (an odd integer, 3 or more) times the size
    the letterbox resizes the full image to. None when that does not at least halve the image's pixels (the resize
    would cost about what it saves: VisA's 2.3 MP objects at 448 px, 1.44 s an epoch in full, 1.51 s at 3x), or when
    the input size is not square (a turned image would then need another size).

    Read at that size by an antialiased bilinear resize, then letterboxed with the full image's geometry (an exact odd
    factor, so every output pixel is centred on an intermediate one), the image gets the single resize from
    the full image: two triangle filters whose corners fall on the same samples make the wider triangle. Only the
    rounding of the intermediate image to 8 bits differs. An even factor puts the second filter's samples between
    pixels: at 2, fine detail lost up to a quarter of its contrast (docs/EXPERIMENTS.md, 2026-11-09)."""
    if out_h != out_w or factor < 3 or factor % 2 != 1:
        return None
    _, rh, rw, _, _ = letterbox_geometry(in_h, in_w, out_h, out_w)
    h, w = factor * rh, factor * rw
    return (h, w) if h < in_h and w < in_w and 2 * h * w <= in_h * in_w else None


def letterbox(rgb, out_h, out_w, device='cpu', antialias=True, geometry=None):
    """rgb: uint8 H x W x 3 array. Returns (tensor [3, out_h, out_w] in [0, 1], geometry). geometry: letterbox_geometry
    of another size to use instead of rgb's: the full image's, for a training image read shrunk (shrink_size)."""
    h, w = rgb.shape[:2]
    geom = letterbox_geometry(h, w, out_h, out_w) if geometry is None else geometry
    _, rh, rw, top, left = geom
    x = _as_tensor(np.ascontiguousarray(rgb)).to(device).permute(2, 0, 1)[None].float()   # one copy at most
    x = F.interpolate(x, size=(rh, rw), mode='bilinear', align_corners=False, antialias=antialias)
    out = torch.full((1, 3, out_h, out_w), float(PAD_VALUE), device=device)
    out[..., top:top + rh, left:left + rw] = x
    return out[0] * float(_F(1.0) / _F(255.0)), geom


def _as_tensor(array):
    """Shares memory with the array. Read-only arrays are fine: nothing writes to it, the next step copies."""
    if array.flags.writeable:
        return torch.from_numpy(array)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        return torch.from_numpy(array)
