# RP-CLASS - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""Turn whatever `predict` receives into RGB uint8 arrays, without ever modifying the caller's images.

Accepted: a numpy array (H x W, or H x W x 1/3/4) with a color tag, a file path, a folder, or a list of these.
Only 8-bit images are accepted, as at serving time.
"""
from pathlib import Path

import numpy as np
from PIL import Image

COLORS = ('RGB', 'BGR', 'GRAY')
_BILINEAR = getattr(Image, 'Resampling', Image).BILINEAR       # Image.Resampling from Pillow 9.1
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def to_rgb(image, color='RGB'):
    """uint8 H x W x 3 RGB view or copy of `image`. Read-only and non-contiguous arrays are fine."""
    color = color.upper()
    if color not in COLORS:
        raise ValueError(f"color must be one of {', '.join(COLORS)}, got {color!r}")
    if image.dtype != np.uint8:
        raise ValueError(f'only 8-bit images are supported, got dtype {image.dtype}')
    if image.ndim == 2:
        image = image[..., None]
    if image.ndim != 3 or image.shape[2] not in (1, 3, 4):
        raise ValueError(f'expected an image H x W, H x W x 1, 3 or 4, got shape {image.shape}')
    c = image.shape[2]
    if c == 1:
        return np.repeat(image, 3, axis=2)
    if color == 'GRAY':
        raise ValueError(f'color="GRAY" but the image has {c} channels')
    return image[..., 2::-1] if color == 'BGR' else image[..., :3]


def read_image(path, fit=None):
    """uint8 RGB array of an image file, a writeable copy (augmentations may edit it in place).

    fit(height, width) -> (height, width) or None: a smaller size to read the image at, decided from the size in its
    header. A JPEG is then decoded at a reduced DCT scale, never below that size, and an antialiased bilinear resize
    maps exactly the image's area onto the size asked: a large photo is never decoded in full."""
    with Image.open(path) as im:
        if im.mode in ('I', 'I;16', 'I;16B', 'I;16L', 'F'):
            raise ValueError(f'{path}: only 8-bit images are supported (mode {im.mode})')
        size = fit(im.height, im.width) if fit is not None else None
        box = None
        if size is not None and im.format in ('JPEG', 'MPO'):
            drafted = im.draft(None, (size[1], size[0]))
            box = drafted[1] if drafted else None     # the image's area in the reduced decode (its last pixels partial)
        if im.mode not in ('RGB', 'L'):
            im = im.convert('RGB')
        if size is not None and (im.size != (size[1], size[0]) or box is not None):
            im = im.resize((size[1], size[0]), _BILINEAR, box=box)
        return to_rgb(np.array(im))


def load_images(source, color='RGB'):
    """Returns a list of (rgb_array, path_or_None)."""
    if isinstance(source, np.ndarray):
        return [(to_rgb(source, color), None)]
    if isinstance(source, (str, Path)):
        p = Path(source)
        if p.is_dir():
            files = sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)
            if not files:
                raise FileNotFoundError(f'no images in {p}')
            return [(read_image(f), str(f)) for f in files]
        if not p.exists():
            raise FileNotFoundError(p)
        return [(read_image(p), str(p))]
    if isinstance(source, (list, tuple)):
        return [item for s in source for item in load_images(s, color)]
    raise TypeError(f'unsupported source type {type(source).__name__}')
