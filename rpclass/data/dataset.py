# RP-CLASS - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""Dataset for training and validation: read -> augment (training images) -> letterbox, exactly as at serving time.

The augmentation is one function: rpclass.data.augment.default_augment, or upright_augment with keep_orientation=True,
unless RPCLASS.train gets augment=None (none) or another function, for example from an augmentation library:

    def augment(sample, ctx):
        # sample.image   uint8 RGB [H, W, 3] at original resolution (a large one shrunk if the function is
        #                declared with @any_resolution: see below)
        # sample.label   int, the class id (an augmentation does not change it)
        # sample.path    where it comes from
        # ctx.epoch      the current epoch, from 0
        # ctx.epochs     the epoch where training ends: the epochs asked for, or earlier once patience brings the
        #                end forward
        # ctx.clean      with a time budget, whether this epoch is in the clean last stretch (None without one: the
        #                stretch is a fraction of ctx.epochs)
        # ctx.imgsz      [height, width] the image will be letterboxed into
        # ctx.rng        numpy random generator for this sample
        return sample   # any size: the letterbox comes after

Large training images are read shrunk when the augmentation works at any resolution: the library's own do, and
@any_resolution declares your own (only for a function that uses no size in pixels: no crop, blur radius or shift).
A photo of 9 MP cost 380 ms of CPU per epoch to read, augment and letterbox at full size, the GPU waiting
meanwhile, and the letterbox then threw those pixels away.

An image larger than SHRINK_FACTOR (3) times the size the letterbox resizes it to, by at least twice the pixels, is
read at exactly that multiple of it (a JPEG decoded at a reduced DCT scale), then augmented, then letterboxed with
the full image's geometry, as read or turned by 90 degrees. With an odd factor, the two resizes make the single
resize from the full image (rpclass.data.letterbox.shrink_size): the tensor is the full-resolution one, up to the
rounding of the intermediate image to 8 bits. An augmentation that changes the size (a zoom, a free rotation)
letterboxes it by its own size instead. Smaller images, every val and test image, and every image with a non-square
input size are read as they are. SHRINK_FACTOR = None reads every image in full.
"""
import multiprocessing as mp
import time
import weakref
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from .check import DatasetError
from .letterbox import letterbox, letterbox_geometry, shrink_size
from .sources import read_image

SHRINK_FACTOR = 3
_ANY_RESOLUTION = weakref.WeakSet()


def any_resolution(fn):
    """Declares that an augmentation uses no size in pixels, so that large training images reach it already shrunk
    (module docstring). The library's own are declared. A wrapper made with functools.wraps is another function: it is
    not, unless declared too."""
    _ANY_RESOLUTION.add(fn)
    return fn


def _declared(fn):
    try:
        return fn in _ANY_RESOLUTION
    except TypeError:                    # unhashable callables
        return False


@dataclass
class Sample:
    image: np.ndarray
    label: int
    path: str = None


class AugmentError(RuntimeError):
    pass


class AugmentContext:
    def __init__(self, epoch, epochs, imgsz, rng, clean=None):
        """clean: the phase of this epoch when a time budget decides it (None: follow `epochs`)."""
        self.epoch, self.epochs, self.imgsz, self.rng, self.clean = epoch, epochs, list(imgsz), rng, clean


def class_map(dataset_names, model_names):
    """Array mapping dataset class ids to model class ids, by name. The sets must be identical; the order may differ."""
    ds, md = set(dataset_names.values()), set(model_names.values())
    if ds != md:
        missing, extra = sorted(md - ds), sorted(ds - md)
        raise DatasetError('the dataset classes do not match the model classes'
                           + (f'; missing in the dataset: {missing}' if missing else '')
                           + (f'; not in the model: {extra}' if extra else ''))
    by_name = {n: i for i, n in model_names.items()}
    return np.array([by_name[dataset_names[i]] for i in range(len(dataset_names))], np.int64)


class ClassificationDataset(Dataset):
    def __init__(self, records, imgsz, augment=None, epochs=1, seed=0, id_map=None):
        """records: one split of check_dataset. id_map: dataset class id -> model class id (another dataset)."""
        self.records, self.imgsz, self.augment, self.seed, self.id_map = records, list(imgsz), augment, seed, id_map
        # fixed here, so that workers started by spawn or forkserver (no copy of this process) read the same way
        square = self.imgsz[0] == self.imgsz[1]
        self.shrink_factor = SHRINK_FACTOR if augment is not None and _declared(augment) and square else None
        self._epoch = mp.Value('i', 0)           # shared with the dataloader workers
        self._epochs = mp.Value('i', epochs)     # the end of training, which patience can bring forward
        self._clean = mp.Value('i', -1)          # time budget: 1/0 this epoch is/is not clean; -1 no budget

    def __len__(self):
        return len(self.records)

    def set_epoch(self, epoch):
        self._epoch.value = int(epoch)

    @property
    def epochs(self):
        return self._epochs.value

    def set_epochs(self, epochs):
        """The epoch where training now ends (augmentations scale their phases with it)."""
        self._epochs.value = int(epochs)

    @property
    def clean(self):
        return None if self._clean.value < 0 else bool(self._clean.value)

    def set_clean(self, clean):
        """With a time budget the clean stretch is a share of the time, which the epochs cannot tell: the trainer sets
        it before every epoch."""
        self._clean.value = int(bool(clean))

    def label(self, i):
        """Model class id of record i, without reading the image."""
        c = self.records[i]['label']
        return int(c if self.id_map is None else self.id_map[c])

    def labels(self):
        return np.array([self.label(i) for i in range(len(self))], np.int64)

    def read(self, record):
        """Sample at original resolution. A fresh array: augmentations can change it freely."""
        return Sample(read_image(record['path']), self._label(record), record['path'])

    def _label(self, record):
        c = record['label']
        return int(c if self.id_map is None else self.id_map[c])

    def _read(self, record):
        """(Sample, the full image's (height, width) if it was read shrunk, else None)."""
        full = []

        def fit(h, w):
            size = shrink_size(h, w, *self.imgsz, self.shrink_factor)
            if size is not None:
                full.append((h, w))
            return size

        image = read_image(record['path'], fit if self.shrink_factor else None)
        return Sample(image, self._label(record), record['path']), (full[0] if full else None)

    def _full_size(self, full, read_shape, shape):
        """(geometry, shape) of the full image an image read shrunk stands for, now `shape` after its augmentation:
        as read, or turned by 90 degrees. (None, shape) when it was read in full or the augmentation resized it."""
        if full is not None:
            if shape == read_shape:
                return letterbox_geometry(*full, *self.imgsz), full
            if shape == read_shape[::-1]:
                return letterbox_geometry(full[1], full[0], *self.imgsz), full[::-1]
        return None, shape

    def __getitem__(self, i):
        t0 = time.perf_counter()
        record = self.records[i]
        sample, full = self._read(record)
        read_shape = sample.image.shape[:2]
        if self.augment is not None:
            epoch = self._epoch.value
            rng = np.random.default_rng([self.seed, epoch, i])     # the same with any number of workers
            ctx = AugmentContext(epoch, self._epochs.value, self.imgsz, rng, self.clean)
            sample = self._checked(self.augment(sample, ctx), record, sample.label)
        geometry, shape = self._full_size(full, read_shape, sample.image.shape[:2])
        image, geom = letterbox(sample.image, *self.imgsz, geometry=geometry)
        meta = {'index': i, 'path': record['path'], 'orig_shape': shape, 'geom': geom,
                'seconds': time.perf_counter() - t0}        # read + augment + letterbox, in the worker
        return image, sample.label, meta

    def _checked(self, sample, record, label):
        """Validate what the augmentation returned."""
        where = f"augment() on {record['path']}"
        if not isinstance(sample, Sample):
            raise AugmentError(f'{where}: must return the Sample it received (or a new rpclass Sample)')
        img = sample.image
        if not isinstance(img, np.ndarray) or img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
            raise AugmentError(f'{where}: image must be uint8 RGB [H, W, 3], got '
                               f"{getattr(img, 'dtype', type(img))} {getattr(img, 'shape', '')}")
        if min(img.shape[:2]) < 1:
            raise AugmentError(f'{where}: the image has no pixels left ({img.shape[1]}x{img.shape[0]})')
        if sample.label != label:
            raise AugmentError(f'{where}: an augmentation must not change the class ({label} -> {sample.label})')
        return sample


def collate(batch):
    images, labels, metas = zip(*batch)
    return torch.stack(images), torch.tensor(labels, dtype=torch.int64), list(metas)


CACHE_BYTES = 1 << 30            # a split is kept in memory up to this size of letterboxed images


class CachedSplit(Dataset):
    """A split read once (val: no augmentation, so the same tensors every epoch), its letterboxed images kept in
    memory: the validations after the first neither read nor resize. A validation spawned its workers and decoded
    its images again every epoch: with ~10 training steps an epoch of a 300-image dataset, that was a fifth of
    the epoch (VisA, 448 px: 0.8 of 3.6 s), the GPU idle. Everything else (records, read) is the split's own."""

    def __init__(self, dataset, workers=0, batch=32):
        from torch.utils.data import DataLoader
        self.dataset = dataset
        loader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=workers, collate_fn=_items)
        self.items = [item for items in loader for item in items]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

    def __getattr__(self, name):
        if name in ('dataset', 'items'):
            raise AttributeError(name)
        return getattr(self.dataset, name)


def cached(dataset, workers=0):
    """CachedSplit(dataset) when its letterboxed images fit in CACHE_BYTES (float32), else the dataset itself."""
    h, w = dataset.imgsz
    return CachedSplit(dataset, workers) if len(dataset) * 3 * h * w * 4 <= CACHE_BYTES else dataset


def _items(batch):
    return batch
