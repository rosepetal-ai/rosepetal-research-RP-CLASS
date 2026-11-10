# RP-CLASS - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""The training loader: shuffled micro-batches, each one prepared by all the workers at once.

A DataLoader worker prepares a whole batch, its images one after another. At the start of every epoch the first
step waited for `micro` images to be read, augmented and letterboxed in a row (VisA at 448 px: 32 x 40 ms, 1.3 s).
With the ~10 steps an epoch of a 300-image dataset has, every worker held one micro-batch: the GPU waited for
all of them at once, then ran the steps while the CPU sat idle.

Here a worker prepares a chunk of a micro-batch, micro / workers images rounded up, and the main process joins the
chunks, in pinned memory for the copy to the GPU. The first step waits for one chunk, and the steps run as the
micro-batches come in. The order, the images and the micro-batches are those of a DataLoader of whole batches with the
same sampler (tests/test_dataset.py compares them; tests/test_train.py compares every number of a training): only who
prepares each image changes.
"""
import math

import torch
from torch.utils.data import DataLoader, RandomSampler, Sampler

from .dataset import collate

CHUNK = None            # images per worker task; None: micro / workers rounded up (micro without workers)


class MicroBatches:
    """Iterable of (images, labels, metas) micro-batches of `micro` images in the order of
    RandomSampler(dataset, generator=order); the last incomplete one is dropped when the dataset holds a whole one."""

    def __init__(self, dataset, micro, order, workers=0, pin=False, seed=0):
        self.micro, self.pin = micro, pin
        self.drop_last = len(dataset) >= micro
        self.chunk = CHUNK or (max(1, math.ceil(micro / workers)) if workers else micro)
        self.sampler = RandomSampler(dataset, generator=order)
        self.loader = DataLoader(dataset, batch_sampler=_Chunks(self.sampler, micro, self.chunk, self.drop_last),
                                 num_workers=workers, collate_fn=collate, persistent_workers=workers > 0,
                                 generator=torch.Generator().manual_seed(seed))

    def __len__(self):
        full, rest = divmod(len(self.sampler), self.micro)
        return full + int(bool(rest) and not self.drop_last)

    def __iter__(self):
        chunks = iter(self.loader)
        n = len(self.sampler)
        for b in range(len(self)):
            size = min(self.micro, n - b * self.micro)
            yield _joined([next(chunks) for _ in range(math.ceil(size / self.chunk))], self.pin)


class _Chunks(Sampler):
    """The sampler's indices in micro-batches, each cut into chunks."""

    def __init__(self, sampler, micro, chunk, drop_last):
        self.sampler, self.micro, self.chunk, self.drop_last = sampler, micro, chunk, drop_last

    def __iter__(self):
        batch = []
        for i in self.sampler:
            batch.append(i)
            if len(batch) == self.micro:
                yield from _cut(batch, self.chunk)
                batch = []
        if batch and not self.drop_last:
            yield from _cut(batch, self.chunk)

    def __len__(self):
        full, rest = divmod(len(self.sampler), self.micro)
        return full * math.ceil(self.micro / self.chunk) + (math.ceil(rest / self.chunk) if rest and not self.drop_last
                                                            else 0)


def _cut(batch, chunk):
    return [batch[k:k + chunk] for k in range(0, len(batch), chunk)]


def _joined(chunks, pin):
    images, labels, metas = zip(*chunks)
    return _cat(images, pin), _cat(labels, pin), [m for part in metas for m in part]


def _cat(tensors, pin):
    """The tensors end to end (one copy), in pinned memory if asked."""
    if len(tensors) == 1:
        return tensors[0].pin_memory() if pin else tensors[0]
    out = torch.empty((sum(len(t) for t in tensors), *tensors[0].shape[1:]), dtype=tensors[0].dtype, pin_memory=pin)
    return torch.cat(tensors, out=out)
