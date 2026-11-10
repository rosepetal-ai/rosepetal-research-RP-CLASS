# RP-CLASS - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""Prediction: letterbox -> network -> softmax. The class is the one with the highest score; no threshold.

The post-processing a serving pipeline applies to the exported model is the same: softmax over the logits, argmax.
"""
import numpy as np
import torch
from PIL import Image, ImageDraw

from ..data import letterbox, load_images


class Result:
    def __init__(self, image, probs, names, path=None):
        self.orig_img = image        # the RGB image that was analysed
        self.orig_shape = image.shape[:2]
        self.probs = probs           # [C] numpy, sums to 1
        self.names = names
        self.path = path
        self.top1 = int(probs.argmax())
        self.conf = float(probs[self.top1])

    @property
    def name(self):
        return self.names[self.top1]

    def top(self, k=5):
        """The k most probable classes as (class id, probability), the most probable first."""
        idx = np.argsort(-self.probs)[:k]
        return [(int(i), float(self.probs[i])) for i in idx]

    def __repr__(self):
        return f"Result({self.path or 'array'}, {self.orig_shape[1]}x{self.orig_shape[0]}: {self.name} {self.conf:.2f})"

    def plot(self):
        """RGB image (numpy) with the predicted class written on it."""
        im = Image.fromarray(np.ascontiguousarray(self.orig_img))
        d = ImageDraw.Draw(im)
        text = f'{self.name} {self.conf:.2f}'
        x0, y0, x1, y1 = d.textbbox((0, 0), text)
        d.rectangle([0, 0, x1 + 8, y1 + 8], fill=(255, 255, 255))
        d.text((4, 4), text, fill=(0, 0, 0))
        return np.asarray(im)

    def save(self, path):
        Image.fromarray(self.plot()).save(path)
        return path


@torch.no_grad()
def predict(net, names, imgsz, source, color='RGB', device='cpu', batch=16):
    items = load_images(source, color)
    net = net.to(device).eval()
    results = []
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        x = torch.stack([letterbox(img, imgsz[0], imgsz[1], device)[0] for img, _ in chunk])
        probs = net(x).float().softmax(1).cpu().numpy()
        results += [Result(img, p, names, path) for (img, path), p in zip(chunk, probs)]
    return results
