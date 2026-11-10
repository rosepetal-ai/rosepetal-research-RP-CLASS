# RP-CLASS - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""Validation plots: the confusion matrix, and a grid of images with their true and predicted classes, the most
confident mistakes first (the images that say most about what the model gets wrong)."""
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

GRID_IMAGES = 16
TILE = 480
MAX_CELL_TEXT = 30       # more classes than this: no numbers in the confusion matrix cells
RIGHT, WRONG = (60, 200, 90), (230, 60, 60)


def plot_confusion_matrix(metrics, names, path):
    m = metrics.confusion_matrix
    labels = [names[i] for i in range(len(names))]
    norm = m / np.maximum(m.sum(1, keepdims=True), 1)
    n = len(labels)
    fig, axes = plt.subplots(1, 2, figsize=(min(6 + n, 28), min(4 + 0.6 * n, 16)), dpi=120)
    for ax, data, title, fmt in ((axes[0], m, 'counts', '{:d}'), (axes[1], norm, 'per true class', '{:.2f}')):
        ax.imshow(data, cmap='Blues')
        ax.set(xticks=range(n), yticks=range(n), xlabel='predicted', ylabel='true',
               title=f'Confusion matrix, {title}\nmacro-F1 {metrics.macro_f1:.3f} · top-1 {metrics.top1:.3f}')
        ax.set_xticklabels(labels, rotation=45 if n <= MAX_CELL_TEXT else 90, ha='right',
                           fontsize=8 if n <= MAX_CELL_TEXT else 5)
        ax.set_yticklabels(labels, fontsize=8 if n <= MAX_CELL_TEXT else 5)
        if n > MAX_CELL_TEXT:
            continue
        top = data.max() if data.size else 1
        for i in range(n):
            for j in range(n):
                v = data[i, j]
                ax.text(j, i, fmt.format(int(v) if fmt == '{:d}' else v), ha='center', va='center', fontsize=8,
                        color='white' if v > top / 2 else 'black')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def pick_examples(probs, labels, k=GRID_IMAGES):
    """Indices for the grid: the mistakes, most confident first, then the right ones, least confident first."""
    pred, conf = probs.argmax(1), probs.max(1)
    wrong = np.flatnonzero(pred != labels)
    right = np.flatnonzero(pred == labels)
    return [int(i) for i in np.concatenate([wrong[np.argsort(-conf[wrong])], right[np.argsort(conf[right])]])[:k]]


def plot_image_grid(items, names, path):
    """items: up to 16 (rgb image, true class, predicted class, confidence). Green frame: right; red: wrong."""
    tiles = []
    for img, true, pred, conf in items[:GRID_IMAGES]:
        im = Image.fromarray(np.ascontiguousarray(img))
        s = (TILE - 8) / max(im.size)
        im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))))
        canvas = Image.new('RGB', (TILE, TILE), (24, 24, 24))
        canvas.paste(im, ((TILE - im.width) // 2, (TILE - im.height) // 2))
        d = ImageDraw.Draw(canvas)
        col = RIGHT if true == pred else WRONG
        d.rectangle([0, 0, TILE - 1, TILE - 1], outline=col, width=4)
        text = f'{names[int(pred)]} {conf:.2f}' + ('' if true == pred else f'  (true: {names[int(true)]})')
        x0, y0, x1, y1 = d.textbbox((0, 0), text)
        d.rectangle([4, 4, x1 + 12, y1 + 12], fill=col)
        d.text((8, 8), text, fill=(0, 0, 0))
        tiles.append(canvas)
    if not tiles:
        return
    cols = int(np.ceil(np.sqrt(len(tiles))))
    rows = int(np.ceil(len(tiles) / cols))
    grid = Image.new('RGB', (cols * TILE, rows * TILE), (0, 0, 0))
    for k, t in enumerate(tiles):
        grid.paste(t, ((k % cols) * TILE, (k // cols) * TILE))
    grid.save(path, quality=90)


def save_plots(metrics, names, items, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(metrics, names, out_dir / 'confusion_matrix.png')
    plot_image_grid(items, names, out_dir / 'predictions.jpg')
