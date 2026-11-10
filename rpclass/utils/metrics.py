# RP-CLASS - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""Classification metrics of a split. Every image counts once, with the class of its highest score: no threshold.

- top-1 accuracy, and top-5 when there are more than 5 classes (-1 otherwise);
- per class: precision, recall, F1 and the number of images;
- macro-F1 and balanced accuracy (the mean recall) over the classes that have images in the split, so that a class
  missing from validation does not count (it shows as '-'), and every class weighs the same however few images it has:
  in inspection the rare class is usually the one that matters;
- the mean cross-entropy of the split, which breaks ties between epochs of equal macro-F1;
- the confusion matrix: rows the true class, columns the predicted one.

The best epoch has the highest macro-F1 (fitness). Macro-F1 and balanced accuracy are computed exactly from the counts
(fractions), so two epochs with the same rational value compare equal and the loss breaks the tie (docs/DESIGN.md,
decision 7); summed as floats they could differ in the last bit. Images whose scores are not finite (a float16
overflow) count as wrong, never as the first class, and make the loss infinite. The tests check every number
against scikit-learn.
"""
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

TOP_K = 5


@dataclass
class Metrics:
    top1: float
    top5: float               # -1: not more than TOP_K classes
    macro_f1: float
    balanced_accuracy: float
    loss: float
    fitness: float
    images: int = 0
    per_class: list = field(default_factory=list)
    confusion_matrix: np.ndarray = None
    nonfinite: int = 0        # images whose scores were not finite (counted as wrong)

    def to_dict(self):
        keys = ('top1', 'top5', 'macro_f1', 'balanced_accuracy', 'loss', 'fitness')
        return {k: float(getattr(self, k)) for k in keys}

    def key(self):
        """What makes an epoch better: a higher macro-F1, then a lower loss."""
        return self.macro_f1, -self.loss

    def table(self, names):
        rows = [f"{'class':24}{'images':>8}{'P':>7}{'R':>7}{'F1':>7}"]
        rows.append(f"{'all':24}{self.images:8d}{'':>7}{self.balanced_accuracy:7.3f}{self.macro_f1:7.3f}")
        for r in self.per_class:
            if r['images'] == 0:   # no image of this class in the split: no metric makes sense
                rows.append(f"{r['name'][:23]:24}{0:8d}" + ''.join(f'{"-":>7}' for _ in range(3)))
                continue
            rows.append(f"{r['name'][:23]:24}{r['images']:8d}{r['precision']:7.3f}{r['recall']:7.3f}{r['f1']:7.3f}")
        return '\n'.join(rows)


def softmax(logits):
    z = logits - logits.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def evaluate(logits, labels, names):
    """logits: [N, C] class scores before the softmax; labels: [N] true class ids."""
    logits = np.asarray(logits, np.float64).reshape(len(labels), len(names))
    labels = np.asarray(labels, np.int64)
    n, c = logits.shape
    if n == 0:
        return Metrics(0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0, [], np.zeros((c, c), np.int64))
    finite = np.isfinite(logits).all(1)
    loss = float('inf')
    if finite.all():
        z = logits - logits.max(1, keepdims=True)
        log_p = z - np.log(np.exp(z).sum(1, keepdims=True))
        loss = float(-log_p[np.arange(n), labels].mean())
    pred = np.where(finite, np.nan_to_num(logits).argmax(1), -1)          # -1: no prediction, always wrong
    cm = np.zeros((c, c), np.int64)
    np.add.at(cm, (labels[finite], pred[finite]), 1)
    tp, predicted = np.diag(cm), cm.sum(0)
    support = np.bincount(labels, minlength=c)                           # the wrong non-finite images included
    present = [i for i in range(c) if support[i] > 0]
    f1 = [Fraction(2 * int(tp[i]), int(support[i] + predicted[i])) if support[i] + predicted[i] else Fraction(0)
          for i in range(c)]
    recall = [Fraction(int(tp[i]), int(support[i])) if support[i] else Fraction(0) for i in range(c)]
    precision = [Fraction(int(tp[i]), int(predicted[i])) if predicted[i] else Fraction(0) for i in range(c)]
    macro_f1 = float(sum(f1[i] for i in present) / len(present))
    top5 = -1.0
    if c > TOP_K:
        top = np.argsort(-np.nan_to_num(logits, nan=-np.inf), 1)[:, :TOP_K]
        top5 = float(((top == labels[:, None]).any(1) & finite).mean())
    per_class = [{'name': names[i], 'images': int(support[i]), 'precision': float(precision[i]),
                  'recall': float(recall[i]), 'f1': float(f1[i])} for i in range(c)]
    return Metrics(top1=float(tp.sum() / n), top5=top5, macro_f1=macro_f1,
                   balanced_accuracy=float(sum(recall[i] for i in present) / len(present)), loss=loss,
                   fitness=macro_f1, images=n, per_class=per_class, confusion_matrix=cm,
                   nonfinite=int((~finite).sum()))
