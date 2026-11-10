# RP-CLASS - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""Every metric against scikit-learn, and the rules that are ours: classes missing from the split, what makes an
epoch better."""
import numpy as np
import pytest

from rpclass.utils.metrics import evaluate

sk = pytest.importorskip('sklearn.metrics')


def case(n, c, seed, missing=()):
    rng = np.random.default_rng(seed)
    labels = rng.choice([k for k in range(c) if k not in missing], n)
    logits = rng.normal(0, 1, (n, c)) + 2.5 * np.eye(c)[labels] * rng.random((n, 1))   # often, not always, right
    return logits, labels


@pytest.mark.parametrize('n,c,seed', [(200, 2, 0), (300, 7, 1), (50, 12, 2)])
def test_metrics_match_scikit_learn(n, c, seed):
    logits, labels = case(n, c, seed)
    names = {i: f'c{i}' for i in range(c)}
    m = evaluate(logits, labels, names)
    pred = logits.argmax(1)
    present = sorted(set(labels))
    assert m.top1 == pytest.approx(sk.accuracy_score(labels, pred))
    assert m.macro_f1 == pytest.approx(sk.f1_score(labels, pred, labels=present, average='macro', zero_division=0))
    assert m.balanced_accuracy == pytest.approx(sk.balanced_accuracy_score(labels, pred))
    p = np.exp(logits - logits.max(1, keepdims=True))
    assert m.loss == pytest.approx(sk.log_loss(labels, p / p.sum(1, keepdims=True), labels=range(c)))
    assert (m.confusion_matrix == sk.confusion_matrix(labels, pred, labels=range(c))).all()
    if c > 5:
        assert m.top5 == pytest.approx(sk.top_k_accuracy_score(labels, logits, k=5, labels=range(c)))
    else:
        assert m.top5 == -1
    per = sk.precision_recall_fscore_support(labels, pred, labels=range(c), zero_division=0)
    for i, row in enumerate(m.per_class):
        assert (row['precision'], row['recall'], row['f1'], row['images']) == pytest.approx(
            (per[0][i], per[1][i], per[2][i], per[3][i]))


def test_a_class_missing_from_the_split_does_not_count():
    """A rare class with no validation image would add a 0 to macro-F1 for no fault of the model."""
    logits, labels = case(100, 4, 3, missing=(2,))
    names = {i: f'c{i}' for i in range(4)}
    m = evaluate(logits, labels, names)
    pred = logits.argmax(1)
    assert m.macro_f1 == pytest.approx(sk.f1_score(labels, pred, labels=[0, 1, 3], average='macro', zero_division=0))
    assert m.per_class[2]['images'] == 0 and '-' in m.table(names).splitlines()[4]


def test_the_better_epoch_has_a_higher_macro_f1_then_a_lower_loss():
    names = {0: 'a', 1: 'b'}
    labels = np.array([0, 1, 0, 1])
    sure = evaluate(np.array([[3, -3], [-3, 3], [3, -3], [-3, 3]], float), labels, names)
    unsure = evaluate(np.array([[1, -1], [-1, 1], [1, -1], [-1, 1]], float), labels, names)
    wrong = evaluate(np.array([[3, -3], [-3, 3], [-3, 3], [-3, 3]], float), labels, names)
    assert sure.macro_f1 == unsure.macro_f1 == 1.0 and sure.key() > unsure.key() > wrong.key()
    assert sure.fitness == sure.macro_f1


def test_an_empty_split_gives_zeros_not_an_error():
    m = evaluate(np.zeros((0, 3)), np.zeros(0, np.int64), {0: 'a', 1: 'b', 2: 'c'})
    assert m.images == 0 and m.top1 == 0.0 and m.confusion_matrix.shape == (3, 3)


def test_equal_macro_f1_values_are_equal_floats():
    """One mistake 0 -> 2, or one mistake 2 -> 0: the same macro-F1 (6/7, 1 and 8/9 in another order). Summed as
    floats the two could differ in the last bit, and the tie would not go to the lower loss."""
    names = {0: 'a', 1: 'b', 2: 'c'}
    labels = np.repeat([0, 1, 2], 4)
    one, other = np.eye(3)[labels] * 4, np.eye(3)[labels] * 4
    one[0] = [0, 0, 4]
    other[8] = [4, 0, 0]
    a, b = evaluate(one, labels, names), evaluate(other, labels, names)
    assert a.macro_f1 == b.macro_f1


def test_non_finite_scores_count_as_wrong_not_as_the_first_class():
    names = {0: 'a', 1: 'b'}
    logits = np.array([[np.nan, np.nan], [5, -5], [-5, 5], [np.inf, 0]])
    m = evaluate(logits, np.array([0, 0, 1, 0]), names)
    assert m.nonfinite == 2 and m.top1 == 0.5 and m.loss == float('inf') and m.key()[1] == -float('inf')
    assert m.confusion_matrix.sum() == 2 and m.per_class[0]['recall'] == pytest.approx(1 / 3)


def test_the_prediction_grid_never_reads_more_than_16_images():
    """The most confident mistakes first, then the least confident right answers, 16 in all: RP-OD caps it too."""
    from rpclass.utils.plots import pick_examples
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(3), 1000)
    labels = rng.integers(0, 3, 1000)
    picks = pick_examples(probs, labels, 16)
    wrong = probs.argmax(1) != labels
    assert len(picks) == 16 and wrong[picks].all()
    assert list(probs[picks].max(1)) == sorted(probs[picks].max(1), reverse=True)
