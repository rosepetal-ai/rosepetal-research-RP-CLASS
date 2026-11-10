# RP-CLASS - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""Recipe overrides for the recipe experiments (benchmarks/run.py, benchmarks/public/benchmark.py).

    --recipe backbone_lr=2e-4 lr=2e-3 label_smoothing=0.1 --tag lr2x-ls

--recipe sets entries of rpclass.engine.trainer.RECIPE for every run of the command (values read as Python literals;
an unknown key is an error), and needs --tag, which names those runs in their folder: <model>[+<augment>]@<tag>, so
that runs of different recipes never share a folder and one summary compares them. The augmentation part follows
benchmarks/augments.py: none for hflip, +<name> for any other, 'default' resolved to its concrete name. The library
never reads anything from here: a recipe that wins moves into RECIPE itself.
"""
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rpclass.data.dataset as D  # noqa: E402
from rpclass.data import any_resolution  # noqa: E402
from rpclass.engine import trainer as T  # noqa: E402

from augments import resolve  # noqa: E402


def apply(pairs, tag):
    """Set the overrides in T.RECIPE; returns them as a dict."""
    out = {}
    for pair in pairs or ():
        key, sep, value = pair.partition('=')
        if not sep or key not in T.RECIPE or key == 'seed':
            raise SystemExit(f"--recipe {pair}: expected key=value with a key of the recipe other than seed "
                             f"({', '.join(k for k in T.RECIPE if k != 'seed')})")
        try:
            out[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            raise SystemExit(f'--recipe {pair}: the value is no Python literal (e.g. 2e-4, 0.1, True)') from None
    if out and not tag:
        raise SystemExit('--recipe needs --tag, so that its runs get their own folder')
    T.RECIPE.update(out)
    return out


class _AnyResolution:
    """An augmentation declared any_resolution (rpclass.data.dataset): large training images reach it shrunk. Its
    name (args.yaml) says so."""

    def __init__(self, fn, factor):
        self.fn, self.__name__ = fn, f"{getattr(fn, '__name__', 'augment')}+shrink{factor}"
        any_resolution(self)

    def __call__(self, sample, ctx):
        return self.fn(sample, ctx)


def shrunk(augment, factor, tag):
    """--shrink-factor: the augmentation, with large training images read at `factor` times their letterbox, as the
    library's own augmentations read them (rpclass.data.dataset, SHRINK_FACTOR). benchmarks/augments.py's functions
    read every image in full, so that their runs never move with the library."""
    if not tag:
        raise SystemExit('--shrink-factor needs --tag, so that its runs get their own folder')
    if augment is None:
        raise SystemExit('--shrink-factor needs an augmentation: without one, every training image is read as it is')
    if factor < 3 or factor % 2 != 1:
        raise SystemExit(f'--shrink-factor {factor}: an odd integer, 3 or more (rpclass.data.letterbox.shrink_size)')
    D.SHRINK_FACTOR = factor
    return _AnyResolution(augment, factor)


def folder(model, augment='default', tag=''):
    """The run folder's name for a model, an augmentation and a recipe tag."""
    augment = resolve(augment)
    name = model if augment == 'hflip' else f'{model}+{augment}'
    return f'{name}@{tag}' if tag else name
