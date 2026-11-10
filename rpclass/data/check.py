# RP-CLASS - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""Classification datasets as folders: read the layout, check every image once, cache the result.

    <root>/
        train/<class>/<image>       required
        val/<class>/<image>         required, not empty ('validation' or 'valid' are accepted too)
        test/<class>/<image>        optional, never used for any training decision

The classes are the class folders found in any split, and their ids follow the sorted names. On the usual layouts
(every class folder in every split, some of them empty) that is exactly YOLO-cls's rule, the sorted folders of
train/; a class folder only in val/ or test/ is kept too, with a warning that it has no training images.
'unlabeled/' and any other top-level folder are ignored with a warning. Images may sit in subfolders of their class
folder.

Errors (the dataset cannot be used as it is) stop everything and list every offending file with the reason: an
unreadable or truncated image (every image is decoded once), an image that is not 8-bit or too large to decode, an
image directly inside a split folder. Warnings are reported and training goes on: identical images across or inside
splits, a class without training images, no test images.
"""
import hashlib
import io
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from .sources import IMAGE_EXTENSIONS

CACHE_VERSION = 1
CACHE_DIR = Path(os.environ.get('RPCLASS_CACHE', Path.home() / '.cache' / 'rpclass')) / 'datasets'
SPLITS = {'train': ('train',), 'val': ('val', 'validation', 'valid'), 'test': ('test',)}
EIGHT_BIT_MODES = {'1', 'L', 'LA', 'P', 'PA', 'RGB', 'RGBA', 'RGBX', 'CMYK', 'YCbCr'}


class DatasetError(ValueError):
    pass


# ------------------------------------------------------------------------------------------------ layout
def load_layout(root):
    """Returns {'root', 'names': {id: name}, 'splits': {'train': [(path, class name)], 'val': [...], 'test': [...] or
    None}, 'ignored': [folders], 'loose': {split: [images outside any class folder]}}."""
    root = Path(root)
    if not root.is_dir():
        raise DatasetError(f'dataset folder not found: {root}')
    folders = {p.name: p for p in sorted(root.iterdir()) if p.is_dir()}
    found, used = {}, set()
    for split, aliases in SPLITS.items():
        name = next((a for a in aliases if a in folders), None)
        if name is None:
            if split != 'test':
                raise DatasetError(f"{root}: a '{split}' folder is required "
                                   f"({' or '.join(repr(a) for a in aliases)}, with one subfolder per class)")
            continue
        found[split] = folders[name]
        used.add(name)
    classes, splits, loose = set(), {}, {}
    for split, folder in found.items():
        items, loose[split] = [], []
        for p in sorted(folder.iterdir()):
            if p.is_dir():
                classes.add(p.name)
                items += [(str(f), p.name) for f in sorted(p.rglob('*')) if f.suffix.lower() in IMAGE_EXTENSIONS]
            elif p.suffix.lower() in IMAGE_EXTENSIONS:
                loose[split].append(str(p))
        splits[split] = items
    splits.setdefault('test', None)
    if len(classes) < 2:
        raise DatasetError(f'{root}: classification needs at least 2 class folders, found {sorted(classes)}')
    for split in ('train', 'val'):
        if not splits[split] and not loose[split]:
            raise DatasetError(f"{root}: '{split}' contains no images")
    names = dict(enumerate(sorted(classes)))
    return {'root': root, 'names': names, 'splits': splits, 'ignored': sorted(set(folders) - used), 'loose': loose}


# ------------------------------------------------------------------------------------------------ checking
def _check_one(path):
    """Returns (record or None, problems) for one image: size, md5, and whether it can be used at all. The image is
    decoded once, as training will decode it: verify() reads no pixel of a JPEG, BMP or TIFF, so a truncated one would
    pass and stop the training at its first read. A JPEG is decoded at 1/8 scale: all its data is still read."""
    try:
        data = Path(path).read_bytes()
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
        with Image.open(io.BytesIO(data)) as im:
            mode, (w, h) = im.mode, im.size
            if im.format == 'JPEG':
                im.draft(im.mode, (max(1, w // 8), max(1, h // 8)))
            im.load()
    except FileNotFoundError:
        return None, [('error', path, 'image file not found')]
    except Image.DecompressionBombError:
        return None, [('error', path, f'image too large to decode (over {2 * Image.MAX_IMAGE_PIXELS:,} pixels): '
                                      f'downscale it')]
    except Exception as e:  # noqa: BLE001 - any decoding failure makes the image unusable
        return None, [('error', path, f'unreadable or truncated image ({type(e).__name__})')]
    if mode not in EIGHT_BIT_MODES:
        return None, [('error', path, f'not an 8-bit image (mode {mode}); RP-CLASS only works with 8 bits')]
    return {'path': path, 'w': w, 'h': h, 'hash': hashlib.md5(data).hexdigest()}, []


def _cache_key(paths):
    h = hashlib.sha1(f'{CACHE_VERSION}'.encode())
    for p in paths:
        h.update(os.fsencode(p))         # file names that are not valid UTF-8 too (archives made on Windows)
        try:
            st = os.stat(p)
            h.update(f'{st.st_size}:{st.st_mtime_ns}'.encode())
        except FileNotFoundError:
            h.update(b'-')
    return h.hexdigest()


def scan_images(paths, workers=8, use_cache=True):
    """Check every image of a list. Returns (records keyed by path, problems). Cached until any file changes."""
    cache = CACHE_DIR / f'{_cache_key(paths)}.pkl' if use_cache else None
    if use_cache and cache.exists():
        with open(cache, 'rb') as f:
            return pickle.load(f)
    with ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(_check_one, paths))
    records = {r['path']: r for r, _ in results if r is not None}
    problems = [p for _, ps in results for p in ps]
    if use_cache:
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_suffix(f'.tmp{os.getpid()}')
            with open(tmp, 'wb') as f:
                pickle.dump((records, problems), f)
            os.replace(tmp, cache)
        except OSError:
            pass  # a read-only home only costs a re-scan next time
    return records, problems


# ------------------------------------------------------------------------------------------------ report
def check_dataset(root, use_cache=True):
    """Check a whole dataset. Raises DatasetError listing every error; returns the dataset, its warnings and stats:
    {'root', 'names', 'records': {split: [{'path', 'label', 'w', 'h', 'hash'}]}, 'warnings', 'stats'}."""
    ds = load_layout(root)
    names = ds['names']
    ids = {n: i for i, n in names.items()}
    problems = [('error', split, path, 'image outside any class folder: put it in the folder of its class')
                for split, paths in ds['loose'].items() for path in paths]
    problems += [('warning', '-', str(Path(root) / f), 'folder ignored (not train, val or test)')
                 for f in ds['ignored']]
    records = {}
    for split, items in ds['splits'].items():
        if items is None:
            continue
        scanned, found = scan_images([p for p, _ in items], use_cache=use_cache)
        problems += [(sev, split, where, why) for sev, where, why in found]
        records[split] = [dict(scanned[p], label=ids[c]) for p, c in items if p in scanned]
    if not ds['splits']['test']:            # no test/ folder, or an empty one (a 0 % test split)
        problems.append(('warning', 'test', str(root), 'no test images: validation metrics are optimistic, because '
                         'validation is used to pick the best epoch'))
    counts = {s: [sum(1 for r in recs if r['label'] == i) for i in names] for s, recs in records.items()}
    for i, n in names.items():
        if counts['train'][i] == 0:
            problems.append(('warning', 'train', n, 'class without training images: the model cannot learn it'))
    seen = {}
    for split, recs in records.items():
        for r in recs:
            seen.setdefault(r['hash'], []).append((split, r['path']))
    for copies in seen.values():
        if len(copies) > 1:
            splits = {s for s, _ in copies}
            kind = f"across {' and '.join(sorted(splits))}" if len(splits) > 1 else f'inside {copies[0][0]}'
            problems.append(('warning', copies[0][0], copies[0][1],
                             f"identical image {kind}: {', '.join(p for _, p in copies[1:])}"))
    errors = [p for p in problems if p[0] == 'error']
    warnings = [p for p in problems if p[0] == 'warning']
    if errors:
        raise DatasetError(format_problems(errors, f'{len(errors)} error(s) in {root}, fix them and run again'))
    stats = {s: {'images': len(recs), 'per_class': counts[s]} for s, recs in records.items()}
    return {'root': ds['root'], 'names': names, 'records': records, 'warnings': warnings, 'stats': stats}


def format_problems(problems, title, limit=50):
    lines = [title + ':']
    for _, split, where, why in problems[:limit]:
        lines.append(_printable(f'  [{split}] {where}: {why}'))
    if len(problems) > limit:
        lines.append(f'  ... and {len(problems) - limit} more')
    return '\n'.join(lines)


def _printable(path):
    """A file name that is not valid UTF-8 written with escapes, so printing it cannot fail."""
    return str(path).encode('utf-8', 'backslashreplace').decode('utf-8')
