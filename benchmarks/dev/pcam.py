# RP-CLASS - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""Development suite (never part of the benchmark): PatchCamelyon's validation split, tumour (nok) or no tumour (ok)
in the centre of a 96 x 96 patch of an H&E lymph-node slide, one task, grouped by slide.

Data: Zenodo record 2546921, the mirror the authors' README names (Veeling, Linmans, Winkens, Cohen and Welling,
'Rotation Equivariant CNNs for Digital Pathology', MICCAI 2018, arXiv:1806.03962), only the validation split:
    camelyonpatch_level_2_split_valid_x.h5.gz     805,965,320 bytes   x: 32,768 x 96 x 96 x 3, uint8 RGB
    camelyonpatch_level_2_split_valid_y.h5.gz           3,038 bytes   y: 32,768 x 1 x 1 x 1, uint8, 1 = tumour
    camelyonpatch_level_2_split_valid_meta.csv      1,850,810 bytes   per row: coord_y, coord_x (level-0 px, up to
                                                                      217,664), tumor_patch, center_tumor_patch, wsi
each checked by its size and the MD5 Zenodo publishes (all three matched on 2026-11-06; torchvision pins the same two
h5 MD5s). The README's MD5 table gives valid the test files' sums and test the valid ones; these files are the
validation split, as the meta file shows: its 54 slides are all of the Camelyon16 training set (32 normal, 22 with
tumour), the 20 % the authors held out of it. A patch is 96 x 96 px at Camelyon16's level 2 (10x, 0.97 um/px), from
slides scanned at two centres, and is positive when its central 32 x 32 px hold any tumour. The meta file's rows are
the h5 rows (its index column), and its center_tumor_patch equals y on all 32,768.
Licence: CC0 for the data, the README: 'The data is provided under the CC0 License, following the license of
Camelyon16.' (the repository's code is MIT, and so is the Zenodo record's licence field). Cite Veeling et al. 2018
and Ehteshami Bejnordi et al., JAMA 318(22):2199, 2017 (Camelyon16).

What the files hold (counted on 2026-11-06): 16,369 rows nok and 16,399 ok, but 2,181 patches are listed 2 to 22
times, at the same slide and coordinates and with the same pixels (all checked; none with two labels or two
slides): 4,660 repeated rows, 4,647 of them nok, so the split's 50/50 balance comes in part from patches drawn again.
Once each, 28,108 distinct patches: ok 16,386 (10,250 of the 32 normal slides, 259-435 per slide; 6,136 of the tumour
slides) and nok 11,722 (95-1,397 per tumour slide; 3 tumour slides have no ok patch). The patches overlap: 12,694 of
the 28,108 overlap another of their slide (coordinates under 384 level-0 px apart, the 96 px at level 2).

Choices:
    labels    ok (y 0) / nok (y 1), as every binary inspection task here; POSITIVE 'nok'
    groups    the slide (wsi in the meta file): its patches overlap and share a patient, a stain and a scanner, so
              no slide crosses test, val and train
    copies    a patch listed again is listed once (its first row) before anything else: finalize would keep one of
              the byte copies anyway, and so the presample holds distinct patches and the shares are those finalize
              would compute on the whole split, ok 0.583 and nok 0.417, not the rows' 50/50
    presample PER_CLASS = 2,000 distinct patches of each class at random (common.presample of the distinct listing,
              with common.shares of it): 4,000 PNG files instead of 28,108. The test takes 500 of each (its cap;
              0.4 x 2,000 is 800), so about a quarter of each class's slides; val 42 + 58 and a training draw
              125 + 175 come from the rest
    files     the h5 files are gunzipped (gzip) and read with h5py, once; only the presampled patches are written,
              as patches/<label>/<h5 row>.png; listing.csv keeps the 28,108 distinct patches (row, label, slide), from
              which rows() draws the same presample again and takes the shares
    size      the patches stay 96 x 96; IMGSZ 224 upsamples them 2.3 times, the same for every method
    disk      0.81 GB of x.h5.gz and 0.91 GB of x.h5 at the peak (the .gz is deleted once unpacked, the .h5 once the
              patches are out), 0.1 GB of PNG after; the prepared task takes 48 MB

Counts (prepare.py, 2026-11-06, 2 minutes): the 4,000 presampled patches hold no copy, no look-alike and 8 featureless
patches. Test 500 nok (from 6 slides) + 500 ok (from 15: 329 of normal slides and 171 of tumour slides, 66 / 34 %
against 63 / 37 % among all ok patches); val 42 nok (1 slide) + 58 ok (2 slides); pool 1,404 + 1,353; a training
draw 125 + 175 for each seed, from 15 and 34 slides. 57 and 86 patches are left out with the slides the test and val
drew. Every label agrees with y and the meta file, and every PNG with its h5 row.

Caveats: 22 tumour slides only, so the test's nok comes from 6 slides and val's from 1: few patients decide them. The
task is the central 32 x 32 px, the rest of the patch is context: 88 ok patches show tumour outside their centre (meta
tumor_patch), kept as labelled. Negatives were picked by the authors' hard-negative mining with a small CNN, so ok
patches are harder than random tissue. Histology, not parts: it checks that a recipe choice holds on a very different
texture domain (stain colour varies between the two centres).
"""
import csv
import gzip
import re
import shutil
from pathlib import Path

from PIL import Image

import common
import fetch

NAME = 'pcam'
TASKS = (NAME,)
IMGSZ = 224
LICENCE = "CC0 (the authors' README, following Camelyon16's licence; the Zenodo record's licence field says MIT)"
SOURCE = 'https://github.com/basveeling/pcam'
POSITIVE = 'nok'

FILES = 'https://zenodo.org/records/2546921/files'
X = ('camelyonpatch_level_2_split_valid_x.h5.gz', 805965320, 'd5b63470df7cfa627aeec8b9dc0c066e')   # MD5, as Zenodo
Y = ('camelyonpatch_level_2_split_valid_y.h5.gz', 3038, '2b85f58b927af9964a4c15b8f7e8f179')
META = ('camelyonpatch_level_2_split_valid_meta.csv', 1850810, '67589e00a4a37ec317f2d1932c7502ca')
SLIDE = re.compile(r'camelyon16_train_(normal|tumor)_\d{3}')
LISTING = 'listing.csv'
ROWS, DISTINCT = 32768, 28108
PER_CLASS = 2000
NOTES = {'copies': '4,660 rows that repeat a patch (the same slide, coordinates and pixels; 4,647 of them nok) are '
                   'listed once: 28,108 distinct patches, nok share 0.417',
         'presample': f'{PER_CLASS:,} distinct patches of each class, at random, with the shares of all 28,108'}


def download(raw):
    """raw/listing.csv (the distinct patches, from y and the meta file) and raw/patches/<label>/<row>.png (the
    presampled ones), once; each h5 file and its .gz are deleted once read."""
    raw = Path(raw)
    patches = raw / 'patches'
    if not (patches / '.converted').exists():
        listing = _listing(raw)
        _convert(_unpacked(raw, X), common.presample(listing, PER_CLASS), patches)
    for name, _, _ in (X, Y):
        (raw / name).unlink(missing_ok=True)
        (raw / name[:-3]).unlink(missing_ok=True)
    return raw


def rows(folder, task):
    folder = Path(folder)
    listing = _read(folder / LISTING)
    out = [{'path': folder / 'patches' / r['label'] / f"{r['row']:05d}.png", 'label': r['label'], 'group': r['slide']}
           for r in common.presample(listing, PER_CLASS)]
    missing = sum(not r['path'].is_file() for r in out)
    if len(listing) != DISTINCT or missing:
        raise RuntimeError(f'pcam: {len(listing)} distinct patches listed ({DISTINCT} expected) and {missing} '
                           f'presampled patches missing in {folder}; delete it to download and convert again')
    return out, common.shares(listing)


def _listing(raw):
    """The distinct patches, from raw/listing.csv, which is written once from y and the meta file."""
    path = raw / LISTING
    if not path.exists():
        meta = fetch.download(f'{FILES}/{META[0]}?download=1', raw / META[0], size=META[1], md5=META[2])
        with _h5(_unpacked(raw, Y)) as f:
            y = f['y'][:].reshape(-1)
        _write(meta, y, path)
    return _read(path)


def _unpacked(raw, spec):
    """<name>.h5.gz -> <name>.h5, once: downloaded and checked, unpacked, then the .gz is deleted."""
    name, size, md5 = spec
    h5 = raw / name[:-3]
    if not h5.exists():
        gz = fetch.download(f'{FILES}/{name}?download=1', raw / name, size=size, md5=md5)
        part = h5.with_suffix('.part')
        with gzip.open(gz, 'rb') as src, open(part, 'wb') as dst:
            shutil.copyfileobj(src, dst, 1 << 20)
        part.rename(h5)
        gz.unlink()
    return h5


def _h5(path):
    """An h5 file, read-only. h5py is imported here: only the one-time conversion reads h5 files."""
    try:
        import h5py
    except ImportError:
        raise RuntimeError('pcam: h5py is not importable; install it where python -I finds it (not ~/.local), e.g. in '
                           'a venv made with --system-site-packages') from None
    return h5py.File(path, 'r')


def _write(meta, y, path):
    """listing.csv: one row per distinct patch (the first row of its slide and coordinates), each checked against y."""
    with open(meta, newline='') as f:
        listed = list(csv.DictReader(f))
    if len(listed) != ROWS or y.shape != (ROWS,):
        raise RuntimeError(f'pcam: {len(listed)} meta rows and {y.shape} labels, {ROWS} expected; delete {path.parent} '
                           f'to download again')
    out, seen = [], set()
    for i, r in enumerate(listed):
        tumour = {'True': 1, 'False': 0}.get(r['center_tumor_patch'])
        if r[''] != str(i) or tumour != y[i] or not SLIDE.fullmatch(r['wsi']):
            raise RuntimeError(f'pcam: meta row {i} ({r}) disagrees with y or is not in the expected format; delete '
                               f'{path.parent} to download again')
        key = (r['wsi'], r['coord_y'], r['coord_x'])
        if key not in seen:
            seen.add(key)
            out.append([i, 'nok' if tumour else 'ok', r['wsi']])
    part = path.with_suffix('.part')
    with open(part, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['row', 'label', 'slide'])
        w.writerows(out)
    part.rename(path)


def _read(path):
    with open(path, newline='') as f:
        return [{'row': int(r['row']), 'label': r['label'], 'slide': r['slide']} for r in csv.DictReader(f)]


def _convert(h5, chosen, patches):
    """The chosen patches -> patches/<label>/<row>.png, then a marker; the h5 file is deleted after."""
    with _h5(h5) as f:
        x = f['x']
        if x.shape != (ROWS, 96, 96, 3) or x.dtype != 'uint8':
            raise RuntimeError(f'pcam: x is {x.shape} {x.dtype}, ({ROWS}, 96, 96, 3) uint8 expected; delete {h5} to '
                               f'download it again')
        for r in chosen:
            path = patches / r['label'] / f"{r['row']:05d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(x[r['row']]).save(path)
    (patches / '.converted').write_text(f'{len(chosen)} patches from {h5.name}\n')
    h5.unlink()
