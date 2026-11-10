# RP-CLASS - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""Wood surface defects (VSB-TU Ostrava; docs/BENCHMARK.md, I5): 8 defect types, one crop per labelled box, no OK
class.

Source: Zenodo record 4694695 (Kodytek, Bodzas and Bilik, F1000Research 10:581, 2021), CC BY 4.0 (the record's
licence id cc-by-4.0, and the paper). 20,276 colour BMPs of sawn timber taken on a sawmill line by a line-scan
camera, 2,800 x 1,024 (41 of the 4,000 used here are 2,708-2,797 px wide), in Images1.zip ... Images10.zip of
15.3-17.6 GB each; 1,992 of them show no defect. The 43,974 boxes of 10 types are in Bouding_Boxes.zip (sic): one
text file per image, empty for a clean one, one line per box, '<type>\t<left>\t<top>\t<right>\t<bottom>', fractions
of the image's own width and height. The paper once writes 'left, top, bottom, right', but the values settle it: the
2nd and 4th times 1,024 are whole pixels on all 43,974 lines, the 3rd only on 170 of the 9,211 used (it is a width
fraction). 132 files write the decimals with a comma.

Used: Images1.zip (15,282,458,754 bytes) and Images2.zip (15,347,429,322 bytes), this benchmark's choice: 4,000
images with 9,211 boxes, 30.6 GB to download. The crops are cut here, one zip at a time (download it, crop, delete
it), so the disk holds one zip and the crops at most, about 16 GB. Zenodo publishes MD5 sums only: fetch checks them
and the pinned sizes, the annotation zip's SHA-256 is pinned too (from a download whose MD5 matched Zenodo's), and
zipfile checks the CRC-32 of every image it reads.

The crop rules (docs/BENCHMARK.md, Protocol), with what they do on Images1-2:
    margin      25 % of the box's width added left and right, 25 % of its height above and below, clipped to the
                image: padding would put a black band on the crops of the boxes at the image edges, mostly long
                cracks, quartzity and marrow, a cue that is not in the wood
    too small   a box whose shorter side is under 24 px is dropped: 543 of the 9,211, mostly specks under 12 px
                that the labellers' painting tool left behind
    overlap     a crop that overlaps a box of another type, of any size, is dropped: 631 more
    types       Blue_Stain (96 boxes in all) and overgrown (10) are dropped, this benchmark's choice (Images1-2 hold
                none); a type with under 50 crops would be dropped too, and printed (none is: the smallest has 95)
Crops, prepared on the GPU machine, 2026-11-06: live_knot 3,681, dead_knot 2,666, resin 577, crack 409,
knot_with_crack 360, marrow 151, knot_missing 98, quartzity 95; 8,037 in all, about 0.6 GB of PNG. None is an exact
copy, a look-alike, a near-copy or featureless. The test is built (common.py: 40 % of each class, at most 500, drawn
by board): 1,676 crops from 106 boards, 13 (marrow) to 89 (dead_knot) per class; val 111 from 6 boards; pool 4,153;
a training draw 359 per seed. The draws take whole boards (the largest holds 111 crops), then exactly these counts:
the other 1,840 crops of the test's boards and 257 of val's are left out.

Groups: the board. An image is a block of 1,024 lines (15 cm) of the timber passing under the camera, numbered
capture cycle x 100,000 + its index in the cycle (0-84; the authors filtered out the blocks without wood). A run of
consecutive numbers is therefore one board: the runs are 1 to 32 blocks long, and 32 blocks are the 5 m of a board.
A defect across two blocks is boxed in both (in Images1-2, 439 boxes at the bottom of a block go on at the top of the
next with the same type), so an image is no group. Images1-2 hold 239 boards; a board holds crops of several
types, and all go to one split.

Caveats: one trained person painted the zones the boxes come from; the pixels are not square (16.66 px/mm across
the board, 6.67 along it), so a round knot is 2.5 times wider than tall; the crops range from 30 to 1,788 px, and
their shape is a cue (quartzity, marrow and cracks run along the board, knots are wide), as in any defect crop;
Images1-2 are the first 81 of the 407 capture cycles (991-1156), a fifth of the four-hour session; 17 of the 9,211
boxes are no whole pixels of their image's width (3 are fractions of 2,800 on a narrower image, 6-12 px off; 14 fit
no width), a small error of unknown size.
"""
import csv
import io
import re
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image

import fetch

NAME = 'wood'
TASKS = (NAME,)
IMGSZ = 224
LICENCE = 'CC BY 4.0'
SOURCE = 'https://zenodo.org/api/records/4694695'

FILES = 'https://zenodo.org/records/4694695/files'
BOXES = ('Bouding_Boxes.zip', 5002318, '19cd3cf53ac95625558e365a3482524add3e1dcce4a03aaf62dd0447a49ab814')
IMAGES = (('Images1.zip', 15282458754, 'cba808d221d78c46e79958efd4785bca'),      # MD5, as Zenodo gives
          ('Images2.zip', 15347429322, '9ebc7082feda068b76fd13b1aed02f78'))
NOTES = {'dropped_types': {'Blue_Stain': 'by the proposal (no box in Images1-2)', 'overgrown': 'by the proposal (no '
                           'box in Images1-2)'}}
CLASSES = {'Live_Knot': 'live_knot', 'Dead_Knot': 'dead_knot', 'resin': 'resin', 'Crack': 'crack',
           'knot_with_crack': 'knot_with_crack', 'Marrow': 'marrow', 'Knot_missing': 'knot_missing',
           'Quartzity': 'quartzity'}                         # Blue_Stain and overgrown are not cropped
MARGIN, MIN_SIDE, MIN_CROPS = 0.25, 24, 50
MAX_MEMBER = 1 << 26                                         # a 2,800 x 1,024 BMP is 8.6 MB


def download(raw):
    raw = Path(raw)
    boxes = _boxes(fetch.download(f'{FILES}/{BOXES[0]}?download=1', raw / BOXES[0], size=BOXES[1],
                                  sha256=BOXES[2]))
    boards = _boards(boxes)
    for name, size, md5 in IMAGES:
        index = raw / 'crops' / f'{Path(name).stem}.csv'
        if not index.exists():
            _crop(fetch.download(f'{FILES}/{name}?download=1', raw / name, size=size, md5=md5), boxes, boards, raw,
                  index)
        (raw / name).unlink(missing_ok=True)
    return raw


def rows(folder, task):
    out = _listed(folder)
    small = _small(out)
    if small:
        print(f'{task}: dropped, under {MIN_CROPS} crops: ' + ', '.join(f'{c} ({n})' for c, n in small.items()),
              flush=True)
    return [r for r in out if r['label'] not in small]


def notes(folder, task):
    """What is dropped: the two types this benchmark leaves out, and any type under MIN_CROPS crops."""
    small = {c: f'{n} crops, under {MIN_CROPS}' for c, n in _small(_listed(folder)).items()}
    return {'dropped_types': {**NOTES['dropped_types'], **small}}


def _listed(folder):
    """The crops written by download(), as rows."""
    folder, out = Path(folder), []
    for name, _, _ in IMAGES:
        with open(folder / 'crops' / f'{Path(name).stem}.csv', newline='') as f:
            out += [{'path': folder / r['file'], 'label': r['label'], 'group': str(r['board'])}
                    for r in csv.DictReader(f)]
    return out


def _small(crops):
    count = Counter(r['label'] for r in crops)
    return {c: count[c] for c in sorted(count) if count[c] < MIN_CROPS}


def _boxes(archive):
    """{image number: [(type, left, top, right, bottom)]}, fractions of the image size; [] for a clean image."""
    out = {}
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            m = re.fullmatch(r'Bouding Boxes/(\d+)_anno\.txt', info.filename)
            if m:
                lines = z.read(info).decode('ascii').splitlines()
                out[int(m.group(1))] = [_box(line, info.filename) for line in lines if line.strip()]
    return out


def _box(line, where):
    kind, *v = line.split('\t')
    if len(v) != 4:
        raise RuntimeError(f'{where}: {line!r} is not a box line; check the annotation zip')
    return (kind, *(float(x.replace(',', '.')) for x in v))


def _boards(images):
    """{image number: its board, the first image of its run of consecutive numbers}. An index in a cycle never
    passes 84, so a run never crosses two cycles."""
    out, first, prev = {}, None, None
    for n in sorted(images):
        if prev is None or n != prev + 1:
            first = n
        out[n], prev = first, n
    return out


def _crop(archive, boxes, boards, raw, index):
    """The crops of every image of one zip -> raw/crops/<class>/<image>_<box>.png; index, their list, is written
    last: it marks the zip as done."""
    listed, seen = [], set()
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            m = re.fullmatch(r'Images\d+/(\d+)\.bmp', info.filename)
            image = int(m.group(1)) if m else None
            if not boxes.get(image) or image in seen:          # no box (a clean image), or listed twice
                continue
            if info.file_size > MAX_MEMBER:
                raise RuntimeError(f'{archive.name}: {info.filename} has {info.file_size} bytes, not a 2,800 x 1,024 '
                                   'BMP; check the archive')
            seen.add(image)
            with Image.open(io.BytesIO(z.read(info))) as src:
                im = src.convert('RGB')
            for k, label, box in _crops(boxes[image], im.size):
                path = Path('crops') / label / f'{image}_{k:02d}.png'
                (raw / path).parent.mkdir(parents=True, exist_ok=True)
                im.crop(box).save(raw / path)
                listed.append([path.as_posix(), label, image, boards[image], *box])
    part = index.with_suffix('.part')
    with open(part, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['file', 'label', 'image', 'board', 'left', 'top', 'right', 'bottom'])
        w.writerows(listed)
    part.rename(index)
    print(f'  {archive.name}: {len(listed)} crops from {len(seen)} images', flush=True)


def _crops(boxes, size):
    """[(box index, class, (left, top, right, bottom) in pixels)]: the crops of one image that the rules keep."""
    width, height = size
    px = [(kind, x0 * width, y0 * height, x1 * width, y1 * height) for kind, x0, y0, x1, y1 in boxes]
    out = []
    for k, (kind, x0, y0, x1, y1) in enumerate(px):
        w, h = x1 - x0, y1 - y0
        if kind not in CLASSES or min(w, h) < MIN_SIDE:
            continue
        crop = (max(0, round(x0 - MARGIN * w)), max(0, round(y0 - MARGIN * h)),
                min(width, round(x1 + MARGIN * w)), min(height, round(y1 + MARGIN * h)))
        if not any(other != kind and _overlap(crop, b) for other, *b in px):
            out.append((k, CLASSES[kind], crop))
    return out


def _overlap(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])
