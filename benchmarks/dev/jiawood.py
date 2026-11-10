# RP-CLASS - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""Development suite (never part of the benchmark): water-based coated wood (Jia et al.), the defect type of a box
crop, 4 types, no OK class, one task.

Data: OriginalDataset.zip of Zenodo record 15679291 (Jia, Chen, Liu and Wang, 'A Public Image Dataset for Surface
Defect Detection of Water-Based Coated Wood Products', v1.0 of 2025-06-17; the article is Scientific Data 13:128,
2026, doi:10.1038/s41597-025-06443-1), 19,217,292 bytes; its MD5 matched the one Zenodo publishes on 2026-11-06. It
holds OriginalDataset/Images/000001.jpg ... 000335.jpg, 600 x 600 RGB JPEGs, a YOLO label file for each in Labels/
('<id> <x centre> <y centre> <width> <height>', fractions of the image) and classes.txt, which names the ids 0-3
hole, blister, crack, scratch (the article's holes, bubbles, cracks and scratches). 829 boxes: hole 188, blister 190,
crack 225, scratch 226; 1 to 11 per image; 262 images hold one type, 71 two, 2 three. The record's eight other zips
are augmented copies of these 335 (noise, brightness, mirroring) and are never used: the article's random 7:2:1 split
of all of them puts copies of one image on both sides.
Licence: CC BY 4.0, the record's licence (cc-by-4.0). The article is CC BY-NC-ND 4.0, which covers the article only;
it calls the data 'openly available to academic and industrial users worldwide through the Zenodo platform'.

The article: panels of six substrates (finger-jointed rubberwood, plywood, MDF, LVL, finger-jointed board, OSB) in
seven colours, painted at a door factory in Jiangshan and filmed on a conveyor from above by a 2,448 x 2,048 px
area-scan camera under diffuse D65 light; 'Image sequences were captured frame-by-frame (video acquisition)', 335
frames were kept by hand, and a region of each was cut and resized to 600 x 600.

Crops, by the rules of benchmarks/public/wood.py:
    margin      25 % of the box's width added left and right, 25 % of its height above and below, clipped to the image
    too small   a box whose shorter side is under 24 px is dropped: 345 of the 829 (hole 137: the pinholes' median is
                16 px; blister 89, crack 93, scratch 26)
    overlap     a crop that overlaps a box of another type, of any size, is dropped: 18 more (scratch 9, crack 5,
                blister 2, hole 2)
    types       a type with under MIN_CROPS = 30 crops would be dropped from every split, and noted; none is
Crops kept: scratch 191, crack 127, blister 99, hole 49; 466 from 266 images, written once as PNG (the zip is then
deleted), with a shorter side of 36 to 600 px (medians: hole 43, blister 47, crack 66, scratch 117), so most are
upsampled to IMGSZ 224.

Groups: the source image, but frames of one surface are joined, as a frame is no independent sample here: one defect
is often boxed in 2 to 10 consecutive frames, shifted or turned (288-291: one crack in four frames). Found by eye on
all 335 images, then at 300 px where in doubt: the runs of RUNS, the white door filmed in 187-211 (its mitre cracks
recur) among them, and pairs apart, 258 and 264 (one blister cluster). The review found eight more runs from the
nearest crop across splits over the 8 orientations (the same scratch in 35-36, 45-46, 60-61, 66-68, 72-73, 78-79, 85-86,
143-144, some turned 90 degrees) and three doubtful pairs (1 and 3, 17 and 23, 65 and 71); the doubtful ones are
joined anyway, as finalize joins look-alikes. finalize's own look-alike test joins only 3 pairs of these crops (a
defect turned or seen at another scale is no look-alike). Grouped by image alone (common.py run on these crops), 10 of
the 16 repeats sure at first sight straddle test, val and train. No rule replaces the list. The step from a frame to
the next, the largest change of a channel's share of the frame's median colour, in percentage points, has a median
of 1.8; it reaches 3.01
inside a sure repeat (146-147), and the other steps near it (2.93, 2.97, 3.01, 3.03, 3.05, 3.06) show no gap. Joining
frames whose step is under a threshold swings the splits: at 3 points no hole is left to train on (the task cannot be
built), at 4 the pool is 128 crops, at 5 it is 75.

Counts (prepare.py, 2026-11-06): no copies, 3 look-alike pairs, 18 crops featureless at 32 x 32 (faint defects on
plain paint). Test 187: blister 40, crack 51, hole 20, scratch 76; val 100: 21, 27, 11, 41; pool 152: 35, 33, 14, 70,
and every training draw is the whole pool (each class is under its share of 300), so the three seeds differ only by
the run seed. 27 crops are left out with the groups the test and val drew. A small task: it can tell a large effect
from none, not a point from another. Every
label agrees with its YOLO line, and every crop with a fresh crop of the original.

Caveats: type and board go together: the red panels of 51-104 hold 77 scratch crops and no other, and 19 of the 49
holes lie on the textured green surface of 105-119, so colour and substrate are cues, and boards (not defects) cross
the splits. Each frame is a region resized by an unknown factor, so a crop's pixel size is not the defect's size. The
labels are the authors': large round holes that look drilled are not boxed (14, 30, 219) while black blobs are boxed
as holes (228, 239), and the door panels' cracks all run along mitre joints (187-211, 297), a context cue. Repeats
that were not seen may remain.
"""
import csv
import io
import re
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image

import fetch

NAME = 'jiawood'
TASKS = (NAME,)
IMGSZ = 224
LICENCE = 'CC BY 4.0'
SOURCE = 'https://zenodo.org/api/records/15679291'

URL = 'https://zenodo.org/records/15679291/files/OriginalDataset.zip?download=1'
ZIP, SIZE, MD5 = 'OriginalDataset.zip', 19217292, 'df663cbf553a63e3ec5b372bc6ca3d3c'     # MD5, as Zenodo gives
TOP, IMAGES, INDEX = 'OriginalDataset', 335, 'crops.csv'
CLASSES = ('hole', 'blister', 'crack', 'scratch')             # classes.txt: the YOLO ids 0 to 3
CROPS = {'blister': 99, 'crack': 127, 'hole': 49, 'scratch': 191}
MARGIN, MIN_SIDE, MIN_CROPS = 0.25, 24, 30
RUNS = ((13, 14), (17, 18), (27, 28), (29, 30), (35, 36), (40, 41), (45, 46), (47, 50), (51, 52), (53, 54), (55, 59),
        (60, 61), (66, 68), (72, 73), (78, 79), (83, 84), (85, 86), (95, 96), (118, 119), (129, 130), (140, 141),
        (143, 144), (146, 148), (187, 211), (213, 214), (224, 226), (288, 291), (295, 296), (304, 305))
#      (first, last): consecutive frames of one surface; the review added 35-36, 45-46, 60-61, 66, 72-73, 78-79,
#      85-86 and 143-144 (the same scratch in two frames, some turned 90 degrees) and joined 55-56 with 57-59
PAIRS = ((258, 264), (1, 3), (17, 23), (65, 71))     # the same surface, not consecutive (the last three doubtful)
NOTES = {'groups': 'the source image; frames of one surface joined by eye: '
                   + ', '.join(f'{a}-{b}' for a, b in RUNS) + ', ' + ', '.join(f'{a} and {b}' for a, b in PAIRS)}


def download(raw):
    """OriginalDataset.zip -> raw/crops/<class>/<image>_<box>.png and their list raw/crops.csv, once; the zip is
    deleted when its crops are out."""
    raw = Path(raw)
    if not (raw / INDEX).exists():
        _crop(fetch.download(URL, raw / ZIP, size=SIZE, md5=MD5), raw)
    (raw / ZIP).unlink(missing_ok=True)
    return raw


def rows(folder, task):
    out = _crops(folder)
    small = _small(out)
    if small:
        print(f'{task}: dropped, under {MIN_CROPS} crops: ' + ', '.join(f'{c} ({n})' for c, n in small.items()),
              flush=True)
    return [r for r in out if r['label'] not in small]


def notes(folder, task):
    """The types dropped for having under MIN_CROPS crops (none on 2026-11-06)."""
    return {'dropped_types': {c: f'{n} crops, under {MIN_CROPS}' for c, n in _small(_crops(folder)).items()}}


def _crops(folder):
    folder = Path(folder)
    with open(folder / INDEX, newline='') as f:
        out = [{'path': folder / r['file'], 'label': r['label'], 'group': str(_group(int(r['image'])))}
               for r in csv.DictReader(f)]
    count = dict(Counter(r['label'] for r in out))
    if count != CROPS:
        raise RuntimeError(f'jiawood: crops {count} in {folder / INDEX}, {CROPS} expected; delete {folder} to crop '
                           f'again')
    return out


def _small(crops):
    count = Counter(r['label'] for r in crops)
    return {c: count[c] for c in sorted(count) if count[c] < MIN_CROPS}


def _group(image):
    """The first image of the run or pair an image is joined to, else the image itself."""
    for first, last in RUNS:
        if first <= image <= last:
            return first
    for first, other in PAIRS:
        if image == other:
            return first
    return image


def _crop(archive, raw):
    """The crops the rules keep, of every image of the zip -> raw/crops/<class>/<image>_<box>.png; crops.csv, their
    list, is written last: it marks the zip as done."""
    listed = []
    with zipfile.ZipFile(archive) as z:
        names = set(z.namelist())
        classes = tuple(z.read(f'{TOP}/classes.txt').decode('ascii').split())
        images = sorted(n for n in names if re.fullmatch(rf'{TOP}/Images/\d{{6}}\.jpg', n))
        if classes != CLASSES or len(images) != IMAGES:
            raise RuntimeError(f'jiawood: classes {classes} and {len(images)} images in {archive.name}, {CLASSES} and '
                               f'{IMAGES} expected; check the archive')
        for name in images:
            image = int(name[-10:-4])
            label = f'{TOP}/Labels/{image:06d}.txt'
            if label not in names:
                raise RuntimeError(f'jiawood: {name} has no {label}; check the archive')
            with Image.open(io.BytesIO(z.read(name))) as src:
                im = src.convert('RGB')
            for k, kind, box in _kept(_boxes(z.read(label).decode('ascii'), label), im.size):
                path = Path('crops') / kind / f'{image:06d}_{k:02d}.png'
                (raw / path).parent.mkdir(parents=True, exist_ok=True)
                im.crop(box).save(raw / path)
                listed.append([path.as_posix(), kind, image, *box])
    part = raw / f'{INDEX}.part'
    with open(part, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['file', 'label', 'image', 'left', 'top', 'right', 'bottom'])
        w.writerows(listed)
    part.rename(raw / INDEX)
    print(f'  {archive.name}: {len(listed)} crops from {len({r[2] for r in listed})} images', flush=True)


def _boxes(text, where):
    """[(class id, x centre, y centre, width, height)], fractions of the image: one YOLO label file."""
    out = []
    for line in text.splitlines():
        v = line.split()
        if not v:
            continue
        if len(v) != 5 or v[0] not in [str(i) for i in range(len(CLASSES))]:
            raise RuntimeError(f'jiawood: {where}: {line!r} is not a YOLO box line; check the archive')
        out.append((int(v[0]), *(float(x) for x in v[1:])))
    return out


def _kept(boxes, size):
    """[(box index, class, (left, top, right, bottom) in pixels)]: the crops of one image that the rules keep."""
    width, height = size
    px = [(c, (x - w / 2) * width, (y - h / 2) * height, (x + w / 2) * width, (y + h / 2) * height)
          for c, x, y, w, h in boxes]
    out = []
    for k, (c, x0, y0, x1, y1) in enumerate(px):
        w, h = x1 - x0, y1 - y0
        if min(w, h) < MIN_SIDE:
            continue
        crop = (max(0, round(x0 - MARGIN * w)), max(0, round(y0 - MARGIN * h)),
                min(width, round(x1 + MARGIN * w)), min(height, round(y1 + MARGIN * h)))
        if not any(other != c and _overlap(crop, b) for other, *b in px):
            out.append((k, CLASSES[c], crop))
    return out


def _overlap(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])
