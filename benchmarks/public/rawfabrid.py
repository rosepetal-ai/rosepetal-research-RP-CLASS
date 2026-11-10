# RP-CLASS - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""RAW-FABRID (docs/BENCHMARK.md, I4): OK / NOK on 256 x 256 patches of raw fabric from a line-scan camera, one
task: the in-line textile inspection of the benchmark.

Data: RAW_FABRID.zip, the one file of Mendeley Data record db6g85xsyg v1 (Pérez-Llorens, Albero-Albero and
Silvestre-Blanes, Universitat Politècnica de València; published 2026-04-01, no article yet), 4,133,046,386 bytes,
with the SHA-256 the record publishes. 709 frames of 1,792 x 1,024 px (a Basler line-scan camera at 4 px/mm, light
from above and below) of 7 rolls of fabric (Rollo1A ... Rollo7A, the anonymised origin), 202 with defects (masks,
COCO boxes) and 507 without. Each frame gives 28 patches of 256 x 256, grey 8-bit PNG, one per cell of its 4 x 7 grid,
in an MVTec layout: train/good 14,196 (the 507 clean frames), test/good 4,969 and test/defect 687 (the 202 defective
frames). A clean cell is the exact cell; a defect cell is a window re-centred on its defect, clipped to the frame (15
of the 18 of Rollo4A 90-93 sit off the grid, e.g. y 484, x 419 for cell 2_2). RAW_FABRID_Patches_Metadata.csv gives
every patch its folder and its source frame.
Licence: CC BY 4.0, the record's data_licence field.

Classes:
    ok     the 19,165 good patches: 14,196 of the 507 clean frames (MVTec/train/good), 4,969 of the 202 defective
           ones (MVTec/test/good)
    nok    the 687 defect patches (MVTec/test/defect), 1 to 14 per defective frame

Choices:
    test      built, as the authors' split is the one-class layout (only good patches to train on) that the
              benchmark pools and re-splits (docs/BENCHMARK.md, Protocol): 0.4 of each class, at most 500
    groups    the source frame, both classes: its patches never split (a nok patch shares pixels with its good
              neighbours: it is re-centred, they are exact cells). A frame also joins the next one of its roll when
              one defect runs on across their border (a COCO box reaching the last row of one frame overlaps in x a
              box starting on the first row of the next): 11 links, 9 groups of 2 to 4 frames, e.g. Rollo4A 90-93,
              one long fault. 698 groups in all
    strata    the frame's kind, clean or defective: the frames drawn for their nok patches bring many good ones, and
              without it the test's ok would come mostly from next to a defect (74 % of the ok patches are of clean
              frames, and so are 74 % of the test's)
    tiny      7 defect patches hold under 50 px of defect (1, 2, 2, 8, 10, 35 and 38 px; the median is 2,801 px),
              kept as the authors label them
    disk      only the two patch folders and the two metadata files are extracted (0.87 GB), then the zip is deleted:
              5 GB at the peak

Splits (common.py), prepared on the GPU machine, 2026-11-06: 19,852 patches listed; no exact copy, near-copy or
featureless patch; 12 look-alikes grouped, which join a few distinct frames (of Rollo6A, in the manifest): the largest
group is 5 frames, 140 patches. Test 500 ok + 275 nok from 95 frames (80 defective, 15 clean), val 97 ok + 5 nok from 5
frames, pool 16,653 ok + 399 nok, a training draw 290 ok + 20 nok per seed (share ok 0.965). The draws take whole
groups, then exactly these counts: the other 1,885 patches of the test's frames and 38 of val's are left out.

Caveats: a nok patch mostly has its defect near its centre, while an ok patch is a grid cell. The rolls differ in
defect rate, from 1.9 % of the patches (Rollo6A, Rollo7A) to 10.6 % (Rollo3A), and Rollo4A holds 269 of the 687 nok,
so a roll's weave is a weak cue for the class. Consecutive frames of a roll are adjacent stretches of one fabric. The
labels are the authors': a thin dark fibre lies on at least one ok patch (Rollo4A frame 6, cell 0_6), much like the
one marked as a defect on Rollo1A frame 1, cell 0_1. The full-resolution frames are RGB PNGs with three equal
channels (the record says grey); they are not used. The record is v1, with no paper yet.
"""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import fetch

NAME = 'rawfabrid'
TASKS = (NAME,)
IMGSZ = 224
LICENCE = 'CC BY 4.0'
SOURCE = 'https://data.mendeley.com/public-api/datasets/db6g85xsyg?version=1'
URL = ('https://data.mendeley.com/public-files/datasets/db6g85xsyg/files/93e17d48-b5e6-45f7-9064-be84cc5fb91c/'
       'file_downloaded')
SIZE, SHA256 = 4133046386, 'f6b4f82cce2f0220bbab3968a74b5337ebf2bbfd5e33d0548d34e1ebf80f85b9'
PATCHES, COCO = 'RAW_FABRID_Patches_Metadata.csv', 'RAW_FABRID_HighRes_COCO.json'
CLASSES = {'train_good': ('MVTec/train/good', 'ok'), 'test_good': ('MVTec/test/good', 'ok'),
           'test_defect': ('MVTec/test/defect', 'nok')}
COUNT = {'nok': 687, 'ok': 19165}
POSITIVE = 'nok'


def download(raw):
    """RAW_FABRID.zip -> raw/patches (the ok and nok patch folders, the patch list and the COCO boxes), once; the zip
    is deleted when they are out."""
    out = Path(raw) / 'patches'
    if not (out / '.extracted').exists():
        archive = fetch.download(URL, Path(raw) / 'RAW_FABRID.zip', size=SIZE, sha256=SHA256)
        fetch.extract(archive, out, keep=_wanted)
        archive.unlink()
    return out / 'RAW_FABRID'


def _wanted(name):
    return (name in (f'RAW_FABRID/{PATCHES}', f'RAW_FABRID/{COCO}')
            or any(name.startswith(f'RAW_FABRID/{sub}/') for sub, _ in CLASSES.values()))


def rows(folder, task):
    folder = Path(folder)
    group = _groups(json.loads((folder / COCO).read_text()))
    out, bad = [], []
    with open(folder / PATCHES, newline='') as f:
        for r in csv.DictReader(f):
            if r['type'] not in CLASSES:
                continue
            sub, label = CLASSES[r['type']]
            path = folder / sub / r['patch_name']
            if path.parent != folder / sub or not path.is_file() or r['parent_raw'] not in group:
                bad.append(r['patch_name'])
                continue
            out.append({'path': path, 'label': label, 'group': group[r['parent_raw']],
                        'stratum': 'clean' if r['type'] == 'train_good' else 'defective'})
    count = Counter(r['label'] for r in out)
    if bad or count != COUNT:
        raise RuntimeError(f'rawfabrid: {dict(count)} patches found in {folder}, {COUNT} expected, {len(bad)} listed '
                           f'without a file or a source frame (e.g. {bad[:1]}); delete {folder.parent} to download '
                           f'it again')
    return out


def _groups(coco):
    """{frame file name: its group}: the frame itself, or the first frame of the run of frames one defect crosses."""
    boxes = defaultdict(list)
    for a in coco['annotations']:
        boxes[a['image_id']].append(a['bbox'])
    frames = {}
    for im in coco['images']:
        roll, number = im['file_name'].split('_')[:2]
        frames[(roll, int(number))] = (im['file_name'], boxes[im['id']], im['height'])
    out = {}
    for (roll, number), (name, below, _) in sorted(frames.items()):
        prev = frames.get((roll, number - 1))
        out[name] = out[prev[0]] if prev and _continues(prev[1], prev[2], below) else name
    return out


def _continues(above, height, below):
    """A box [x, y, w, h] reaching the last row of the frame above overlaps in x one starting on the first row
    below."""
    return any(xa < xb + wb and xb < xa + wa
               for xa, ya, wa, ha in above if ya + ha >= height - 1
               for xb, yb, wb, hb in below if yb <= 0)
