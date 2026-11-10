# RP-CLASS - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""CLEVR count_all, as in VTAB: the number of objects in a rendered scene, 3 to 10, as 8 classes n03 ... n10 (zero-
padded, so that sorted order is numeric order). Counting stands in for completeness checks: are all the parts there?

Data: CLEVR v1.0 (Johnson et al., CVPR 2017), one zip of 19,021,600,724 bytes. The host publishes no SHA-256, only
an MD5 in its S3 metadata (b11922020e72d0cd9154779b2d3d07d2), which fetch checks with the pinned size; zipfile
checks the CRC-32 of every member read or extracted. 70,000 train, 15,000 val and 15,000 test renders, PNG,
RGBA, 480 x 320 (2,000 of them read: alpha 255 everywhere, which both loaders drop), and the scene graphs of train
and val in CLEVR_v1.0/scenes/CLEVR_{train,val}_scenes.json. The label is len(scene['objects']): VTAB's 'number of
objects minus 3', named by the count itself. The official test renders have no scene graph and are not used.
Licence: CC BY 4.0, 'All data is released under the Creative Commons CC BY 4.0 license.' (SOURCE); the zip's
COPYRIGHT.txt: 'CLEVR (c) 2017, Facebook, Inc. CLEVR is licensed under a Creative Commons Attribution 4.0
International License.' Cite Johnson et al. 2017.

Splits:
    test   the official val split whole, as VTAB uses it: 15,000 images, n03 1,898 · n04 1,925 · n05 1,843 ·
           n06 1,852 · n07 1,808 · n08 1,941 · n09 1,877 · n10 1,856.
    pool   the official train split, 70,000 scenes (n07 8,592 to n10 8,913), presampled to PER_CLASS = 400 of each
           class before anything is extracted or hashed. Val takes 12 or 13 of each class and each seed's draw 37
           or 38, so the draws still come from about ten times their size, and 3,200 images are extracted and hashed
           instead of 70,000 (0.6 GB instead of 13.2). share: the class shares of all 70,000 (12.3 % to 12.7 %).
Prepared on the GPU machine, 2026-11-06: 18,200 listed (the 15,000 and the 3,200), no exact copy, look-alike,
near-copy or featureless image; test 15,000, val 99, train 300 per seed of the 3,101 left (387 or 388 a class).

Disk: the zip is downloaded whole, in one request; the scene graphs are read from it, and only they, the 15,000
test and the 3,200 presampled renders are extracted (3.6 GB), then the zip is deleted. Peak about 22.7 GB.

Caveats: synthetic renders, a weak stand-in for inspection images. At 224 the long side shrinks 2.1 times, and
the smallest objects with it. Every scene has the same floor, lights and kinds of object, yet no two of the 18,200
used are look-alikes (on 2,000 renders read first, the 7 pairs at a dHash distance of 4 differ by 5.9 or more at
32 x 32, where the limit is 3), so no group joins distinct scenes.
"""
import json
import re
import zipfile
from pathlib import Path

import common
import fetch

NAME = 'clevr_count'
TASKS = (NAME,)
IMGSZ = 224
LICENCE = 'CC BY 4.0'
SOURCE = 'https://cs.stanford.edu/people/jcjohns/clevr/'
URL = 'https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip'
SIZE = 19_021_600_724
MD5 = 'b11922020e72d0cd9154779b2d3d07d2'             # the host's x-amz-meta-s3cmd-attrs (2026-11-06)
PER_CLASS = 400
SCENES = {'val': 'CLEVR_v1.0/scenes/CLEVR_val_scenes.json', 'train': 'CLEVR_v1.0/scenes/CLEVR_train_scenes.json'}


def _scenes(read, official):
    """The rows of one official split, from its scene graphs; paths are the zip's member names."""
    out = []
    for scene in json.loads(read(SCENES[official]))['scenes']:
        name = scene['image_filename']
        if not re.fullmatch(rf'CLEVR_{official}_\d{{6}}\.png', name):
            raise RuntimeError(f'clevr_count: unexpected image name {name!r} in the {official} scene graphs; delete '
                               f'the download and prepare again')
        out.append({'path': f'CLEVR_v1.0/images/{official}/{name}', 'label': f"n{len(scene['objects']):02d}",
                    'split': 'test' if official == 'val' else 'train'})
    return out


def _select(read):
    """read(member name) -> bytes. The rows used (official val whole, official train presampled) and the class
    shares of the whole official train split."""
    train = _scenes(read, 'train')
    return _scenes(read, 'val') + common.presample(train, PER_CLASS), common.shares(train)


def download(raw):
    """The zip, then only what rows() uses; the zip is deleted once that is out."""
    raw = Path(raw)
    archive = raw / 'CLEVR_v1.0.zip'
    if not (raw / '.extracted').exists():
        fetch.download(URL, archive, size=SIZE, md5=MD5)
        with zipfile.ZipFile(archive) as z:
            used, _ = _select(z.read)
        keep = {r['path'] for r in used} | set(SCENES.values())
        fetch.extract(archive, raw, keep=lambda name: name in keep)
    archive.unlink(missing_ok=True)
    return raw


def rows(folder, task):
    folder = Path(folder)
    used, share = _select(lambda name: (folder / name).read_bytes())
    used = [dict(r, path=folder / r['path']) for r in used]
    missing = [r['path'] for r in used if not r['path'].is_file()]
    if missing:
        raise RuntimeError(f'clevr_count: {len(missing)} images missing, e.g. {missing[0]}; delete {folder} and '
                           f'prepare again')
    return used, share
