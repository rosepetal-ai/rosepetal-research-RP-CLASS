# RP-CLASS - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""SDNET2018 (Utah State University; docs/BENCHMARK.md, I7): cracked (nok) / uncracked (ok) concrete, pooled over
bridge decks, walls and pavements, the structure as stratum, grouped by parent photo.

Source: Maguire, Dorafshan and Thomas, SDNET2018 (2018; Data in Brief 21:1664), https://digitalcommons.usu.edu/
all_datasets/48/, which states 'This work is licensed under a Creative Commons Attribution 4.0 License.' 230 photos
(54 decks, 72 walls, 104 pavements) from a 16 MP Nikon, cut into 56,092 tiles of 256 x 256, each labelled C (a crack
in it) or U. Cracks from 0.06 to 25 mm, among shadows, roughness, scaling, edges, holes and debris.

The record sits behind a Cloudflare challenge (a script gets HTTP 403): download it once by hand in a browser and put
it in <raw>/sdnet/. Its Download button gives DATA_Maguire_20180517_ALL.zip (528,286,896 bytes, the readme and
SDNET2018.zip inside), whose MD5 is the 'Checksum 677411e784f194422c90f52d9ed0d7c6' the record prints; the file link
gives SDNET2018.zip alone (528,284,810 bytes; MD5 905bd2c33490848ac7d0b868ea166265, SHA-256 617afc07...7331, read
from the inner zip of the checked bundle, 2026-11-06). Either is checked (fetch.placed), recorded in downloads.json
and kept when prepare.py deletes the raw folder (KEEP): no script can fetch it again.

The layout, read from the original zip's central directory (names and sizes only; the USU file as the Wayback Machine
captured it on 2024-04-14, 528,284,810 bytes): D/CD, D/UD, W/CW, W/UW, P/CP, P/UP with 2,025 / 11,595, 3,851 /
14,287 and 2,608 / 21,726 tiles, exactly Table 1 of the article (8,484 C and 47,608 U: 15.1 % C). A tile is
<photo>-<index>.jpg: photos 7001-7057 for the decks (54 numbers), 7069-7140 for the walls (72), 001-104 for the
pavements (104), the record's 230 photos; indices 1-252 (decks, walls) and 1-234 (pavements), 232-260 tiles per
photo, of which 23-41 cracked (medians per structure). 50 tiles have '_2' before the extension; 13 of them repeat
the photo and index of another tile, with the same size: 11 with the same label (kept once) and 2 with the other one
(D/UD/7039-112_2 and D/CD/7039-112, W/UW/7074-105_2 and W/CW/7074-105: all four dropped). 56,077 tiles remain.

Choices:
    labels      nok (C, cracked) / ok (U), as every binary inspection task of the benchmark; the structure is the
                stratum, so test and val keep each class's mix of decks, walls and pavements (cracked 24 / 45 / 31 %,
                uncracked 24 / 30 / 46 %; the test's 25 / 45 / 30 and 24 / 29 / 47 %)
    groups      the parent photo, both classes: its tiles share a surface and its light, and a crack runs over
                several of them
    presample   PER_CLASS = 2,000 tiles of each class, at random, drawn with the class shares of all 56,077 (nok 0.151;
                common.presample and common.shares). Whole, a photo holds about 240 tiles, so a test of 500 per class
                would come from a handful of photos; thinned to 2,000 per class, a photo keeps about 17 (at most 45) and
                the test spans 62 photos. 2,000 is four times the test's 500 per class and leaves 1,391 nok and 1,314 ok
                for the training draws
Splits, prepared on the GPU machine, 2026-11-06: test 500 nok + 500 ok (the 40 % capped at 500 per class), val 16 + 84
from 12 photos (the formula's 15 + 85, rounded per structure), pool 1,391 + 1,314, a training draw 45 + 255 per seed.
The draws take whole photos, then exactly these counts: the other 108 tiles of the test's photos and 87 of val's are
left out. No exact copy, look-alike or near-copy; 38 tiles are featureless (never matched as look-alikes, still grouped
by their photo).

Caveats: C marks any crack, from hairline to 25 mm, so some cracked tiles are hard to tell even by eye; both classes
hold the record's obstructions (shadows, edges, holes, debris); the tiles are 256 x 256 JPEG, barely resized at 224.
"""
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import common
import fetch

NAME = 'sdnet'
TASKS = (NAME,)
IMGSZ = 224
LICENCE = 'CC BY 4.0'
SOURCE = 'https://digitalcommons.usu.edu/all_datasets/48/'

POSITIVE = 'nok'
BUNDLE = 'DATA_Maguire_20180517_ALL.zip'      # the record's Download button
BUNDLE_URL = 'https://digitalcommons.usu.edu/context/all_datasets/article/1047/type/native/viewcontent'
BUNDLE_SIZE, BUNDLE_MD5 = 528286896, '677411e784f194422c90f52d9ed0d7c6'
ZIP = 'SDNET2018.zip'                         # the file link, or the zip inside the bundle
URL = 'https://digitalcommons.usu.edu/cgi/viewcontent.cgi?filename=2&article=1047&context=all_datasets&type=additional'
SIZE, SHA256 = 528284810, '617afc074b53bfb11121b9bbd10512784e1b0b21529f542c61f2c422ebf77331'
KEEP = (BUNDLE, ZIP)
STRUCTURES = {'D': 'deck', 'W': 'wall', 'P': 'pavement'}
EXPECTED = {'D/CD': 2025, 'D/UD': 11595, 'W/CW': 3851, 'W/UW': 14287, 'P/CP': 2608, 'P/UP': 21726}
TILE = re.compile(r'([DPW])/([CU])\1/(\d+)-(\d+)(_2)?\.jpg')
PER_CLASS = 2000


def download(raw):
    """The tiles of the zip downloaded by hand: the bundle (checked against the record's MD5, then its SDNET2018.zip)
    or SDNET2018.zip itself."""
    raw = Path(raw)
    folder = raw / 'SDNET2018'
    if (folder / '.extracted').exists():
        return folder
    archive = raw / ZIP
    if (raw / BUNDLE).exists():
        fetch.placed(raw / BUNDLE, BUNDLE_URL, size=BUNDLE_SIZE, md5=BUNDLE_MD5)
        inner = fetch.extract(raw / BUNDLE, raw / 'bundle', keep=lambda name: name.endswith('/' + ZIP))
        archive = next(inner.rglob(ZIP))
        bad = fetch.check(archive, SIZE, SHA256)
        if bad:
            raise RuntimeError(f'{archive}: {bad}; the bundle holds another SDNET2018.zip than the one pinned here')
    elif archive.exists():
        fetch.placed(archive, URL, size=SIZE, sha256=SHA256)
    else:
        raise RuntimeError(f'{raw}: neither {BUNDLE} nor {ZIP}; scripts get a Cloudflare challenge, so download the '
                           f'record by hand in a browser from {SOURCE} (Download) and put it in {raw}')
    fetch.extract(archive, folder, keep=lambda name: TILE.fullmatch(name) is not None)
    shutil.rmtree(raw / 'bundle', ignore_errors=True)
    return folder


def rows(folder, task):
    folder = Path(folder)
    count, tiles = Counter(), defaultdict(list)
    for path in sorted(folder.glob('*/*/*.jpg')):
        m = TILE.fullmatch(path.relative_to(folder).as_posix())
        if not m:
            continue
        structure, crack, photo, index, _ = m.groups()
        count[f'{structure}/{crack}{structure}'] += 1
        label = 'nok' if crack == 'C' else 'ok'
        tiles[structure, photo, index].append({'path': path, 'label': label, 'group': f'{structure}{photo}',
                                               'stratum': STRUCTURES[structure]})
    if dict(count) != EXPECTED:
        raise RuntimeError(f'{task}: tiles per folder {dict(sorted(count.items()))}, {EXPECTED} expected; check '
                           f'{ZIP}, or delete {folder} to extract it again')
    out = []
    for copies in tiles.values():                # a tile listed twice: once, or not at all when the labels disagree
        if len({r['label'] for r in copies}) == 1:
            out.append(copies[0])
    return common.presample(out, PER_CLASS), common.shares(out)
