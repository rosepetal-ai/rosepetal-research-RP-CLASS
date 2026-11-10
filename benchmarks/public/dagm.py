# RP-CLASS - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""DAGM 2007 ("Weakly Supervised Learning for Industrial Optical Inspection", the competition of the DAGM 2007
symposium; data by Wieler and Hahn, Robert Bosch Corporate Research, with Hamprecht): 10 tasks, one per synthetic
texture, class01 ... class10, each OK against NOK, an image being NOK when it carries a defect.

    task          train half: OK + NOK   test half: OK + NOK   val       pool        train, each seed
    class01         496 + 79               504 + 71            86 + 14   410 + 65    259 + 41
    class02         509 + 66               491 + 84            89 + 11   420 + 55    266 + 34
    class03         509 + 66               491 + 84            89 + 11   420 + 55    266 + 34
    class04         493 + 82               507 + 68            86 + 14   407 + 68    257 + 43
    class05         505 + 70               495 + 80            88 + 12   417 + 58    263 + 37
    class06         492 + 83               508 + 67            86 + 14   406 + 69    257 + 43
    class07-10    1,000 + 150 each       1,000 + 150 each      87 + 13   913 + 137   261 + 39

Data: Zenodo record 12750201 (version 2 of 10.5281/zenodo.8086135), Class1.zip ... Class10.zip, 2,947,345,573 bytes
in all; each is checked by its size and the MD5 Zenodo publishes (it publishes no SHA-256; fetch records one). A zip
holds Class<k>/{Train,Test}/<id>.PNG, 512 x 512 8-bit grey, the ids unique across the halves, and in each half a
folder Label/ with a mask <id>_label.PNG for each defective image (0 and 255: an ellipse roughly around the defect,
the weak label of the competition) and Labels.txt (CRLF lines):
    1
    <id> TAB <1 defective, 0 not> TAB <id>.PNG TAB 0 TAB <id>_label.PNG if defective, else 0
The label comes from Labels.txt, checked against the folder: it must list exactly the half's images, and the images
it marks defective must be exactly those with a mask. Licence: CC BY 4.0 (the record's licence field, cc-by-4.0).

Splits: the official halves ("randomly split into a training and testing sub-dataset of equal size", the record):
the Test half is the test, whole; val comes from the Train half, and the rest of it is the pool the training draws
come from. The val, pool and train above were prepared on the GPU machine, 2026-11-06 (OK + NOK, the same size for
every seed); no image is dropped or left out.

Code of honour: the authors ask researchers not to use or analyse the competition datasets, classes 7-10, "before
the development is completed". The library is never developed or tuned on them: they are prepared like the others,
and only final benchmark runs read them. Their halves above come from the zip listings alone (member names and sizes:
150 masks per half, and each Labels.txt is 27,103 bytes, exactly 1,000 non-defective and 150 defective lines of the
format above), and the preparation found the same; no image or label file of theirs was opened to write or check
this script.

Caveats: the textures and defects are synthetic ("artificially generated, but similar to real world problems"), with
exactly one defect per defective image: the only synthetic set among the industrial ones. The halves do not split the
defects evenly (class01: 79 in train, 71 in test). Shrunk to the 32 x 32 of common.py's look-alike test, independently
generated images of a texture can look alike: in class04, 165 pairs pass it (dHashes within 4 bits, thumbnails within
3 / 255; the closest 1.93 apart, and 7.65 or more at 128 x 128, so no near-copies). Look-alikes only join a group:
99 joins in class04 and 102 in class10 (an image joining an earlier look-alike's group; groups of at most 4, no
near-copy among them); the tests stay whole, and
val takes whole groups without leaving any image out. The other eight classes have no look-alike.

Choices: the masks are extracted only for the check above; the Thumbs.db files some folders carry are not. The
classes come one at a time, each zip deleted once extracted (the images are stored, not compressed, so the files take
what the zip took): the download peaks at 3.4 GB and leaves 2.9 GB. Images keep their size: the benchmark resizes them
to 448 (IMGSZ).
"""
from pathlib import Path

import fetch

NAME = 'dagm'
TASKS = tuple(f'class{k:02d}' for k in range(1, 11))
IMGSZ = 448
LICENCE = 'CC BY 4.0'
SOURCE = 'https://zenodo.org/records/12750201'
URL = 'https://zenodo.org/records/12750201/files/Class{}.zip?download=1'
ZIPS = {1: (267943772, '9ed7f0480aea9bd0a83b429a17e3e750'), 2: (228042837, '024f1f6c6d11d72ee12f19dcc98aa3d8'),
        3: (202923699, '57a337e93e38c71e01d15770e252f46d'), 4: (196885903, 'd6852ae3667db59e6a4d586d98633540'),
        5: (215005526, '463a932d87a5c0e02ee10cdb3265c318'), 6: (260830045, '40ba70f232fe1cb891a46c1b77dc4940'),
        7: (380424293, '67571381c8aa5f82d5180e6e8838f735'), 8: (392568394, 'd02a892a8b6180893c8ea018848606cb'),
        9: (392362320, '083d8da49d46614202a332f9e9dd1bca'), 10: (410358784, '2743ebc79cb91551b51599a67f0e7528')}
LABELS = {'0': 'ok', '1': 'nok'}
POSITIVE = 'nok'


def download(raw, tasks=TASKS):
    """raw/class<kk>/Class<k>/{Train,Test}/ for the tasks asked (all by default), one zip at a time: classes 7-10 are
    never fetched while only the development classes are prepared."""
    raw = Path(raw)
    for task in tasks:
        k = int(task[5:])
        archive, dest = raw / f'Class{k}.zip', raw / task
        if not (dest / '.extracted').exists():
            size, md5 = ZIPS[k]
            fetch.download(URL.format(k), archive, size=size, md5=md5)
            fetch.extract(archive, dest, keep=lambda name: name.endswith(('.PNG', '/Labels.txt')))
        archive.unlink(missing_ok=True)
    return raw


def _half(path, split):
    lines = (path / 'Label' / 'Labels.txt').read_text().splitlines()[1:]
    entries = [line.split('\t') for line in lines if line.strip()]
    if any(len(e) < 5 or e[1] not in LABELS for e in entries):
        raise RuntimeError(f'{path}: Labels.txt is not in the expected format; see the docstring of dagm.py')
    images = {p.name for p in path.glob('*.PNG')}
    masked = {p.name.replace('_label', '') for p in (path / 'Label').glob('*_label.PNG')}
    if {e[2] for e in entries} != images or {e[2] for e in entries if e[1] == '1'} != masked:
        raise RuntimeError(f'{path}: Labels.txt disagrees with the images or the masks; delete {path.parents[1]} to '
                           'download it again')
    return [{'path': path / e[2], 'label': LABELS[e[1]], 'split': split} for e in entries]


def rows(folder, task):
    root = Path(folder) / task / f'Class{int(task[5:])}'
    return _half(root / 'Train', 'train') + _half(root / 'Test', 'test')
