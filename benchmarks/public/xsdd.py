# RP-CLASS - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""X-SDD, the Xsteel surface defect dataset (Feng et al., Symmetry 13(4):706, 2021): 7 types of surface defect on
hot-rolled steel strip, one type per image, 1,360 grey images of 128 x 128 stored as RGB. Already small (1,360
images) and imbalanced (397 down to 63 per class), so it is used whole.

Data: the authors' GitHub repository, which the paper's Data Availability Statement names as the dataset's location,
at its last commit (adf8808, 2021-04-04, read from the GitHub API), as the zip GitHub makes of that commit:
6,642,107 bytes, the same bytes on two downloads, SHA-256
7ddbb1168e406a03aace6e62558026b89da214dce15aa714af5fbc1a06fd7e07. GitHub publishes no checksum, so only the size is
pinned. Only X-SDD/datas/<class>/ is extracted; the repository's code is never run.
Licence: Apache-2.0, the repository's LICENSE, with its copyright line never filled in ('Copyright [yyyy] [name of
copyright owner]'). It is a software licence put on images, and the rights to the images (taken via xBang Inc.,
where the first author was an intern, per the paper's acknowledgements) are not documented. We publish manifests
and metrics only, never an image; the authors should confirm before any image is shown.

Classes, folder -> label (short; the folders have spaces): images
    finishing roll printing             roll_printing       203 (142 PNG, 61 JPEG)
    iron sheet ash                      iron_sheet_ash      122 (one image is 122 x 128)
    oxide scale of plate system         oxide_plate          63
    oxide scale of temperature system   oxide_temperature   203
    red iron                            red_iron            397 (one image is 130 x 128; 9 pairs of byte copies,
                                                                 2078-2081 = 2082-2085 and 2205-2209 = 2210-2214)
    slag inclusion                      slag_inclusion      238
    surface scratch                     scratch             134
Every class is kept: the smallest, 63, has more than the protocol's 50.

Groups: none but finalize's look-alikes and near-copies, and it finds neither. The file names are numbers (1xxx to
7xxx, the readme's order of classes) with no source image or strip id; consecutive numbers are no more alike than two
random images of the class (mean 32 x 32 thumbnail difference), except in part in red_iron (median 14.1 against
20.2), where some neighbours may come from one frame, but none is a look-alike. No image is a flipped or rotated copy
of another (all 8 checked).

PNG -> JPEG. Every image is a baseline JPEG with one set of quantization tables (Pillow's quality 90) and 4:2:0
chroma, except 142 PNGs, all of roll_printing (6001-6142). They carry no JPEG 8 x 8 grid (blockiness, the mean
step across the grid over the mean step inside it, best of the 8 grid phases: median 1.03 for the PNGs, 1.21 to
1.44 for the JPEGs of every class), so the missing grid alone would find 70 % of roll_printing. download() encodes
them once as the rest of the dataset is encoded, so that the label is only in what the images show: then the same
tables, blockiness median 1.47 (1.44 for the class's own JPEGs), at most 1.65 grey levels of mean change. Their
manifest source is the .jpg next to the zip's .png; Pillow 9.0 and 11.3 (libjpeg-turbo 2.1 and 3.1) write the same
bytes.

Splits (common.py), prepared on the GPU machine, 2026-11-06: 1,360 listed, the 9 byte copies dropped, no look-alike
or near-copy; 7 faint images are featureless (a 32 x 32 thumbnail standard deviation under 2: slag_inclusion 1210,
1211 and 1219, scratch 4009 and 4019, roll_printing 6025 and 6167), so they join no group. Test built, 40 % of each
class: 540 (iron_sheet_ash 49 · oxide_plate 25 · oxide_temperature 81 · red_iron 155 · roll_printing 81
· scratch 54 · slag_inclusion 95); val 101 (9 · 5 · 15 · 29 · 15 · 10 · 18); pool 710 (64 · 33 · 107 · 204
· 107 · 70 · 125); train 306 per seed (27 · 20 · 45 · 86 · 45 · 30 · 53); no image left out.
"""
from pathlib import Path

from PIL import Image

import fetch

NAME = 'xsdd'
TASKS = (NAME,)
IMGSZ = 224
REPO = 'Fighter20092392/X-SDD-A-New-benchmark'
COMMIT = 'adf880814dbd365c73fc3643019272b5637fecc2'
LICENCE = 'Apache-2.0 (the repository LICENSE, copyright line never filled in; image rights not documented)'
SOURCE = f'https://github.com/{REPO}/blob/{COMMIT}/LICENSE'
URL = f'https://codeload.github.com/{REPO}/zip/{COMMIT}'
SIZE, SHA256 = 6_642_107, '7ddbb1168e406a03aace6e62558026b89da214dce15aa714af5fbc1a06fd7e07'
NOTES = {'reencoded': 'the 142 PNG of finishing roll printing (6001-6142) written as JPEG once (quality 90, 4:2:0, '
                      'as the other 1,218): every PNG sits in that one class, and the file format alone would give '
                      'away 70 % of it'}
DATA = f'X-SDD-A-New-benchmark-{COMMIT}/X-SDD/datas'
CLASSES = {'finishing roll printing': 'roll_printing', 'iron sheet ash': 'iron_sheet_ash',
           'oxide scale of plate system': 'oxide_plate', 'oxide scale of temperature system': 'oxide_temperature',
           'red iron': 'red_iron', 'slag inclusion': 'slag_inclusion', 'surface scratch': 'scratch'}
JPEG = {'quality': 90, 'subsampling': '4:2:0'}          # the settings of the dataset's 1,218 JPEGs


def download(raw):
    """The commit's zip, its data folder only; the zip is deleted once extracted and the PNGs become JPEGs."""
    raw = Path(raw)
    archive = raw / f'X-SDD-A-New-benchmark-{COMMIT}.zip'
    if not (raw / '.extracted').exists():
        fetch.download(URL, archive, size=SIZE, sha256=SHA256)
        fetch.extract(archive, raw, keep=lambda name: name.startswith(DATA + '/'))
    archive.unlink(missing_ok=True)
    for png in sorted((raw / DATA).glob('*/*.png')):
        _to_jpeg(png)
    return raw / DATA


def _to_jpeg(png):
    """png -> the same name in .jpg, encoded as the rest of the dataset; the PNG is deleted."""
    jpg = png.with_suffix('.jpg')
    if not jpg.exists():
        part = png.with_suffix('.part')
        with Image.open(png) as im:
            im.convert('RGB').save(part, 'JPEG', **JPEG)
        part.rename(jpg)
    png.unlink()


def rows(folder, task):
    folder = Path(folder)
    found = sorted(p.name for p in folder.iterdir() if p.is_dir())
    if found != sorted(CLASSES):
        raise RuntimeError(f'xsdd: class folders {found} in {folder}, expected {sorted(CLASSES)}; delete the download '
                           f'and prepare again')
    out = []
    for name, label in CLASSES.items():
        for p in sorted((folder / name).iterdir()):
            if p.suffix != '.jpg':
                raise RuntimeError(f'xsdd: unexpected file {p}; delete the download and prepare again')
            out.append({'path': p, 'label': label})
    return out
