# RP-CLASS - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""Development suite (never part of the benchmark): Kather 2016, tissue types in tiles of H&E colorectal cancer
histology, 8 classes, one task.

Data: Kather_texture_2016_image_tiles_5000.zip of Zenodo record 53169 (Kather et al., 'Multi-class texture analysis
in colorectal cancer histology', Scientific Reports 6:27988, 2016), 258,098,431 bytes. Its MD5 matched the one Zenodo
publishes (0ddbebfc56344752028fda72602aaade) on 2026-11-06, and the SHA-256 pinned beside it is that download's. The
zip holds Kather_texture_2016_image_tiles_5000/<nn>_<CLASS>/<hex>_<region>.tif_Row_<y>_Col_<x>.tif and nothing else:
5,000 tiles, 625 of each class. Every tile is a 150 x 150 RGB TIFF (PackBits) of 74 x 74 um (0.495 um/px, Aperio
ScanScope, 20x), and all of them decode. The paper's classes:
    tumour epithelium
    simple stroma (tumour and extra-tumoural stroma, smooth muscle)
    complex stroma (with single tumour cells and/or few immune cells)
    immune cells (conglomerates, sub-mucosal lymphoid follicles)
    debris (necrosis, haemorrhage, mucus)
    normal mucosal glands
    adipose tissue
    background (no tissue)
Licence: CC BY 4.0. That is the record's licence (cc-by-4.0, open access, no login) and the paper's data statement
('We release all raw data under a Creative Commons Attribution 4.0 International License'). Cite the paper.

Provenance, read from the names. The paper took 'Ten anonymized H&E stained CRC tissue slides', and 'contiguous
tissue areas were manually annotated and tessellated, creating 625 non-overlapping tissue tiles'.
    <region>   the annotated area's image: CRC-Prim-HE-<slide> (01 to 10), mostly followed by _<nnn>, sometimes
               with a, b, c or _copy after it
    Row, Col   the tile's top-left pixel in that image, in steps of 150 from 1
    <hex>      2 to 5 hex digits, not unique (4,795 distinct), of no use here
The tiles of a region are neighbours. Across the border of two horizontal neighbours, the pixels differ as little as
two adjacent columns inside one tile: medians of 9.7 against 10.4 levels out of 255 for tumour, 2.2 against 2.3 for
adipose. Two tumour tiles of different regions differ by 46.4. Regions are numbered within each class folder: 13
positions recur in two folders with other pixels. Tiles per slide:
    slide       01    02    03    04    05    06    07    08    09    10
    tumor       54   107    30    62    57    38   120    55    77    25
    stroma      51    50   170   124    27    54    68    23    31    27
    complex     72    16   132     7    90    90    84    25    52    57
    lympho     208   106     4    16   159    20     6     8    85    13
    debris      15     2   111   141    32    20    59   193    21    31
    mucosa       1     3   111     0   152   120    77    85    76     0
    adipose      0     0   182    31   144   129    89     0     0    50
    empty        0     0     0     0     0   590     0     0     0    35
    all        401   284   740   381   661 1,061   503   389   342   238

Choices:
    classes   the folder names without their number, lowercased, as TFDS's colorectal_histology names them: tumor,
              stroma, complex, lympho, debris, mucosa, adipose, empty
    test      built, as there is no official split (the paper cross-validated over random tenths): 250 tiles of each
              class, 40 %
    groups    the region: the class folder and the region name without its a, b, c or _copy. 13 such names sit beside
              their plain name in one folder. None shares a 50 x 50 pixel block with it at any offset, but they read as
              parts of one area. That gives 306 groups of 1 to 182 tiles: tumor 91, complex 71, stroma 46, lympho 38,
              debris 36, mucosa 17, adipose 7. empty is left ungrouped. Its 625 tiles come from 2 regions, 590 of
              slide 06 and 35 of slide 10, which whole-group draws cannot split into test, val and pool. It is blank
              glass (447 of its tiles are featureless for finalize), with nothing of a slide to learn
    slides    no split can keep them apart, since empty lies on 2 slides and adipose on 6. Test, val and the training
              draws all come from the same 10 slides. That is fine for development, and it is why the set is not in
              the benchmark
    files     the tiles as they are: RP-CLASS and YOLO read TIFF (NCT-CRC-HE's patches are TIFF too)
    size      150 px tiles; IMGSZ 224 upsamples them (x 1.49), the same for every method
    disk      the zip is deleted once extracted: 0.59 GB at the peak (the zip and 331 MB of tiles), 0.33 GB after

Counts (prepare.py, 2026-11-06): 5,000 listed. No exact copy, look-alike or near-copy; 453 tiles are featureless (447
empty, 6 adipose).
    test    250 of each class, 2,000, from 5 (adipose) to 35 (tumor) regions
    val     12 of each class, 96, from 1 region per class (2 for complex, 3 for debris)
    pool    adipose 144, complex 339, debris 307, empty 363, lympho 302, mucosa 330, stroma 356, tumor 349: 2,490
    train   38 of each class per seed: 304
The other tiles of the regions that test and val took, 257 and 157, are used nowhere. Of adipose's 7 regions, 5 went to
the test and 1 to val. Its pool is therefore one region of 144 tiles (CRC-Prim-HE-05_032), and every adipose training
tile comes from that one area.

Caveats: one lab, one scanner, 10 slides of '10 independent samples' (the paper). The test holds regions never trained
on, from slides that are: it measures new areas under known patients and stains, not new patients, so it is optimistic
(the same for every method). Val takes whole regions too, so its 12 tiles of a class come from 1 to 3 regions: a
noisy guide. The test, 250 per class from 5 to 35 regions, is the score to read. The set shares its domain and its
group with the benchmark's NCT-CRC-HE (G3: colorectal H&E, Kather, partly the same Mannheim archive). A choice that
helps here may help G3 for that reason, so weigh it with the other development sets. It is balanced by design, 625
tiles per class, unlike most inspection data.
"""
import re

import fetch

NAME = 'kather2016'
TASKS = (NAME,)
IMGSZ = 224
LICENCE = 'CC BY 4.0'
SOURCE = 'https://zenodo.org/api/records/53169'
URL = 'https://zenodo.org/records/53169/files/Kather_texture_2016_image_tiles_5000.zip?download=1'
ZIP, SIZE = 'Kather_texture_2016_image_tiles_5000.zip', 258098431
MD5 = '0ddbebfc56344752028fda72602aaade'                                    # as Zenodo gives it
SHA256 = '936d0b95ae0f04f1f7b18a2b7e8c55776c3e887d102f645bcee752d1a6626a32'  # that download's
CLASSES = ('01_TUMOR', '02_STROMA', '03_COMPLEX', '04_LYMPHO', '05_DEBRIS', '06_MUCOSA', '07_ADIPOSE', '08_EMPTY')
TILE = re.compile(r'[0-9A-F]{2,5}_(CRC-Prim-HE-\d\d(?:_\d{3})?)(?:[a-c]|_copy)?\.tif_Row_\d+_Col_\d+\.tif')
UNGROUPED = '08_EMPTY'
NOTES = {'ungrouped': {'empty': '625 tiles from 2 regions (590 and 35), too few whole groups for test, val and pool; '
                                'blank glass'}}
COUNT = 5000


def download(raw):
    """Kather_texture_2016_image_tiles_5000.zip -> raw/images, once; the zip is deleted when its tiles are out."""
    images = raw / 'images'
    if not (images / '.extracted').exists():
        fetch.extract(fetch.download(URL, raw / ZIP, size=SIZE, sha256=SHA256, md5=MD5), images)
    (raw / ZIP).unlink(missing_ok=True)
    return images / 'Kather_texture_2016_image_tiles_5000'


def rows(folder, task):
    """Labels: the folder names without their number (01_TUMOR -> tumor). Groups: the tile's region (the TILE match),
    within its class folder, except for empty."""
    out = []
    for c in CLASSES:
        for p in sorted((folder / c).glob('*.tif')):
            m = TILE.fullmatch(p.name)
            if m is None:
                raise RuntimeError(f'kather2016: {p.name} is not a tile name; delete {folder.parent} to download it '
                                   f'again')
            out.append({'path': p, 'label': c[3:].lower(), 'group': '' if c == UNGROUPED else f'{c}/{m.group(1)}'})
    if len(out) != COUNT:
        raise RuntimeError(f'kather2016: {len(out)} tiles in {folder}, {COUNT} expected; delete {folder.parent} to '
                           f'download it again')
    return out
