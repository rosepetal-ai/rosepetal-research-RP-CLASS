# RP-CLASS - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""Tyres (docs/BENCHMARK.md, I6): OK / NOK on photos of tyres, defective vs good, one task: the high-resolution set
of the benchmark (IMGSZ 448).

Data: Mendeley Data record bn7ch8tvyp v1, 'Digital images of defective and good condition tyres' (Pathmanaban P. et
al., Velammal Engineering College; published 2023-08-22): 1,856 JPEGs in two folders, defective 1,028 and good 828,
2,917,776,336 bytes (the description says 1,854). Mendeley serves the record as one zip, bn7ch8tvyp-1.zip of
2,859,136,943 bytes, and publishes its size and SHA-256 (https://data.mendeley.com/public-api/zip/bn7ch8tvyp/1, built
2023-08-22): one request instead of 1,856. Its central directory holds exactly the API's 1,856 names and sizes, under
'<record title>/defective/' and '<record title>/good/', and the 852 members read from it (46 %) have the SHA-256 the
API lists for each file. The download URL answers with a redirect to an S3 URL signed for 300 s; a resumed download
asks again. Should Mendeley ever rebuild the zip, its SHA-256 changes and the download stops: the per-file SHA-256 of
the API list then tell whether the photos are the same.
Licence: CC BY 4.0, the record's data_licence field.

Photos: RGB JPEGs of three kinds, in both classes (EXIF of the 852 read; defective / good):
    2017, Photoshop CS6    160 / 101   squares (all but one) of 224-3,024 px, close-ups of sidewalls and treads
                                       (an iPhone in the EXIF of most)
    no EXIF                151 / 211   squares (all but 7) of 224-4,032 px, 61 under 300 px: crops of many brands
    2022 phone photos      101 / 128   4,000 x 1,800, 4,000 x 3,000 and 4,032 x 2,268 px: OnePlus 89 / 45, OPPO F17
                                       Pro 12 / 67, iPhone 11 0 / 16; landscape 89 / 44, portrait 12 / 84

Choices:
    test      built, as there is no official split: 400 defective and 331 good (0.4 of each class)
    copies    28 pairs of byte-identical files, all defective and numbered in sequence ((32)/(33), then every pair of
              (812)-(849) and (852)-(867)): finalize keeps one of each, leaving 1,000 defective and 1,828 photos
    groups    none: the record names no tyre or session; finalize groups 9 look-alikes (and finds 5 near-copies,
              none with two labels): groups of 2 or 3 photos, the largest good (521)-(523)
    size      the photos stay as they are; IMGSZ 448 (docs/BENCHMARK.md, Protocol)
    disk      the zip is deleted once extracted: 2.86 GB to download, 5.8 GB at the peak, 2.9 GB after

Counts (prepared on the GPU machine, 2026-11-06): test 400 defective + 331 good, val 55 + 45, pool 545 + 452, train
164 + 136 per seed, share defective 0.547; no photo of the test's or val's groups is left out.

Caveats: consecutive numbers are often shots of one tyre taken seconds apart (Defective 386-387, 391-397 and 409-413,
1-7 s apart; good 352-354 and 604-606); they are no look-alikes, so finalize does not group them (Defective 391-394 are
in the test, 395 and 396 in training draws): the built test shares tyres with train and every method scores higher than
it would on unseen tyres. The classes come partly from different sessions: the 2022 defective tyres are worn and dusty,
photographed outdoors, mostly with a OnePlus and in landscape, the good ones new, in a shop, mostly with an OPPO and in
portrait; scene, camera and aspect are cues. good (379), in the test, carries EXIF orientation 8 (1 of the 852 read; 74
more carry the invalid 0, which both readers ignore): finalize writes it with the tag set to 1 (the one tag it cleared),
so RP-CLASS (PIL) and YOLO-cls (OpenCV) both see it as stored. The 2017 photos predate the record (they look like
Siegel's CC0 tyre set, Harvard Dataverse doi:10.7910/DVN/Z3ZYLI, not checked photo by photo), and the small squares'
origin is undocumented. Used tyres, not in-line inspection.
"""
from pathlib import Path

import fetch

NAME = 'tyres'
TASKS = (NAME,)
IMGSZ = 448
LICENCE = 'CC BY 4.0'
SOURCE = 'https://data.mendeley.com/public-api/datasets/bn7ch8tvyp'
URL = 'https://data.mendeley.com/public-api/zip/bn7ch8tvyp/download/1'
SIZE, SHA256 = 2859136943, '897278a26d4df27889c87011337219ab46c862833b2b25815a3f3f87682c05d7'
TOP = 'Digital images of defective and good condition tyres'
COUNT = {'defective': 1028, 'good': 828}                  # the record's folders
LABELS = {'defective': 'nok', 'good': 'ok'}               # as every binary inspection task of the benchmark
POSITIVE = 'nok'


def download(raw):
    """The record's zip -> raw/images, once; the zip is deleted when its photos are out."""
    images = Path(raw) / 'images'
    if not (images / '.extracted').exists():
        archive = fetch.download(URL, Path(raw) / 'bn7ch8tvyp-1.zip', size=SIZE, sha256=SHA256)
        fetch.extract(archive, images)
        archive.unlink()
    return images / TOP


def rows(folder, task):
    folder = Path(folder)
    out = [{'path': p, 'label': LABELS[sub]} for sub in COUNT for p in sorted((folder / sub).glob('*.jpg'))]
    count = {sub: sum(r['label'] == LABELS[sub] for r in out) for sub in COUNT}
    if count != COUNT:
        raise RuntimeError(f'tyres: {count} photos in {folder}, {COUNT} expected; delete {folder.parent} to download '
                           f'it again')
    return out
