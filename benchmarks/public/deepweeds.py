# RP-CLASS - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""DeepWeeds (docs/BENCHMARK.md, G5): weeds of northern Australia photographed in situ, 9 classes (8 species and
negative), one task: cluttered field scenes with a natural imbalance.

Data: images.zip on Google Drive (file 1xnK3B6K6KekDI55vwJ0vnc2IGoDga9cj, linked from the README of
github.com/AlexOlsen/DeepWeeds; Olsen et al., Scientific Reports 2019), 491,516,047 bytes. Drive gives no checksum: the
SHA-256 pinned below is that of the download of 2026-11-06. The zip holds 17,509 RGB JPEGs of 256 x 256 px at its
root, named YYYYMMDD-HHMMSS-<instrument>.jpg (the README says instruments 0 to 3; 27 images, all of 2017-09-06, carry
a 4). The labels are labels/labels.csv (Filename, Label, Species), 598,898 bytes, pinned at the repository's last commit
da084f6 (2020-10-05; the file is unchanged since the first commit). Every image has one row and every row an image,
with the counts of the paper's Table 1:
    Chinee apple 1,125   Lantana 1,064   Parkinsonia 1,031   Parthenium 1,022   Prickly acacia 1,062
    Rubber vine 1,009   Siam weed 1,074   Snake weed 1,016   Negative 9,106 (52 %)
Licence: CC BY 4.0 for the images and labels ('The source code, images and annotations are licensed under CC BY 4.0
license', the README); the repository's contents are Apache-2.0.

Choices:
    classes   the Species column, lowercased, '_' for ' ': chinee_apple, lantana, ..., snake_weed and negative
    test      built: labels/{train,val,test}_subset<k>.csv are the paper's five 60 / 20 / 20 cross-validation subsets,
              not one held-out test. 500 negatives (the cap; 40 % would be 3,642) and round(0.4 x n_c) of each
              species, 404 to 450
    groups    none. The names give the capture time, but a run of shots is not a burst of one plant: an instrument
              moves between shots, 16 s apart at the median (5 to 113 s for 90 % of them), and of the 17,504 pairs of
              consecutive shots of one instrument, the 828 under 5 s apart differ nearly as much as two random images
              (their 32 x 32 grey thumbnails differ by 31.9 / 255 per pixel at the median, against 37.0); 33 of them
              change class. Only 31 consecutive pairs, 1 to 64 s apart, are within 8 / 255: 30 retakes of one scene
              and two patches of bare sand. A time window would chain whole sessions, whose species alternate within
              seconds, into a few large mixed-label groups: a built split would take whole sessions and leave out
              what it does not need of them. What groups this set is finalize's look-alike check: it joins 57 pairs,
              each shot by one instrument on one day, 18 of them among those 31, and none is a near-copy: at 128 px
              they are 2.05 to 12.75 / 255 apart.
    disk      the zip is deleted once extracted: 0.98 GB at the peak, 0.49 GB after

Counts (prepared on the GPU machine, 2026-11-06): 17,509 listed, no exact copy or featureless image, the 57
look-alike pairs grouped, no near-copy, none left out.
    test    chinee_apple 450, lantana 426, negative 500, parkinsonia 412, parthenium 409, prickly_acacia 425,
            rubber_vine 404, siam_weed 430, snake_weed 406: 3,862
    val     6 of each species and 52 negatives: 100
    pool    599 to 669 of each species and 8,554 negatives: 13,547
    train   20 of each species and 156 negatives per seed: 316

Caveats: species are tied to places (Parkinsonia was photographed only at Cluden, Siam weed almost only at Black
River; the paper's Table 1), and labels.csv gives no place, so the test cannot be place-disjoint: a model can score
from the scene (soil, light, neighbouring flora) rather than the plant. It is the same split for every method, but
the numbers overstate how well the weeds themselves are recognised. The 12 retakes finalize does not join (3.1 to
7.9 / 255 apart) can straddle test and pool; a training draw takes 1.8 % (negatives) to 3.3 % (species) of the pool,
so on average fewer than 0.1 of them reaches a training set.
"""
import csv
import re

import fetch

NAME = 'deepweeds'
TASKS = (NAME,)
IMGSZ = 224
LICENCE = 'CC BY 4.0 for the images and annotations; the repository contents are Apache-2.0'
SOURCE = 'https://github.com/AlexOlsen/DeepWeeds'
DRIVE_ID = '1xnK3B6K6KekDI55vwJ0vnc2IGoDga9cj'
SIZE, SHA256 = 491516047, '0961f63c01b997bfab1559ad09e99c0e8130617fd96a8b92fdc09940e01b0ce8'
COMMIT = 'da084f62bcd1a2b0afeb8b81f1a27be3186391a1'
LABELS_URL = f'https://raw.githubusercontent.com/AlexOlsen/DeepWeeds/{COMMIT}/labels/labels.csv'
LABELS_SIZE, LABELS_SHA256 = 598898, '6fb95b89fd9d384f94e185a5cab6c5da7c987649399c90618ecc60acfb0112eb'
FILE = re.compile(r'\d{8}-\d{6}-\d\.jpg')
COUNT = 17509


def download(raw):
    """labels.csv -> raw/labels.csv and images.zip -> raw/images, once; the zip is deleted when its images are out."""
    fetch.download(LABELS_URL, raw / 'labels.csv', size=LABELS_SIZE, sha256=LABELS_SHA256)
    images = raw / 'images'
    if not (images / '.extracted').exists():
        archive = fetch.google_drive(DRIVE_ID, raw / 'images.zip', size=SIZE, sha256=SHA256)
        fetch.extract(archive, images)
        archive.unlink()
    return raw


def rows(folder, task):
    with open(folder / 'labels.csv', newline='') as f:
        out = [{'path': folder / 'images' / r['Filename'], 'label': r['Species'].lower().replace(' ', '_')}
               for r in csv.DictReader(f) if FILE.fullmatch(r['Filename'])]
    if len(out) != COUNT or not all(r['path'].is_file() for r in out):
        raise RuntimeError(f'deepweeds: {folder} does not hold the {COUNT} images of labels.csv; delete it to '
                           f'download it again')
    return out
