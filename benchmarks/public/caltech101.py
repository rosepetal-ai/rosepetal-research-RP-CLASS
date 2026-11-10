# RP-CLASS - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""Caltech-101 (docs/BENCHMARK.md, G2): objects in web photos, 100 classes, one task: the many-class end of the
benchmark.

Data: caltech-101.zip of CaltechDATA record mzrjq-6wc02 (Fei-Fei, Fergus and Perona 2004), 137,414,764 bytes; its MD5
matched the one the record publishes (3138e1922a9193bfa496528edbbc45d0, also torchvision's) on 2026-11-06, and the
SHA-256 pinned below is that download's. The signed download URL answers HEAD with 403 and GET with the file
(fetch.download uses GET). The zip nests 101_ObjectCategories.tar.gz (131,740,031 bytes, MD5
b224c7392d521a49829488ab0f1120d9, the archive torchvision pinned before) beside Annotations.tar, show_annotation.m and
macOS resource forks. The tar holds 101_ObjectCategories/<class>/image_<nnnn>.jpg, 9,144 JPEGs in 102 folders, and
one more file, BACKGROUND_Google/tmp (a JPEG without an extension). Every JPEG decodes: 8,733 RGB, 411 grey.
Licence: CC BY 4.0, the record's rights field. The photos were collected from Google Images, so their own copyright
stays with their owners: the benchmark publishes metrics only and never shows an image (docs/BENCHMARK.md, Never).

Choices:
    classes   100, as CoOp has them: BACKGROUND_Google (467 unrelated photos, not a class) and Faces_easy (tighter
              crops of the 435 Faces photos: each matches its Faces image to 2-3 grey levels out of 255) are dropped.
              Names are lowercased (Faces, Leopards, Motorbikes -> faces, leopards, motorbikes) so that the sorted
              order is alphabetical. 8,242 images, 31 (inline_skate) to 800 (airplanes) per class, 80 to 708 px wide
              and 92 to 494 px high (medians 300 and 231).
    grey      381 of them are grey, in 72 classes, all 123 of car_side among them: kept as they are, the library reads
              8-bit grey
    test      built, as there is no official split (TFDS's 30 per class is a convention): round(0.4 x n_c)
    groups    none given; only finalize's look-alikes and near-copies
    disk      each archive is deleted once extracted: 290 MB at the peak (the tar.gz and the 158 MB of images it gives
              on disk), 158 MB after

Counts (prepared on the GPU machine, 2026-11-06): no exact copies; 45 look-alikes joined into 40 groups of 2 to 5
images in 27 classes, none across classes; 14 of them are also near-copies (menorah image_0014 to 0017 and 0053, for
one, are one drawing, and 0015 to 0017 and 0053 near-copies of each other). Test 3,302 (12 for inline_skate to 320 for
airplanes), val 510 (5 per class; 10 for airplanes and motorbikes), pool 4,424, train 1,955 per seed (20 per class; 29
for airplanes and motorbikes, 14 to 19 for the 17 classes whose pool is smaller). The groups test and val take leave 6
images out: 3 of menorah and 1 of pigeon with the test, 1 of garfield and 1 of pyramid with val.

Caveats: a DINOv2 / DINOv3 retrieval seed (DINOv2 Table 15 lists its train split), so the DINOv3 baseline has likely
seen similar images. car_side is the only class that is all grey, a colour cue a model can learn. With 5 val images
for most classes, val picks the epoch on a noisy macro-F1.
"""
import shutil

import fetch

NAME = 'caltech101'
TASKS = (NAME,)
IMGSZ = 224
LICENCE = ("CC BY 4.0; the photos come from Google Images and keep their owners' copyright: metrics only, no image "
           "shown")
SOURCE = 'https://data.caltech.edu/api/records/mzrjq-6wc02'
URL = 'https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip?download=1'
SIZE, SHA256 = 137414764, '331234750fc7f77520e50d9565e8b6907b03565e30320da1401130db08c61f91'
INNER = 'caltech-101/101_ObjectCategories.tar.gz'
DROPPED = ('BACKGROUND_Google', 'Faces_easy')
NOTES = {'dropped_classes': {'BACKGROUND_Google': '467 images, no object class', 'Faces_easy': '435 images, crops of '
                             'the Faces photos'}}
CLASSES, COUNT = 100, 8242


def download(raw):
    """caltech-101.zip -> its 101_ObjectCategories.tar.gz -> raw/images, once; each archive is deleted when it is
    out."""
    images = raw / 'images'
    if not (images / '.extracted').exists():
        fetch.extract(_inner(raw), images)
        shutil.rmtree(raw / 'zip')
    return images / '101_ObjectCategories'


def _inner(raw):
    unzipped = raw / 'zip'
    if not (unzipped / '.extracted').exists():
        archive = fetch.download(URL, raw / 'caltech-101.zip', size=SIZE, sha256=SHA256)
        fetch.extract(archive, unzipped, keep=lambda name: name == INNER)
        archive.unlink()
    return unzipped / INNER


def rows(folder, task):
    out = [{'path': p, 'label': p.parent.name.lower()} for p in sorted(folder.glob('*/*.jpg'))
           if p.parent.name not in DROPPED]
    found = len({r['label'] for r in out})
    if (len(out), found) != (COUNT, CLASSES):
        raise RuntimeError(f'caltech101: {len(out)} images of {found} classes in {folder}, {COUNT} of {CLASSES} '
                           f'expected; delete {folder.parent} to download it again')
    return out
