# RP-CLASS - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""BloodMNIST, 224 px (MedMNIST v2; docs/BENCHMARK.md, G4): peripheral blood cell types, one cell per image, 8
classes, one task.

Data: bloodmnist_224.npz of Zenodo record 10519652 (MedMNIST+, v3.0), 1,540,731,655 bytes; its MD5 matched the one
Zenodo publishes (b718ff6835fcbdb22ba9eacccd7b2601, also medmnist/info.py's MD5_224) on 2026-11-06, and the SHA-256
pinned below is that download's. The npz holds {train,val,test}_images (n x 224 x 224 x 3, uint8) and _labels (n x 1,
0 to 7), read with numpy (allow_pickle=False). The images are the 17,092 of Acevedo et al. (360 x 363 px JPEGs of a
CellaVision DM96, Hospital Clinic of Barcelona; Mendeley Data doi:10.17632/snkd93bnjr.1), which MedMNIST centre-crops
to 200 x 200 and resizes to 224. Cite MedMNIST v2 (Yang et al., 'MedMNIST v2 - A large-scale lightweight benchmark for
2D and 3D biomedical image classification', Scientific Data 10:41, 2023; the record asks for v1, ISBI 2021, too) and
Acevedo et al., 'A dataset of microscopic peripheral blood cell images for development of automatic recognition
systems', Data in Brief 30:105474, 2020. The official split (7:1:2 by image), counted in the npz:
              basophil  eosinophil  erythroblast  ig     lymphocyte  monocyte  neutrophil  platelet  total
    train     852       2,181       1,085         2,026  849         993       2,330       1,643     11,959
    val       122       312         155           290    122         143       333         235       1,712
    test      244       624         311           579    243         284       666         470       3,421
Licence: CC BY 4.0, the record's licence (cc-by-4.0) and text ('The MedMNIST dataset is licensed under Creative
Commons Attribution 4.0 International (CC BY 4.0), except DermaMNIST'), and medmnist/info.py's for bloodmnist; the
source data are CC BY 4.0 too (Mendeley's data licence).

Choices:
    classes   MedMNIST's labels 0 to 7, named as Acevedo's folders: basophil, eosinophil, erythroblast, ig (immature
              granulocytes: promyelocytes, myelocytes and metamyelocytes), lymphocyte, monocyte, neutrophil, platelet;
              alphabetical, so the sorted order is MedMNIST's
    test      the official test whole
    pool      the official train and val together, 13,671 images: the benchmark draws its own val of about 100 images
    files     every image becomes a PNG file images/<split>/<class>/<index>.png, image <index> of <split>_images:
              17,092 files, 0.95 GB, 3 minutes. None is presampled away, as dedup must see them all: the npz holds 17
              pairs of identical images, 6 of them between the test and the official train or val, and one of these
              has two labels (train 1018 eosinophil, test 3126 neutrophil), which finalize drops on both sides. A
              presample of 400 per class, tried first, left that test image in with its second label.
    groups    none given (the source has no patient ids); only finalize's look-alikes and near-copies
    disk      1.54 GB of npz + 0.95 GB of PNG at the peak, 0.95 GB after; train_images takes 1.8 GB of memory

Counts (prepared on the GPU machine, 2026-11-06): dedup dropped 16 exact copies and the 2 images with two labels, and
found no featureless image, look-alike or near-copy: no group, no pool image tied to the test, none left out. Test
3,420 (the official 3,421 but test 3126), val 100 (7, 18, 9, 17, 7, 8, 20, 14), each training draw 300 (21, 55, 27,
51, 21, 25, 59, 41), pool 13,554.

Caveats: MedMNIST split by image and the source has no patient ids, so one person's cells can be in a training draw
and in the test. The 224 px images are the 200 px crops upsampled (x 1.12). One cell on a plain background: the
closest of the general sets to a part image.
"""
import numpy as np
from PIL import Image

import fetch

NAME = 'bloodmnist'
TASKS = (NAME,)
IMGSZ = 224
LICENCE = 'CC BY 4.0'
SOURCE = 'https://zenodo.org/api/records/10519652'
URL = 'https://zenodo.org/records/10519652/files/bloodmnist_224.npz?download=1'
NPZ, SIZE = 'bloodmnist_224.npz', 1540731655
SHA256 = '534bfb674fcfcfecf691aaca942ced2181d3affae9528a645ae51d12e9f3e1a1'
CLASSES = ('basophil', 'eosinophil', 'erythroblast', 'ig', 'lymphocyte', 'monocyte', 'neutrophil', 'platelet')
SPLITS = ('train', 'val', 'test')
COUNT = 17092


def download(raw):
    """bloodmnist_224.npz -> raw/images/<split>/<class>/<index>.png, once; the npz is deleted when they are out."""
    images = raw / 'images'
    if not (images / '.converted').exists():
        _convert(fetch.download(URL, raw / NPZ, size=SIZE, sha256=SHA256), images)
    (raw / NPZ).unlink(missing_ok=True)
    return images


def _convert(npz, images):
    count = 0
    with np.load(npz, allow_pickle=False) as d:
        for s in SPLITS:
            labels, arrays = d[f'{s}_labels'][:, 0], d[f'{s}_images']      # one array at a time
            for i, y in enumerate(labels):
                path = images / s / CLASSES[y] / f'{i:05d}.png'
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(arrays[i]).save(path)
                count += 1
    (images / '.converted').write_text(f'{count} images from {npz.name}\n')


def rows(folder, task):
    out = [{'path': p, 'label': p.parent.name, 'split': p.parent.parent.name} for p in sorted(folder.glob('*/*/*.png'))]
    if len(out) != COUNT:
        raise RuntimeError(f'bloodmnist: {len(out)} images in {folder}, {COUNT} expected; delete {folder.parent} to '
                           f'download it again')
    return out
