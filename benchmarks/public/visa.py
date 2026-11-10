# RP-CLASS - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""VisA, the Visual Anomaly dataset (Zou et al., "SPot-the-Difference Self-Supervised Pre-training for Anomaly
Detection and Segmentation", ECCV 2022): 12 tasks, one per object, each OK against NOK (an anomalous image, whatever
its defects and however many).

    task         normal + anomalous   size (px)     test: OK + NOK   pool: OK + NOK
    candle        1,000 + 100         1284 x 1168     400 + 40         600 + 60
    capsules        602 + 100         1500 x 1000     241 + 40         361 + 60
    cashew          500 + 100         1274 x 1176     200 + 40         300 + 60
    chewinggum      503 + 100         1342 x 1118     201 + 40         302 + 60
    fryum           500 + 100         1500 x 1000     200 + 40         300 + 60
    macaroni1     1,000 + 100         1500 x 1000     400 + 40         600 + 60
    macaroni2     1,000 + 100         1500 x 1000     400 + 40         600 + 60
    pcb1          1,004 + 100         1404 x 1070     402 + 40         602 + 60
    pcb2          1,001 + 100         1404 x 1070     400 + 40         601 + 60
    pcb3          1,006 + 100         1562 x 960      402 + 40         604 + 60
    pcb4          1,005 + 100         1358 x 1104     402 + 40         603 + 60
    pipe_fryum      500 + 100         1300 x 1154     200 + 40         300 + 60

Data: VisA_20220922.tar, the link of the spot-diff README (1,929,840,640 bytes; the SHA-256 pinned below was computed
on a download that matched S3's own multipart ETag): 10,821 RGB JPEGs, 9,621 normal and 1,200 anomalous, one size
per object. Licence: CC BY 4.0 for the data (README "License": "The data is released under the CC BY 4.0 license.";
LICENSE-DATASET and NOTICE say the same); the repository's code is Apache-2.0, and none of it is used here.

Splits: the authors' supervised split, split_csv/2cls_highshot.csv of github.com/amazon-science/spot-diff at commit
2a692ab (the repository's last, 2022-09-22; the file has not changed since it was added in 6dbf63c). "60%/40% normal
and anomalous images are assigned to train/test" (the paper): its test rows are the official test, used whole; its
train rows are the pool that val and the training draws come from. The tar ships the same file byte for byte, and the
file lists exactly the tar's 10,821 images, each label agreeing with its folder (Normal / Anomaly) and with the tar's
image_anno.csv (normal, or the names of the image's defects). The other two splits (1cls: training on normal images
only; 2cls_fewshot: 20 NOK per object in training) are not used.

Prepared on the GPU machine, 2026-11-06 (OK + NOK; every test whole, as in the table; train the same size each seed):
    candle                    val 91 + 9    train 273 + 27 of the 509 + 50 left
    macaroni1, macaroni2      val 91 + 9    train 273 + 27 of the 509 + 51 left
    pcb1, pcb2, pcb3, pcb4    val 91 + 9    train 273 + 27 of the 511, 510, 513 and 512 + 51 left
    capsules                  val 86 + 14   train 257 + 43 of the 275 + 46 left
    cashew, fryum             val 83 + 17   train all the 217 + 43 left
    chewinggum                val 83 + 17   train all the 219 + 43 left
    pipe_fryum                val 83 + 17   train all the 195 + 42 left
In those last four the three seeds train on the same images and differ only in the run seed. No task has an exact
copy, a featureless image or label noise.

Caveats: the anomalies were staged ("defects were manually generated to produce realistic anomalies", the paper),
4 to 9 defect types per object, all NOK. Each object lies in about the same place on a plain background, so distinct
pieces look alike once shrunk: at the 32 x 32 of common.py's look-alike test a pipe fryum is 1.1 to 2.5 / 255 from its
nearest other piece (1st to 90th percentile), while the repeat captures of one scene found (candle 0129 and 0130,
pipe_fryum 331 and 332) are 0.25 and 0.47 apart. Look-alikes only join a group: 438 joins in pipe_fryum (an image
joining an earlier look-alike's group; groups of up to 27, some with both classes), 249 in candle and 184 in pcb3
(groups of up to 8), 37 in pcb4 (up to 6), and at most 11 pairs in each other product. They do not count against
the official test, so val and the pool hold look-alikes of test pieces; val takes whole groups, which leaves 23
pipe_fryum pool images out. Near-copies, also within 1.5 at 128 x 128, are 8 pairs of
consecutive captures (candle 5, pipe_fryum 2, macaroni2 1), none with two labels; one ties candle's pool image
Anomaly 016 to the test's Anomaly 015, and it is dropped. The margin is thin: the farthest of them (candle 0928 and
0929, pipe_fryum 438 and 439) are 1.43 apart at 128 px, the closest other pair 1.57 (pipe_fryum 172 and 191). No
capture or part ids are given: nothing else is grouped.

Choices: only the images are extracted (Data/Images/; the masks, image_anno.csv and the tar's copies of the split
files are not needed), at their native size: the benchmark resizes them to 448 (IMGSZ). The tar is deleted once
extracted, so the download peaks at 3.8 GB (tar and images) and leaves 1.9 GB.
"""
import csv
from pathlib import Path

import fetch

NAME = 'visa'
TASKS = ('candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3',
         'pcb4', 'pipe_fryum')
IMGSZ = 448
LICENCE = 'CC BY 4.0'
SOURCE = 'https://github.com/amazon-science/spot-diff'
TAR = ('https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar', 1929840640,
       '2eb8690c803ab37de0324772964100169ec8ba1fa3f7e94291c9ca673f40f362')
SPLIT = ('https://raw.githubusercontent.com/amazon-science/spot-diff/2a692ab575001cbde74d402d897a7286086c6199/'
         'split_csv/2cls_highshot.csv', 658163, 'c3331eede15f2da8a75b380d4fbfb24f75ee036ec9a25147eca898da6e972f09')
LABELS = {'normal': 'ok', 'anomaly': 'nok'}
POSITIVE = 'nok'


def download(raw):
    """raw/2cls_highshot.csv and raw/VisA/<object>/Data/Images/{Normal,Anomaly}/*.JPG."""
    raw = Path(raw)
    url, size, sha256 = SPLIT
    fetch.download(url, raw / '2cls_highshot.csv', size=size, sha256=sha256)
    archive = raw / 'VisA_20220922.tar'
    if not (raw / 'VisA' / '.extracted').exists():
        url, size, sha256 = TAR
        fetch.download(url, archive, size=size, sha256=sha256)
        fetch.extract(archive, raw / 'VisA', keep=lambda name: '/Data/Images/' in name)
    archive.unlink(missing_ok=True)
    return raw


def rows(folder, task):
    folder = Path(folder)
    with open(folder / '2cls_highshot.csv', newline='') as f:
        listed = [r for r in csv.DictReader(f) if r['object'] == task]
    out = []
    for r in listed:
        path = folder / 'VisA' / r['image']
        if r['label'] not in LABELS or r['split'] not in ('train', 'test') or not path.is_file():
            raise RuntimeError(f"visa: {r['image']} ({r['split']}, {r['label']}) is not as expected; delete "
                               f'{folder} to download again')
        out.append({'path': path, 'label': LABELS[r['label']], 'split': r['split']})
    if not out:
        raise RuntimeError(f"visa: no image of {task!r} in 2cls_highshot.csv; the tasks are {', '.join(TASKS)}")
    return out
