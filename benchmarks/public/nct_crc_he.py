# RP-CLASS - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""NCT-CRC-HE (Kather 2019; docs/BENCHMARK.md, G3): tissue types in H&E colorectal histology patches, 9 classes,
one task.

Data: two zips of Zenodo record 1214456 (Kather, Halama and Marx; the paper is Kather et al., 'Predicting survival
from colorectal cancer histology slides using deep learning: A retrospective multicenter study', PLOS Medicine
16(1):e1002730, 2019):
    NCT-CRC-HE-100K.zip   11,690,284,003 bytes   100,000 patches of 86 slides, colour-normalized (Macenko)
    CRC-VAL-HE-7K.zip        800,276,929 bytes     7,180 patches of 50 patients, 'no overlap with patients in
                                                   NCT-CRC-HE-100K' (the record)
Each holds <set>/<class>/<class>-<8 letters>.tif (the 7K names add 'TCGA-') and nothing else; every patch is a
224 x 224 RGB TIFF (PackBits) at 0.5 um/px. Both central directories, and members of every class read through HTTP
ranges, were checked on 2026-11-06:
    class    ADI     BACK    DEB     LYM     MUC     MUS     NORM    STR     TUM
    100K     10,407  10,566  11,512  11,557  8,896   13,536  8,763   10,446  14,317
    7K       1,338   847     339     634     1,035   592     741     421     1,233
(adipose, background, debris, lymphocytes, mucus, smooth muscle, normal colon mucosa, cancer-associated stroma,
colorectal adenocarcinoma epithelium). Zenodo publishes MD5 sums only: the sizes are pinned, the 100K zip's MD5 too
(fetch checks it), the 7K zip's SHA-256 (from a download whose MD5 matched Zenodo's, 2fd1651b4f94ebd818ebf90ad2b6ce06,
on 2026-11-06), and zipfile checks the CRC-32 of every 100K member it extracts.
Licence: CC BY 4.0, the record's licence (cc-by-4.0; open access, no login).

Choices:
    test      the 7K set whole: the only patient-disjoint split this data has, and the record's own pairing (it 'can
              be used as a validation set for models trained on the larger data set')
    pool      the colour-normalized 100K set, this benchmark's choice; NCT-CRC-HE-100K-NONORM (11,726,652,899 bytes)
              is the same slides without the normalization, tiled again
    presample 400 patches of each class at random (common.presample over the sorted member names), 3,600 of 100,000:
              the draws need at most 14 val + 43 train of a class (TUM, share 0.143); the shares stay the 100K's
    download  the 100K zip whole, in one request; only its presampled members are extracted (fetch.extract keep=),
              then it is deleted. Its member names stay in NCT-CRC-HE-100K.txt, from which rows() takes the shares
              and draws the same presample again
    classes   the archive's folder codes, short and filesystem-safe; sorted, they are in the record's order
    groups    none given: the names carry no slide or patient id; only finalize's look-alikes and near-copies
    disk      11.7 GB of zip + 1.1 GB of 7K patches + 0.55 GB of presample at the peak, 1.6 GB after

Counts (prepared on the GPU machine, 2026-11-06): test 7,180, the 7K set whole; val 101 (ADI 10, BACK 11, DEB 12,
LYM 12, MUC 9, MUS 14, NORM 9, STR 10, TUM 14); each training draw 301 (31, 32, 35, 35, 27, 41, 26, 31, 43); pool
3,499. Dedup saw 10,780 patches, the 7K set and the presample: no exact copy, look-alike or near-copy, so no group,
no pool patch tied to the test and none left out. 191 of them are featureless (a 32 x 32 thumbnail standard deviation
under 2 grey levels), checked for exact copies only.

Caveats: val and the training draws come from the same 86 slides (no slide id to group by), so val is optimistic and
only the test measures new patients, the same for every method. The 100K set's normal classes were 'augmented with
non-tumorous regions from gastrectomy specimen' (the record); the 7K set has colorectal cancer patients only. Dedup
sees the test and the presample only: a 100K patch left out never reaches a split, but a copy of a test patch under
another label would go unnoticed there (none is expected between patient-disjoint sets).
"""
import re
import zipfile

import common
import fetch

NAME = 'nct_crc_he'
TASKS = (NAME,)
IMGSZ = 224
LICENCE = 'CC BY 4.0'
SOURCE = 'https://zenodo.org/api/records/1214456'

FILES = 'https://zenodo.org/records/1214456/files'
TEST = ('CRC-VAL-HE-7K.zip', 800276929, '1bcb631804a7136dbea427bc83826f2b6b6f61939470d8aec4aede93bfaaf1af', None)
POOL = ('NCT-CRC-HE-100K.zip', 11690284003, '60793a1e3fc27aa192ac3cd5d590d0ac2c1d39405093b7a411d7f643c6d0cc48',
        '6fd702d11df6292bc054397ae038a464')   # SHA-256 of the checked download, and the MD5 Zenodo gives
CLASSES = ('ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM')
MEMBER = re.compile(r'NCT-CRC-HE-100K/(' + '|'.join(CLASSES) + r')/\1-[A-Z]{8}\.tif')     # .../ADI/ADI-AAAMHQMK.tif
LISTING = 'NCT-CRC-HE-100K.txt'
PER_CLASS = 400
COUNTS = (7180, 100000)


def download(raw):
    """CRC-VAL-HE-7K.zip -> raw/test, whole; NCT-CRC-HE-100K.zip -> raw/NCT-CRC-HE-100K.txt (its member names) and
    raw/pool (its presampled patches); once, each zip deleted when its patches are out."""
    if not (raw / 'test' / '.extracted').exists():
        fetch.extract(_zip(raw, *TEST), raw / 'test')
    (raw / TEST[0]).unlink(missing_ok=True)
    if not (raw / 'pool' / '.extracted').exists():
        archive = _zip(raw, *POOL)
        with zipfile.ZipFile(archive) as z:
            (raw / LISTING).write_text(''.join(f'{name}\n' for name in z.namelist()))
        keep = {r['member'] for r in common.presample(_listing(raw), PER_CLASS)}
        fetch.extract(archive, raw / 'pool', keep=keep.__contains__)
    (raw / POOL[0]).unlink(missing_ok=True)
    return raw


def _zip(raw, name, size, sha256, md5):
    return fetch.download(f'{FILES}/{name}?download=1', raw / name, size=size, sha256=sha256, md5=md5)


def _listing(folder):
    """The 100K patches as [{'member', 'label'}], sorted by member name."""
    out = []
    for name in sorted((folder / LISTING).read_text().split('\n')):
        m = MEMBER.fullmatch(name)
        if m:
            out.append({'member': name, 'label': m.group(1)})
    return out


def rows(folder, task):
    test = [{'path': p, 'label': c.lower(), 'split': 'test'} for c in CLASSES
            for p in sorted((folder / 'test' / 'CRC-VAL-HE-7K' / c).glob('*.tif'))]
    listing = _listing(folder)
    if (len(test), len(listing)) != COUNTS:
        raise RuntimeError(f'nct_crc_he: {len(test)} test patches and {len(listing)} 100K names in {folder}, '
                           f'{COUNTS[0]} and {COUNTS[1]} expected; delete {folder} to download it again')
    pool = [{'path': folder / 'pool' / r['member'], 'label': r['label'].lower(), 'split': 'train'}
            for r in common.presample(listing, PER_CLASS)]
    missing = sum(not r['path'].is_file() for r in pool)
    if missing:
        raise RuntimeError(f'nct_crc_he: {missing} presampled patches are not in {folder}/pool (was PER_CLASS '
                           f'changed?); delete that folder to download and extract them again')
    return test + pool, {c.lower(): f for c, f in common.shares(listing).items()}
