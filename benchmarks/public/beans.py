# RP-CLASS - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""Beans (iBean, docs/BENCHMARK.md, G6): bean leaves photographed in Ugandan fields, 3 classes (two diseases and
healthy), one task: leaf disease standing in for defect types, already small (1,295 images).

Data: the Hugging Face dataset AI-Lab-Makerere/beans at revision 27aa014 (2024-01-03, 'Convert dataset to Parquet'),
data/{train,validation,test}-00000-of-00001.parquet: 143,773,054, 18,499,710 and 17,706,070 bytes, each checked
against the SHA-256 Hugging Face publishes for it (its LFS oid). The authors' own links (storage.googleapis.com/ibeans,
which TFDS also uses) answer 403. A row holds the original JPEG (image.bytes: the 128 of the test file are, byte for
byte, the members of data/test.zip of the same revision), its original name (image.path, <class>_<split>.<n>.jpg),
its path in the authors' zips (image_file_path, .../<split>/<class>/<name>) and an integer label. The label names are
in the file's own features metadata: angular_leaf_spot, bean_rust and healthy for 0, 1 and 2; the class folder and
the name prefix of all 1,295 rows agree with them. 1,295 RGB JPEGs of 500 x 500 px, no two with the same bytes:
                        train   validation   test    all
    angular_leaf_spot     345           44     43    432
    bean_rust             348           45     43    436
    healthy               341           44     42    427
(the iBean README counts 1,296, with 428 healthy).
Licence: MIT ('License | MIT', the README of github.com/AI-Lab-Makerere/ibean, whose LICENSE reads 'MIT License,
Copyright (c) 2020 AIR Lab Makerere University'; the dataset card's YAML says mit, its text 'More Information Needed').

Choices:
    test      built: the three splits are pooled and split again (docs/BENCHMARK.md, G6), as the official test has
              only 128 images: round(0.4 x n_c), 171 to 174 per class
    classes   the feature names; each row is written as raw/images/<class>/<original name>, its bytes as they are
              (never re-encoded), and its name must read <class>_<split>.<n>.jpg with the class of its label, which
              checks the mapping again and keeps the name a plain file name
    groups    none given (no plant, field or date): only finalize's look-alikes and near-copies, and there are none
    reader    pyarrow, imported here only; it must be installed where python3 -I sees it (system or virtualenv
              site-packages: -I ignores ~/.local)
    disk      the parquet files are deleted once decoded: 0.36 GB at the peak, 0.18 GB after

Counts (prepared on the GPU machine, 2026-11-06): 1,295 listed, no exact copy, look-alike, near-copy or featureless
image, so every group is one image and none is left out.
    test    angular_leaf_spot 173, bean_rust 174, healthy 171: 518 (423 of the official train, 47 of validation, 48 of
            test)
    val     33, 34 and 33: 100
    pool    226, 228 and 223: 677
    train   100, 101 and 99 per seed: 300

Caveats: smartphone photos taken in the field and labelled there by experts of NaCRRI (the README): one leaf, often
held in a hand, fills most of the frame, with soil, sky or other plants behind it, and the label is the disease the
expert saw, not the state of every leaf in view. No plant or field ids exist, so photos of one plant can sit on both
sides of the split unless they are look-alikes, and none is.
"""
import json
import re

import fetch

NAME = 'beans'
TASKS = (NAME,)
IMGSZ = 224
LICENCE = 'MIT'
SOURCE = 'https://github.com/AI-Lab-Makerere/ibean'
REVISION = '27aa014ce09b193e1a6f58112d4a66e0eddb69c5'
URL = 'https://huggingface.co/datasets/AI-Lab-Makerere/beans/resolve/{}/data/{}-00000-of-00001.parquet'
PARQUET = {'train': (143773054, '7f905a7323966a58e89b8e839ed656bb869fc82d16a3fadc7dce40972a5f8b19'),
           'validation': (18499710, '33f774593d8b31585457b70c224744e9409ffdee4e91a11822b1ebfe8242928f'),
           'test': (17706070, '534a6b0648f585d69b7ec0ad7a7540720d60c8db8106dc6d0508296316f6cb27')}
FILE = re.compile(r'([a-z_]+)_(?:train|val|test)\.\d+\.jpg')
COUNT = 1295


def download(raw):
    """The three parquet files -> raw/images/<class>/<name>, once; they are deleted when all three are decoded."""
    images = raw / 'images'
    if not (images / '.decoded').exists():
        files = [fetch.download(URL.format(REVISION, split), raw / f'{split}.parquet', size=size, sha256=sha)
                 for split, (size, sha) in PARQUET.items()]
        count = sum(_decode(f, images) for f in files)
        (images / '.decoded').write_text(f'{count} images\n')
        for f in files:
            f.unlink()
    return images


def _decode(parquet, images):
    """One parquet file -> images/<class>/<original name>, the JPEG bytes as they are."""
    f = _pyarrow().ParquetFile(parquet)
    names = json.loads(f.schema_arrow.metadata[b'huggingface'])['info']['features']['labels']['names']
    for batch in f.iter_batches(batch_size=64, columns=['image', 'labels']):
        for r in batch.to_pylist():
            label, name = names[r['labels']], r['image']['path']
            m = FILE.fullmatch(name)
            if not m or m.group(1) != label:
                raise RuntimeError(f'beans: {parquet.name} gives {name!r} the label {label}, not a <class>_<split>.<n>'
                                   f'.jpg of that class; check REVISION, this layout is that of 27aa014')
            dst = images / label / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(r['image']['bytes'])
    return f.metadata.num_rows


def _pyarrow():
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise RuntimeError('beans: reading the parquet files needs pyarrow; install it where python3 -I finds it '
                           '(pip install pyarrow, not --user)') from None
    return pq


def rows(folder, task):
    out = [{'path': p, 'label': p.parent.name} for p in sorted(folder.glob('*/*.jpg'))]
    if len(out) != COUNT:
        raise RuntimeError(f'beans: {len(out)} images in {folder}, {COUNT} expected; delete {folder.parent} to '
                           f'download it again')
    return out
