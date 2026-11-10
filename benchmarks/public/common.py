# RP-CLASS - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""The public benchmark protocol (docs/BENCHMARK.md), the same for every dataset: frozen splits, then per seed a
folder layout that RPCLASS.train (and YOLO-cls) read directly.

A dataset module (benchmarks/public/<name>.py) only says where its images are and what they are: `rows(folder, task)`
returns one dict per image,
    path     the image file
    label    its class name
    group    optional: an id shared by images that must stay together (the same source photo, board, capture burst);
             a group may hold several classes (a photo's cracked and uncracked tiles)
    split    optional: 'test' for the official (or the authors') test split; anything else is pool
    stratum  optional: a sub-population the built splits keep in proportion within each class (SDNET's structure)
or (rows, share) when the rows are a random subset of a larger listing (benchmarks too big to prepare whole): share =
{label: fraction of the whole}, from the full listing, so the draws below keep the true class shares.

`finalize` then does the rest, identically for all:

1. Duplicates, in two tiers. Exact copies (same bytes) are kept once, preferring an official test copy; copies that
   disagree on the label are all dropped. Look-alikes: 64-bit dHashes at most HASH_RADIUS bits apart and 32 x 32 grey
   thumbnails at most PRE_MAD apart (a mean difference out of 255): the same photo cropped, shifted or shot again
   (Caltech's repeated photos and drawings sit up to 2.7, DeepWeeds' retakes of one scene up to 3), but also distinct
   pieces on one plain background (VisA, from 1.1) and distinct textures (DAGM). Near-copies: look-alikes whose 128 x
   128 thumbnails are also at most TWIN_MAD apart, the same picture re-encoded or resized (JPEG 75-95 and resized
   copies measured at 0.1-1.3, 90th percentile, 1.8 at most), which DAGM's textures (from 7.6) never are and VisA's
   distinct pieces barely are not (from 1.57 over all pairs; the frozen data's near-copies reach 1.43); measured on
   real data, docs/EXPERIMENTS.md, 2026-11-06. Look-alikes only join a group, so that the
   splits built here keep them on one side. A near-copy pair that disagrees on the label is dropped as label noise,
   that pair only, except the images of an official test, which stays whole. Featureless images (a 32 x 32 thumbnail
   whose standard deviation is under FLAT_STD grey levels: open water, empty background) are neither: they all look
   alike without being the same picture, and chained into groups of hundreds (the first rule joined 867 of EuroSAT's
   flat water tiles) they would make a class's whole test out of them; exact copies of them are still found.
2. Test. The official test split, whole, when there is one; pool images tied to a test image (the same module group,
   or a near-copy) are dropped. Look-alikes do not count here: the authors' split is theirs, and a VisA piece that
   looks like another is still another piece. Else a test drawn by group: round(TEST_FRACTION x n_c) images of each
   class c, at most TEST_CAP, in proportion to the strata: whole groups are taken at random while they hold a class
   (and stratum) that still lacks images, then exactly the wanted number is drawn from them; their other images are
   used nowhere, so a group never crosses splits.
3. Shares. share_c: the class shares of the images train and val come from: the whole dataset when the test is
   built, everything but the official test otherwise (NCT-CRC-HE: the 100K set).
4. Val. From the rest, by group as the test: max(VAL_MIN, round(VAL_TOTAL x share_c)) images of each class (all of
   them if fewer); with strata, each (class, stratum) share is rounded on its own, so a class may get one more or
   one less (SDNET's val: 16 nok where the formula gives 15). The built test is split over the strata the same way.
5. Pool: everything else. Per seed s, train is a random draw from the pool of min(pool_c, max(TRAIN_MIN,
   round(TRAIN_TOTAL x share_c))) images of each class: a small dataset of about 300 images.
6. Written once: <out>/<task>/manifest.csv (every image in test, val or a training draw: its file, class, group,
   split, which seeds train on it, SHA-256 and where it came from), info.json (the counts and the provenance), and
   <out>/<task>/s<seed>/{train,val,test}/<class>/<file> as hard links (copies across file systems). The manifest
   keeps the module's own group next to finalize's (which adds the look-alikes and near-copies). A JPEG whose
   EXIF orientation tag turns the image is written with the tag set to 1: OpenCV (YOLO's reader) applies the tag,
   PIL (RP-CLASS's) does not, and both must see the same pixels.

Val and test are the same for every seed and every method; only the training draw (and the run seed) changes. One
split seed (SPLIT_SEED) makes everything; the same listing always gives the same splits. A class left without test,
val or training images is an error: such a task cannot be scored.
"""
import csv
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

TEST_FRACTION, TEST_CAP = 0.4, 500
VAL_TOTAL, VAL_MIN = 100, 5
TRAIN_TOTAL, TRAIN_MIN = 300, 20
SEEDS = (0, 1, 2)
SPLIT_SEED = 1234                  # the splits' own seed, apart from the run seeds
HASH_RADIUS, PRE_MAD, TWIN_MAD, FLAT_STD = 4, 3.0, 1.5, 2.0
_POPCOUNT = np.array([bin(i).count('1') for i in range(256)], np.uint8)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(path):
    """(64-bit dHash, 32 x 32 grey thumbnail) of an image."""
    with Image.open(path) as im:
        im.draft('L', (128, 128))          # JPEG: decode at a fraction of the size, enough for both
        g = im.convert('L')
        small = np.asarray(g.resize((9, 8), Image.BILINEAR), np.int16)
        thumb = np.asarray(g.resize((32, 32), Image.BILINEAR), np.uint8)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int(''.join('1' if b else '0' for b in bits), 2), thumb


def thumbnail(path, size=128):
    """size x size grey thumbnail of an image (the near-copy confirmation)."""
    with Image.open(path) as im:
        im.draft('L', (2 * size, 2 * size))
        return np.asarray(im.convert('L').resize((size, size), Image.BILINEAR), np.uint8)


def near_copies(a, b):
    """Whether two image files are near-copies under the rule finalize applies (the module docstring)."""
    (ha, ta), (hb, tb) = fingerprint(a), fingerprint(b)
    return (bin(ha ^ hb).count('1') <= HASH_RADIUS and _mad(ta, tb) <= PRE_MAD
            and _mad(thumbnail(a), thumbnail(b)) <= TWIN_MAD)


def _mad(a, b):
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


def shares(rows):
    """{label: fraction of the rows}."""
    count = Counter(r['label'] for r in rows)
    return {c: n / len(rows) for c, n in count.items()}


def presample(rows, per_class, seed=SPLIT_SEED):
    """At most `per_class` rows of each label, at random (kept in listing order): for pools far bigger than the
    benchmark draws from, before extracting or hashing anything. Pass shares(all rows) along with the result."""
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for r in rows:
        by[r['label']].append(r)
    out = []
    for label in sorted(by):
        rs = by[label]
        keep = sorted(rng.permutation(len(rs))[:per_class])
        out += [rs[i] for i in keep]
    return out


def _bits(x):
    """The number of set bits of each uint64."""
    if hasattr(np, 'bitwise_count'):
        return np.bitwise_count(x)
    return _POPCOUNT[x.view(np.uint8)].reshape(-1, 8).sum(1)


class _Groups:
    """Union-find over row indices."""

    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i, j):
        self.parent[self.find(i)] = self.find(j)


def _dedup(rows, log):
    """Drops exact copies and label noise; sets each row's 'group' (module group + look-alikes + near-copies) and 'tie'
    (module group + near-copies: what binds a pool image to an official test image)."""
    by_sha = defaultdict(list)
    for i, r in enumerate(rows):
        r['sha256'] = sha256(r['path'])
        by_sha[r['sha256']].append(i)
    keep, conflicts, copies = [], 0, 0
    for idx in by_sha.values():
        if len({rows[i]['label'] for i in idx}) > 1:
            conflicts += len(idx)
            continue
        keep.append(next((i for i in idx if rows[i].get('split') == 'test'), idx[0]))
        copies += len(idx) - 1
    rows = [rows[i] for i in sorted(keep)]
    n, big = len(rows), {}

    def big_thumb(i):
        if i not in big:
            big[i] = thumbnail(rows[i]['path'])
        return big[i]

    loose, near, alike, twins, noise = _Groups(n), _Groups(n), 0, 0, set()
    hashes, thumbs, leader = np.zeros(n, np.uint64), np.zeros((n, 32, 32), np.uint8), np.zeros(n, bool)
    flat = np.zeros(n, bool)
    for i, r in enumerate(rows):
        hashes[i], thumbs[i] = fingerprint(r['path'])
        flat[i] = thumbs[i].std() < FLAT_STD
        if flat[i]:
            continue
        close = np.flatnonzero((_bits(hashes[:i] ^ hashes[i]) <= HASH_RADIUS) & ~flat[:i])
        if close.size:
            mad = np.abs(thumbs[close].astype(np.int16) - thumbs[i]).mean(axis=(1, 2))
            close = close[np.argsort(mad, kind='stable')][:int((mad <= PRE_MAD).sum())]
        heads = close[leader[close]]
        if heads.size:            # joins the closest look-alike cluster (its first image), not every look-alike
            loose.union(i, heads[0])
            alike += 1
        else:
            leader[i] = True
        for j in close[:8]:       # the closest few look-alikes, confirmed at 128 px
            if _mad(big_thumb(j), big_thumb(i)) <= TWIN_MAD:
                near.union(i, j)
                twins += 1
                if rows[i]['label'] != rows[j]['label']:      # label noise; an official test stays whole
                    noise.update((i, j))
                break
    drop = {i for i in noise if rows[i].get('split') != 'test'}
    group, tie, first = _Groups(n), _Groups(n), {}
    for i, r in enumerate(rows):
        r['declared'] = '' if r.get('group') in (None, '') else str(r['group'])
        if r['declared']:
            j = first.setdefault(r['declared'], i)
            group.union(i, j)
            tie.union(i, j)
        group.union(i, loose.find(i))
        group.union(i, near.find(i))
        tie.union(i, near.find(i))
    for i, r in enumerate(rows):
        r['group'], r['tie'] = f'g{group.find(i)}', tie.find(i)
    rows = [r for i, r in enumerate(rows) if i not in drop]
    sizes = Counter(r['group'] for r in rows)
    log.update(exact_copies_dropped=copies, conflicting_copies_dropped=conflicts, featureless=int(flat.sum()),
               look_alikes_grouped=alike,
               near_copies_found=twins, near_copies_with_two_labels=len(noise),
               images_dropped_for_label_conflicts=len(drop), largest_group=max(sizes.values(), default=0))
    return rows


def _allocate(want, rows, idx):
    """want {label: n} -> {(label, stratum): n}, in proportion to the strata of rows[idx] within each label."""
    count = Counter((rows[i]['label'], rows[i].get('stratum', '')) for i in idx)
    per_label = Counter(rows[i]['label'] for i in idx)
    out = {}
    for (label, stratum), n in sorted(count.items()):
        out[(label, stratum)] = round(want.get(label, 0) * n / per_label[label])
    return out


def _key(r):
    return r['label'], r.get('stratum', '')


def _draw(rows, idx, want, rng):
    """Whole groups of rows[idx] at random, while a (label, stratum) they hold still lacks images; then, within the
    groups taken, exactly `want` images of each (label, stratum), at random. Returns (taken, left): `left` are the other
    images of the groups taken, which no later split may use (their group is in this one)."""
    groups = defaultdict(list)
    for i in idx:
        groups[rows[i]['group']].append(i)
    order = sorted(groups)
    rng.shuffle(order)
    have, chosen = Counter(), []
    for gid in order:
        keys = Counter(_key(rows[i]) for i in groups[gid])
        if any(have[k] < want.get(k, 0) for k in keys):
            chosen.append(gid)
            have.update(keys)
            if all(have[k] >= n for k, n in want.items()):
                break
    by_key = defaultdict(list)
    for gid in chosen:
        for i in groups[gid]:
            by_key[_key(rows[i])].append(i)
    taken = set()
    for k in sorted(by_key):
        members = by_key[k]
        taken.update(members[j] for j in sorted(rng.permutation(len(members))[:want.get(k, 0)]))
    left = {i for gid in chosen for i in groups[gid]} - taken
    return taken, left


def finalize(task, rows, out, imgsz, share=None, root=None, meta=None, seeds=SEEDS):
    """rows -> manifest, info and the per-seed layouts of one task (see the module docstring). root: the folder the
    paths are written relative to in the manifest; meta: provenance written into info.json (licence, source,
    downloads)."""
    out = Path(out) / task
    if (out / 'info.json').exists():
        print(f'{task}: {out} exists, left as it is', flush=True)
        return json.loads((out / 'info.json').read_text())
    rng = np.random.default_rng(SPLIT_SEED)
    log = {'images_listed': len(rows)}
    rows = sorted((dict(r) for r in rows), key=lambda r: str(r['path']))
    rows = _dedup(rows, log)
    labels = sorted({r['label'] for r in rows})
    official = [i for i, r in enumerate(rows) if r.get('split') == 'test']
    if official:
        test = set(official)
        tied = {rows[i]['tie'] for i in test}
        rest = [i for i, r in enumerate(rows) if i not in test and r['tie'] not in tied]
        log['pool_images_tied_to_the_test_dropped'] = len(rows) - len(test) - len(rest)
    else:
        n = Counter(r['label'] for r in rows)
        want = {c: min(TEST_CAP, round(TEST_FRACTION * n[c])) for c in labels}
        test, left = _draw(rows, range(len(rows)), _allocate(want, rows, range(len(rows))), rng)
        rest = [i for i in range(len(rows)) if i not in test and i not in left]
        log['left_out_with_test_groups'] = len(left)
    if share is None:
        share = shares([rows[i] for i in rest] if official else rows)
    want = {c: max(VAL_MIN, round(VAL_TOTAL * share.get(c, 0))) for c in labels}
    val, left = _draw(rows, rest, _allocate(want, rows, rest), rng)
    pool = [i for i in rest if i not in val and i not in left]
    log['left_out_with_val_groups'] = len(left)
    pool_by = defaultdict(list)
    for i in pool:
        pool_by[rows[i]['label']].append(i)
    train = {}
    for s in seeds:
        r_s = np.random.default_rng([SPLIT_SEED, s])
        train[s] = set()
        for c in labels:
            k = min(len(pool_by[c]), max(TRAIN_MIN, round(TRAIN_TOTAL * share.get(c, 0))))
            train[s].update(pool_by[c][j] for j in r_s.permutation(len(pool_by[c]))[:k])
    for name, idx in [('test', test), ('val', val)] + [(f'train s{s}', train[s]) for s in seeds]:
        empty = [c for c in labels if not any(rows[i]['label'] == c for i in idx)]
        if empty:
            raise RuntimeError(f'{task}: {name} has no image of {", ".join(empty)}; such a task cannot be scored, see '
                               f'what the dedup dropped: {log}')
    used = sorted(test | val | set().union(*train.values()))
    names = {i: f'{i:06d}{Path(rows[i]["path"]).suffix.lower()}' for i in used}
    out.mkdir(parents=True, exist_ok=True)
    split_of = {i: 'test' for i in test} | {i: 'val' for i in val} | {i: 'pool' for i in pool}
    with open(out / 'manifest.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['file', 'label', 'group', 'module_group', 'split', *(f's{s}' for s in seeds), 'sha256', 'source'])
        for i in used:
            r = rows[i]
            src = Path(r['path'])
            if root is not None:
                src = src.resolve().relative_to(Path(root).resolve())
            w.writerow([names[i], r['label'], r['group'], r['declared'], split_of[i],
                        *(int(i in train[s]) for s in seeds), r['sha256'], src.as_posix()])
    count = lambda idx: dict(sorted(Counter(rows[i]['label'] for i in idx).items()))  # noqa: E731
    info = {'task': task, 'imgsz': imgsz, 'classes': labels, **(meta or {}),
            'protocol': {'test': 'official' if official else f'built: {TEST_FRACTION} of each class, at most '
                                                              f'{TEST_CAP}', 'val': [VAL_TOTAL, VAL_MIN],
                         'train': [TRAIN_TOTAL, TRAIN_MIN], 'split_seed': SPLIT_SEED,
                         'near_copy': {'hash_radius': HASH_RADIUS, 'mad_32px': PRE_MAD, 'mad_128px': TWIN_MAD,
                                       'featureless_std': FLAT_STD}},
            'dedup': log, 'share': {c: round(share.get(c, 0), 6) for c in labels},
            'test': count(test), 'val': count(val), 'pool': count(pool),
            'train': {f's{s}': count(train[s]) for s in seeds}}
    first = {}                                  # an image placed once, then hard-linked to its other places
    for s in seeds:
        for name, idx in (('train', train[s]), ('val', val), ('test', test)):
            for c in labels:                    # every class folder in every split, as export tools often write them
                (out / f's{s}' / name / _safe(c)).mkdir(parents=True, exist_ok=True)
            for i in idx:
                dst = out / f's{s}' / name / _safe(rows[i]['label']) / names[i]
                first.setdefault(i, (_place(rows[i]['path'], dst), dst))
                if first[i][1] != dst:
                    _link(first[i][1], dst)
    log['jpeg_orientation_tags_cleared'] = sum(cleared for cleared, _ in first.values())
    (out / 'info.json').write_text(json.dumps(info, indent=2, ensure_ascii=False))
    print(f"{task}: {len(labels)} classes · test {len(test)} · val {len(val)} · pool {len(pool)} · train "
          f"{'/'.join(str(len(train[s])) for s in seeds)} · dedup {log}", flush=True)
    return info


def _safe(label):
    """A class name as a folder name."""
    return str(label).replace('/', '_').replace('\\', '_').strip() or 'unnamed'


def _place(src, dst):
    """src at dst: a hard link, or a copy with the JPEG orientation tag set to 1 when it turns the image. Returns
    whether the tag was cleared."""
    if dst.exists():
        return False
    fixed = upright_tag(src) if Path(src).suffix.lower() in ('.jpg', '.jpeg') else None
    if fixed is not None:
        dst.write_bytes(fixed)
        return True
    _link(src, dst)
    return False


def _link(src, dst):
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def upright_tag(path):
    """The bytes of a JPEG whose EXIF orientation tag is not 1, with that tag set to 1 (nothing else changes); None
    for any other file."""
    with open(path, 'rb') as f:
        head = f.read(1 << 17)                   # the EXIF segment comes first, and is at most 64 kB
    if head[:2] != b'\xff\xd8':
        return None
    pos = 2
    while pos + 4 <= len(head) and head[pos] == 0xFF:
        marker, size = head[pos + 1], int.from_bytes(head[pos + 2:pos + 4], 'big')
        if marker in (0xD9, 0xDA):              # end of image, start of the scan: no EXIF
            return None
        if marker == 0xE1 and head[pos + 4:pos + 10] == b'Exif\x00\x00':
            tiff = pos + 10
            order = {b'II': 'little', b'MM': 'big'}.get(head[tiff:tiff + 2])
            if order is None:
                return None
            ifd = tiff + int.from_bytes(head[tiff + 4:tiff + 8], order)
            for k in range(int.from_bytes(head[ifd:ifd + 2], order)):
                e = ifd + 2 + 12 * k
                if e + 12 > len(head):
                    return None
                if int.from_bytes(head[e:e + 2], order) == 0x0112:
                    if int.from_bytes(head[e + 8:e + 10], order) in (0, 1):
                        return None
                    data = bytearray(Path(path).read_bytes())
                    data[e + 8:e + 10] = (1).to_bytes(2, order)
                    return bytes(data)
            return None
        pos += 2 + size
    return None
