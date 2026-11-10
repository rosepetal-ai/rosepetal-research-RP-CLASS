# RP-CLASS - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""The public benchmark's split protocol (benchmarks/public/common.py) and downloads (fetch.py): what the frozen
manifests promise."""
import csv
import functools
import hashlib
import http.server
import io
import json
import sys
import tarfile
import threading
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks' / 'public'))
import common  # noqa: E402
import fetch  # noqa: E402


def _base(seed, size=256):
    """A smooth picture: a stable perceptual hash."""
    a = np.random.default_rng(seed).integers(0, 256, (6, 6, 3), dtype=np.uint8)
    return np.asarray(Image.fromarray(a).resize((size, size), Image.BICUBIC), np.int16)


def _image(path, seed, size=48, fmt='PNG', quality=95, grain=None):
    """seed: the picture; grain: an independent fine texture on it (another shot of the same scene)."""
    a = _base(seed, size)
    if grain is not None:
        a = a + np.random.default_rng(grain).integers(-25, 26, a.shape)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(path, fmt, **({'quality': quality} if fmt == 'JPEG'
                                                                            else {}))
    return str(path)


def _manifest(out, task):
    return list(csv.DictReader(open(out / task / 'manifest.csv')))


def _built_rows(tmp_path, n=(150, 60, 30)):
    rows, k = [], 0
    for c, count in zip('abc', n):
        for i in range(count):
            rows.append({'path': _image(tmp_path / 'raw' / c / f'{i}.png', k), 'label': c,
                         'group': f'{c}-burst{i // 3}', 'stratum': 'x' if i % 4 else 'y'})
            k += 1
    return rows


def test_built_splits_keep_groups_apart_and_follow_the_counts(tmp_path):
    rows = _built_rows(tmp_path)
    info = common.finalize('t', rows, tmp_path / 'out', 224, root=tmp_path / 'raw')
    m = _manifest(tmp_path / 'out', 't')
    split_of_group = defaultdict(set)
    for r in m:
        split_of_group[r['group']].add(r['split'])
    assert all(len(s) == 1 for s in split_of_group.values())      # a group never crosses splits
    share = {c: n / 240 for c, n in zip('abc', (150, 60, 30))}
    for c, n in zip('abc', (150, 60, 30)):
        assert info['test'][c] == round(0.4 * n)                  # exact: groups are drawn whole, then trimmed
        assert info['val'][c] == max(5, round(100 * share[c]))
        want = min(info['pool'][c], max(20, round(300 * share[c])))
        assert all(info['train'][s][c] == want for s in ('s0', 's1', 's2'))
    # each seed's layout: every class folder in every split, the files of the manifest
    for s in ('s0', 's1', 's2'):
        for split in ('train', 'val', 'test'):
            assert sorted(p.name for p in (tmp_path / 'out' / 't' / s / split).iterdir()) == ['a', 'b', 'c']
        files = sum(1 for _ in (tmp_path / 'out' / 't' / s / 'train').rglob('*.png'))
        assert files == sum(int(r[s]) for r in m)
    assert all(not Path(r['source']).is_absolute() for r in m)
    assert {r['module_group'] for r in m} <= {r['group'] for r in rows}


def test_strata_stay_in_proportion(tmp_path):
    rows = []
    for i in range(400):
        stratum = 'deck' if i < 300 else 'wall'
        rows.append({'path': _image(tmp_path / f'{i}.png', i), 'label': 'ok', 'stratum': stratum})
    for i in range(400, 480):
        stratum = 'deck' if i < 420 else 'wall'
        rows.append({'path': _image(tmp_path / f'{i}.png', i), 'label': 'nok', 'stratum': stratum})
    common.finalize('t', rows, tmp_path / 'out', 224)
    m = _manifest(tmp_path / 'out', 't')
    stratum = {Path(r['path']).name: r['stratum'] for r in rows}
    test = Counter((r['label'], stratum[Path(r['source']).name]) for r in m if r['split'] == 'test')
    assert test[('ok', 'deck')] == 120 and test[('ok', 'wall')] == 40      # 40 % of 300 and of 100
    assert test[('nok', 'deck')] == 8 and test[('nok', 'wall')] == 24      # 40 % of 20 and of 60


def test_a_module_group_may_hold_several_classes(tmp_path):
    """A photo's cracked and uncracked tiles (a board's knots of every type) stay together, and nothing is dropped."""
    rows = []
    for photo in range(80):
        for t in range(12):
            label = 'nok' if t < 3 else 'ok'
            rows.append({'path': _image(tmp_path / f'{photo}_{t}.png', photo * 100 + t), 'label': label,
                         'group': f'photo{photo}'})
    info = common.finalize('t', rows, tmp_path / 'out', 224)
    m = _manifest(tmp_path / 'out', 't')
    split_of_photo = defaultdict(set)
    for r in m:
        split_of_photo[r['module_group']].add(r['split'])
    assert all(len(s) == 1 for s in split_of_photo.values())
    assert info['test'] == {'nok': 96, 'ok': 288}                  # 40 % of 240 and of 720, exactly
    assert info['dedup']['images_dropped_for_label_conflicts'] == 0


def test_official_test_is_used_whole_and_its_groups_leave_the_pool(tmp_path):
    rows = []
    for i in range(1200):                                    # an official test larger than the built cap of 500
        rows.append({'path': _image(tmp_path / 'test' / f'{i}.png', i), 'label': 'ok', 'split': 'test',
                     'group': f'p{i % 50}'})
    for i in range(300):
        rows.append({'path': _image(tmp_path / 'train' / f'{i}.png', 10_000 + i), 'label': 'ok', 'split': 'train',
                     'group': f'p{i % 60}'})                 # patients p0-p49 are also in test: those images go
    for i in range(100):
        rows.append({'path': _image(tmp_path / 'train' / f'n{i}.png', 20_000 + i), 'label': 'nok', 'split': 'train'})
    for i in range(40):
        rows.append({'path': _image(tmp_path / 'test' / f'n{i}.png', 30_000 + i), 'label': 'nok', 'split': 'test'})
    info = common.finalize('t', rows, tmp_path / 'out', 224)
    assert info['test'] == {'nok': 40, 'ok': 1200}
    assert info['dedup']['pool_images_tied_to_the_test_dropped'] == 250
    # shares come from what train and val are drawn from (50 ok left + 100 nok), not from the test
    assert info['share'] == {'nok': pytest.approx(100 / 150, abs=1e-6), 'ok': pytest.approx(50 / 150, abs=1e-6)}


def test_an_official_test_keeps_its_look_alikes_and_loses_its_near_copies(tmp_path):
    rows = [{'path': _image(tmp_path / 'test' / 'x.png', 7, size=256, grain=1), 'label': 'ok', 'split': 'test'}]
    for i in range(30):
        rows.append({'path': _image(tmp_path / 'test' / f'{i}.png', 100 + i), 'label': 'nok', 'split': 'test'})
    for i in range(80):
        rows.append({'path': _image(tmp_path / 'pool' / f'{i}.png', 200 + i), 'label': 'nok'})
        rows.append({'path': _image(tmp_path / 'pool' / f'o{i}.png', 300 + i), 'label': 'ok'})
    look_alike = _image(tmp_path / 'pool' / 'alike.png', 7, size=256, grain=2)          # another shot, another label
    copy_other_label = _image(tmp_path / 'pool' / 'copy.jpg', 7, size=256, fmt='JPEG', grain=1)
    rows += [{'path': look_alike, 'label': 'nok'}, {'path': copy_other_label, 'label': 'nok'}]
    info = common.finalize('t', rows, tmp_path / 'out', 224)
    assert info['test'] == {'nok': 30, 'ok': 1}                    # the official test, whole
    assert info['dedup']['near_copies_with_two_labels'] == 2       # x and its re-encoded copy with another label
    assert info['dedup']['images_dropped_for_label_conflicts'] == 1        # only the pool side of it
    assert info['dedup']['pool_images_tied_to_the_test_dropped'] == 0       # the look-alike is not tied to the test
    assert info['pool']['nok'] + info['val']['nok'] == 81                  # 80 + the look-alike


def test_copies_are_kept_once_and_conflicting_ones_dropped(tmp_path):
    rows = []
    for i in range(200):
        rows.append({'path': _image(tmp_path / f'a{i}.png', i), 'label': 'a'})
        rows.append({'path': _image(tmp_path / f'b{i}.png', 1000 + i), 'label': 'b'})
    (tmp_path / 'a0_copy.png').write_bytes((tmp_path / 'a0.png').read_bytes())
    (tmp_path / 'b_conflict.png').write_bytes((tmp_path / 'a1.png').read_bytes())
    rows += [{'path': str(tmp_path / 'a0_copy.png'), 'label': 'a'},
             {'path': str(tmp_path / 'b_conflict.png'), 'label': 'b'}]
    # a near-copy: the same picture re-encoded as JPEG, in another class: label noise, both go
    rows.append({'path': _image(tmp_path / 'b_near.jpg', 2, fmt='JPEG', quality=90), 'label': 'b'})
    info = common.finalize('t', rows, tmp_path / 'out', 224)
    d = info['dedup']
    assert d['exact_copies_dropped'] == 1 and d['conflicting_copies_dropped'] == 2
    assert d['near_copies_with_two_labels'] == 2 and d['images_dropped_for_label_conflicts'] == 2
    kept = sum(info['test'].values()) + sum(info['val'].values()) + sum(info['pool'].values())
    assert kept == 403 - 1 - 2 - 2              # the copy; a1 and its relabelled copy; a2 and its near-copy


def test_look_alikes_stay_on_one_side_of_a_built_split(tmp_path):
    rows = []
    for i in range(200):
        for shot in range(2):                    # two shots of each scene: look-alikes, not near-copies
            rows.append({'path': _image(tmp_path / f'{i}_{shot}.png', i, size=256, grain=shot), 'label': 'ab'[i % 2]})
    info = common.finalize('t', rows, tmp_path / 'out', 224)
    assert info['dedup']['look_alikes_grouped'] == 200 and info['dedup']['near_copies_found'] == 0
    split_of_scene = defaultdict(set)
    for r in _manifest(tmp_path / 'out', 't'):
        split_of_scene[Path(r['source']).name.split('_')[0]].add(r['split'])
    assert all(len(s) == 1 for s in split_of_scene.values())


def test_label_noise_drops_the_pair_not_the_chain(tmp_path):
    """Look-alike chains: one image of another class drops itself and its own near-copies, not the chain."""
    rows = []
    for k in range(21):                          # one picture under ever stronger grain: a chain of look-alikes
        rows.append({'path': _image(tmp_path / f'w{k:02d}.png', 7, size=256, grain=k), 'label': 'water'})
    rows.append({'path': _image(tmp_path / 'f.jpg', 7, size=256, fmt='JPEG', grain=10), 'label': 'forest'})
    for i in range(300):
        rows.append({'path': _image(tmp_path / f'o{i}.png', 1000 + i), 'label': 'forest' if i % 2 else 'water'})
    info = common.finalize('t', rows, tmp_path / 'out', 224)
    assert info['dedup']['images_dropped_for_label_conflicts'] == 2       # the forest copy of w10, and w10


def test_featureless_images_form_no_group(tmp_path):
    """Flat tiles all look alike without being the same picture: no look-alike, no near-copy, no giant group."""
    rows = []
    for v in range(40, 240):
        Image.fromarray(np.full((48, 48, 3), v, np.uint8)).save(tmp_path / f'w{v}.png')
        rows.append({'path': str(tmp_path / f'w{v}.png'), 'label': 'water' if v % 3 else 'forest'})
    for i in range(300):
        rows.append({'path': _image(tmp_path / f'o{i}.png', 1000 + i), 'label': 'forest' if i % 2 else 'water'})
    info = common.finalize('t', rows, tmp_path / 'out', 224)
    d = info['dedup']
    assert d['featureless'] == 200 and d['largest_group'] == 1 and d['images_dropped_for_label_conflicts'] == 0


def test_the_two_tiers_of_likeness(tmp_path):
    a = _image(tmp_path / 'x.png', 7, size=256, grain=1)
    reencoded = _image(tmp_path / 'x.jpg', 7, size=256, fmt='JPEG', quality=80, grain=1)
    shot = _image(tmp_path / 'y.png', 7, size=256, grain=2)
    assert common.near_copies(a, reencoded)
    (ha, ta), (hs, ts) = common.fingerprint(a), common.fingerprint(shot)
    assert bin(ha ^ hs).count('1') <= common.HASH_RADIUS and common._mad(ta, ts) <= common.PRE_MAD    # look-alike
    assert not common.near_copies(a, shot)                                                            # not a copy
    # two flat images of different brightness: the same dHash (no gradient), nothing alike
    for name, v in (('dark.png', 40), ('light.png', 200)):
        Image.fromarray(np.full((48, 48, 3), v, np.uint8)).save(tmp_path / name)
    (hd, td), (hl, tl) = common.fingerprint(tmp_path / 'dark.png'), common.fingerprint(tmp_path / 'light.png')
    assert hd == hl and common._mad(td, tl) > common.PRE_MAD
    assert common._bits(np.array([ha ^ hs, 0, 2 ** 64 - 1], np.uint64)).tolist() == [bin(ha ^ hs).count('1'), 0, 64]


def test_a_class_left_without_training_images_is_an_error(tmp_path):
    rows = [{'path': _image(tmp_path / f'a{i}.png', i), 'label': 'a'} for i in range(400)]
    rows += [{'path': _image(tmp_path / f'b{i}.png', 500 + i), 'label': 'b'} for i in range(6)]   # test 2, val 4
    with pytest.raises(RuntimeError, match='train s0 has no image of b'):
        common.finalize('t', rows, tmp_path / 'out', 224)


def test_the_same_listing_gives_the_same_splits(tmp_path):
    rows = _built_rows(tmp_path)
    common.finalize('t', rows, tmp_path / 'one', 224)
    common.finalize('t', list(reversed(rows)), tmp_path / 'two', 224)
    one, two = tmp_path / 'one' / 't', tmp_path / 'two' / 't'
    assert (one / 'manifest.csv').read_text() == (two / 'manifest.csv').read_text()
    assert json.loads((one / 'info.json').read_text()) == json.loads((two / 'info.json').read_text())


def test_jpeg_orientation_tags_are_cleared_in_the_layout(tmp_path):
    """OpenCV (YOLO) turns an image by its EXIF orientation, PIL (RP-CLASS) does not: the layout clears the tag."""
    rows = []
    for i in range(300):
        path = tmp_path / f'{i}.jpg'
        exif = Image.Exif()
        exif[0x0112] = 6 if i % 2 else 1
        Image.fromarray(_base(i, 64).astype(np.uint8)).save(path, 'JPEG', quality=90, exif=exif.tobytes())
        rows.append({'path': str(path), 'label': 'ab'[i % 4 // 2]})
    info = common.finalize('t', rows, tmp_path / 'out', 224)
    placed = list((tmp_path / 'out' / 't').rglob('*.jpg'))
    assert placed and all(Image.open(p).getexif().get(0x0112) == 1 for p in placed)
    sources = {r['file']: r['source'] for r in _manifest(tmp_path / 'out', 't')}
    for p in placed[:40]:                        # nothing else changed: the same pixels as the source
        original = Path(sources[p.name])
        assert np.array_equal(np.asarray(Image.open(p)), np.asarray(Image.open(original)))
    turned = sum(Image.open(s).getexif().get(0x0112) == 6 for s in sources.values())
    assert turned > 0 and info['dedup']['jpeg_orientation_tags_cleared'] == turned


def test_every_dataset_module_defines_each_name_once():
    """A second definition silently replaces the first: wood.py once had two _crops, and its rows() would have failed
    after a 30 GB download."""
    import ast
    folders = [Path(common.__file__).parent, Path(common.__file__).parents[1] / 'dev']
    for path in sorted(p for f in folders for p in f.glob('*.py')):
        tree = ast.parse(path.read_text())
        names = Counter(n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)))
        assert not [n for n, k in names.items() if k > 1], path.name


def test_presample_keeps_true_shares_and_listing_order():
    rows = [{'path': f'{i}', 'label': 'a' if i % 10 else 'b'} for i in range(1000)]
    sub = common.presample(rows, 50)
    assert Counter(r['label'] for r in sub) == {'a': 50, 'b': 50}
    picked = [int(r['path']) for r in sub if r['label'] == 'a']
    assert picked == sorted(picked)
    assert common.shares(rows) == {'a': 0.9, 'b': 0.1}


def test_extract_refuses_paths_outside_the_folder(tmp_path):
    bad = tmp_path / 'bad.zip'
    with zipfile.ZipFile(bad, 'w') as z:
        z.writestr('ok/a.txt', 'a')
        z.writestr('../escape.txt', 'b')
    with pytest.raises(RuntimeError, match='outside the extraction folder'):
        fetch.extract(bad, tmp_path / 'x')
    assert not (tmp_path / 'escape.txt').exists()
    good = tmp_path / 'good.tar'
    with tarfile.open(good, 'w') as t:
        data = b'hello'
        info = tarfile.TarInfo('d/a.txt')
        info.size = len(data)
        t.addfile(info, io.BytesIO(data))
        link = tarfile.TarInfo('d/link')
        link.type, link.linkname = tarfile.SYMTYPE, '/etc/passwd'
        t.addfile(link)
    fetch.extract(good, tmp_path / 'y')
    assert (tmp_path / 'y' / 'd' / 'a.txt').read_bytes() == b'hello'
    assert not (tmp_path / 'y' / 'd' / 'link').exists()       # links are never extracted


@pytest.fixture
def server(tmp_path):
    """A local web server for tmp_path/site."""
    site = tmp_path / 'site'
    site.mkdir()

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass

    httpd = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Quiet, directory=str(site)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield site, f'http://127.0.0.1:{httpd.server_address[1]}'
    httpd.shutdown()


def test_download_checks_and_records_what_it_gets(tmp_path, server):
    site, url = server
    data = bytes(range(256)) * 1000
    (site / 'data.bin').write_bytes(data)
    md5 = hashlib.md5(data).hexdigest()
    dst = fetch.download(f'{url}/data.bin', tmp_path / 'raw' / 'data.bin', size=len(data), md5=md5)
    assert dst.read_bytes() == data
    ledger = json.loads((tmp_path / 'raw' / 'downloads.json').read_text())
    assert ledger == [{'file': 'data.bin', 'url': f'{url}/data.bin', 'size': len(data),
                       'sha256': hashlib.sha256(data).hexdigest()}]
    with pytest.raises(RuntimeError, match='MD5 differs'):
        fetch.download(f'{url}/data.bin', tmp_path / 'other' / 'data.bin', md5='0' * 32)
    assert (tmp_path / 'other' / 'data.bin.bad').exists() and not (tmp_path / 'other' / 'data.bin.part').exists()


def test_download_refuses_a_web_page(tmp_path, server):
    site, url = server
    (site / 'quota.html').write_text('<html>quota exceeded</html>')
    with pytest.raises(RuntimeError, match='web page, not the file'):
        fetch.download(f'{url}/quota.html', tmp_path / 'raw' / 'images.zip', size=123)
    assert not (tmp_path / 'raw' / 'images.zip.part').exists()


def test_a_file_placed_by_hand_is_checked_and_recorded(tmp_path):
    raw = tmp_path / 'raw'
    with pytest.raises(RuntimeError, match='download it by hand'):
        fetch.placed(raw / 'x.zip', 'https://example.org/x')
    raw.mkdir()
    (raw / 'x.zip').write_bytes(b'abc')
    fetch.placed(raw / 'x.zip', 'https://example.org/x', size=3, md5=hashlib.md5(b'abc').hexdigest())
    assert json.loads((raw / 'downloads.json').read_text())[0]['sha256'] == hashlib.sha256(b'abc').hexdigest()
    with pytest.raises(RuntimeError, match='3 bytes, 4 expected'):
        fetch.placed(raw / 'x.zip', 'https://example.org/x', size=4)
