# RP-CLASS - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""Downloads for the public benchmark: idempotent, resumable, checked, and never trusting what comes in.

    download(url, dst, size=None, sha256=None, md5=None)
        one file, once: a partial download resumes; the size, SHA-256 and MD5 (what the host publishes: Zenodo gives
        MD5 only) are checked when known; every finished download is recorded (URL, size, SHA-256) in downloads.json
        next to it, which the manifests keep
    google_drive(file_id, dst, ...)
        the same for a public Google Drive file
    placed(path, url, size=None, sha256=None, md5=None)
        a file downloaded by hand (a host that lets no script in): checked and recorded the same way
    extract(archive, dest, keep=None)
        unpack a zip or tar (gz, bz2, xz) into its own folder, only regular files and folders, every member path
        checked (no absolute path, no '..'); keep(name) -> bool extracts a subset; done once
    list_zip(url)
        the member names and sizes of a remote zip, read by HTTP ranges from its central directory (a few requests,
        no download): to look at a large archive before fetching it

Everything downloaded is data, never code: nothing here imports or runs a file it fetched.
"""
import hashlib
import io
import json
import shutil
import tarfile
import time
import zipfile
from collections import OrderedDict
from pathlib import Path, PurePosixPath

import requests

HEADERS = {'User-Agent': 'rpclass-benchmark/0.1 (dataset preparation script)'}
CHUNK = 1 << 20
TRIES = 8


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(CHUNK), b''):
            h.update(chunk)
    return h.hexdigest()


def _record(dst, url, digest):
    ledger = dst.parent / 'downloads.json'
    entries = json.loads(ledger.read_text()) if ledger.exists() else []
    entries = [e for e in entries if e['file'] != dst.name]
    entries.append({'file': dst.name, 'url': url, 'size': dst.stat().st_size, 'sha256': digest})
    ledger.write_text(json.dumps(entries, indent=1))


def md5_of(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(CHUNK), b''):
            h.update(chunk)
    return h.hexdigest()


def check(path, size, sha256, md5=None):
    """None when the file is as expected, else what is wrong."""
    have = path.stat().st_size
    if size is not None and have != size:
        return f'{have} bytes, {size} expected'
    if sha256 is not None and sha256_of(path) != sha256.lower():
        return 'SHA-256 differs from the expected one'
    if md5 is not None and md5_of(path) != md5.lower():
        return 'MD5 differs from the one the host publishes'
    return None


def download(url, dst, size=None, sha256=None, md5=None, headers=None):
    """url -> dst (a file path), once. A finished file that fails a check is an error, never silently replaced."""
    dst = Path(dst)
    if dst.exists():
        bad = check(dst, size, sha256, md5)
        if bad:
            raise RuntimeError(f'{dst}: {bad}; delete it to download again')
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + '.part')
    for attempt in range(TRIES):
        have = part.stat().st_size if part.exists() else 0
        if size is not None and have == size:
            break
        h = {**HEADERS, **(headers or {})}
        if have:
            h['Range'] = f'bytes={have}-'
        try:
            with requests.get(url, headers=h, stream=True, timeout=(30, 300), allow_redirects=True) as r:
                if r.status_code == 416 and have:          # nothing left to send: the partial file is complete
                    break
                r.raise_for_status()
                if 'text/html' in r.headers.get('Content-Type', ''):     # a quota, login or warning page
                    raise RuntimeError(f'{url}: the server answered with a web page, not the file; open the URL in a '
                                       f'browser to see why')
                if have and r.status_code != 206:           # range ignored: start again
                    have = 0
                total = have + int(r.headers.get('Content-Length', 0) or 0)
                shown = time.monotonic()
                with open(part, 'ab' if have else 'wb') as f:
                    for chunk in r.iter_content(CHUNK):
                        f.write(chunk)
                        have += len(chunk)
                        if time.monotonic() - shown > 60:
                            shown = time.monotonic()
                            print(f'  {dst.name}: {have / 1e9:.2f} / {total / 1e9:.2f} GB', flush=True)
            break
        except (requests.RequestException, OSError) as e:      # a web page answer is not retried
            wait = min(600, 15 * 2 ** attempt)
            print(f'  {dst.name}: {type(e).__name__}: {e}; retrying in {wait} s', flush=True)
            time.sleep(wait)
    else:
        raise RuntimeError(f'{url}: download failed {TRIES} times')
    bad = check(part, size, sha256, md5)
    if bad:
        part.rename(part.with_suffix('.bad'))      # never resumed: a retry starts from zero
        raise RuntimeError(f'{url}: {bad}; the file is kept as {part.with_suffix(".bad")} for a look, and the next '
                           f'run downloads it again')
    part.rename(dst)
    _record(dst, url, sha256.lower() if sha256 else sha256_of(dst))
    print(f'  {dst.name}: {dst.stat().st_size / 1e9:.2f} GB downloaded', flush=True)
    return dst


def google_drive(file_id, dst, size=None, sha256=None, md5=None):
    """A public Google Drive file (the large-file warning page answered in the URL)."""
    url = f'https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t'
    return download(url, dst, size=size, sha256=sha256, md5=md5)


def placed(path, url, size=None, sha256=None, md5=None):
    """A file downloaded by hand from url and put at path: checked like a download and recorded in downloads.json."""
    path = Path(path)
    if not path.exists():
        raise RuntimeError(f'{path}: missing; download it by hand in a browser from {url} and put it there')
    bad = check(path, size, sha256, md5)
    if bad:
        raise RuntimeError(f'{path}: {bad}; download it again from {url}')
    ledger = path.parent / 'downloads.json'
    if not ledger.exists() or path.name not in {e['file'] for e in json.loads(ledger.read_text())}:
        _record(path, url, sha256.lower() if sha256 else sha256_of(path))
    return path


def _safe_name(name):
    p = PurePosixPath(name.replace('\\', '/'))
    if p.is_absolute() or '..' in p.parts or (p.parts and ':' in p.parts[0]):
        raise RuntimeError(f'archive member {name!r} points outside the extraction folder; refusing the archive')
    return p


def extract(archive, dest, keep=None):
    """archive -> dest (a new folder), once (a '.extracted' marker says it is done). keep(name) -> bool: only the
    members for which it is true. Links, devices and other special members are skipped."""
    archive, dest = Path(archive), Path(dest)
    marker = dest / '.extracted'
    if marker.exists():
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    count = 0
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                kind = (info.external_attr >> 16) & 0o170000
                if info.is_dir() or kind == 0o120000 or (keep is not None and not keep(info.filename)):
                    continue                                   # folders come with their files; links never
                target = dest.joinpath(*_safe_name(info.filename).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as src, open(target, 'wb') as out:
                    shutil.copyfileobj(src, out, CHUNK)
                count += 1
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as t:
            for m in t:
                if not m.isfile() or (keep is not None and not keep(m.name)):
                    continue
                target = dest.joinpath(*_safe_name(m.name).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with t.extractfile(m) as src, open(target, 'wb') as out:
                    shutil.copyfileobj(src, out, CHUNK)
                count += 1
    else:
        raise RuntimeError(f'{archive}: neither a zip nor a tar archive')
    marker.write_text(f'{count} files from {archive.name}\n')
    print(f'  {archive.name}: {count} files extracted', flush=True)
    return dest


class RemoteFile(io.RawIOBase):
    """A read-only, seekable view of a URL by HTTP range requests, in cached blocks: enough for zipfile to read a
    remote archive's central directory."""

    def __init__(self, url, block=1 << 20, cached=32):
        super().__init__()
        self.url, self.block, self.cached, self.pos, self.blocks = url, block, cached, 0, OrderedDict()
        r = requests.get(url, headers={**HEADERS, 'Range': 'bytes=0-0'}, timeout=60, allow_redirects=True)
        r.raise_for_status()
        if r.status_code != 206 or 'Content-Range' not in r.headers:
            raise RuntimeError(f'{url}: the server does not serve byte ranges')
        self.size = int(r.headers['Content-Range'].rsplit('/', 1)[1])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def _get(self, k):
        if k not in self.blocks:
            lo, hi = k * self.block, min(self.size, (k + 1) * self.block) - 1
            r = requests.get(self.url, headers={**HEADERS, 'Range': f'bytes={lo}-{hi}'}, timeout=120,
                             allow_redirects=True)
            r.raise_for_status()
            self.blocks[k] = r.content
            if len(self.blocks) > self.cached:
                self.blocks.popitem(last=False)
        self.blocks.move_to_end(k)
        return self.blocks[k]

    def readinto(self, b):
        n = max(0, min(len(b), self.size - self.pos))
        got = 0
        while got < n:
            at = self.pos + got
            k = at // self.block
            data = self._get(k)[at - k * self.block:]
            take = min(len(data), n - got)
            b[got:got + take] = data[:take]
            got += take
        self.pos += got
        return got


def list_zip(url):
    """[(member name, uncompressed size)] of a remote zip, from its central directory."""
    with zipfile.ZipFile(RemoteFile(url)) as z:
        return [(i.filename, i.file_size) for i in z.infolist()]
