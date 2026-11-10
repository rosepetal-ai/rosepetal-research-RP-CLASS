# RP-CLASS - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""EuroSAT RGB (docs/BENCHMARK.md, G1): land use and land cover in Sentinel-2 satellite tiles, 10 classes, one task.

Data: EuroSAT_RGB.zip of Zenodo record 7711810 (Helber et al., IEEE JSTARS 2019), 94,658,721 bytes; its MD5 matched
the one Zenodo publishes (f46e308c4d50d4bf32fedad2d3d62f3b) on 2026-11-06, and the SHA-256 pinned below is that
download's. The zip holds EuroSAT_RGB/<class>/<class>_<n>.jpg and nothing else: 27,000 RGB JPEGs of 64 x 64 px,
    AnnualCrop 3,000   Forest 3,000   HerbaceousVegetation 3,000   Highway 2,500   Industrial 2,500
    Pasture 2,000   PermanentCrop 2,500   Residential 3,000   River 2,500   SeaLake 3,000
Licence: MIT ('The dataset is licensed under the MIT license', the README of github.com/phelber/EuroSAT; the record's
licence field says mit-license too). The tiles are Copernicus Sentinel data, free to reproduce and adapt; an image
shown anywhere needs the notice 'Contains modified Copernicus Sentinel data [year]'. The benchmark shows none.

Choices:
    test      built, as there is no official split: 500 tiles of each class, the cap (40 % would be 800 to 1,200)
    groups    none given: the RGB tiles carry no scene, date or place; only finalize's look-alikes and near-copies
    classes   the folder names, the paper's class names, in snake_case (AnnualCrop -> annual_crop); alphabetical
    size      the tiles stay 64 x 64; IMGSZ 224 (docs/BENCHMARK.md, Protocol) upsamples them, the same for every method
    disk      the zip is deleted once extracted: 235 MB at the peak (the zip and 140 MB of tiles on disk), 140 MB after

Counts (prepared on the GPU machine, 2026-11-06): no exact copies. 2,705 tiles are featureless, dark open water and
forest whose 32 x 32 thumbnails are flat (2,016 SeaLake, 686 Forest, 3 AnnualCrop; a median standard deviation of 1.0
grey levels, 15 for the other tiles): distinct tiles that all look alike, so they join no group. Of the others, 32
join a look-alike: 16 groups of 2 to 6 tiles, 15 of SeaLake and one pair of River tiles; 6 SeaLake pairs among them
are near-copies, none with two labels. Nothing is dropped or left out: test 5,000 (500 per class), val 98 (11 for a
class of 3,000 tiles, 9 for 2,500, 7 for 2,000), pool 21,902, train 299 per seed (33 / 28 / 22).

Caveats: tiny tiles that are not of objects, far from images of industrial parts; the classes are textures and
colours. With no scene or place to group by, tiles cut from one Sentinel-2 scene can sit on both sides of the split,
so the built test is easier than a place-disjoint one would be (the same split for every method).
"""
import re

import fetch

NAME = 'eurosat'
TASKS = (NAME,)
IMGSZ = 224
LICENCE = ("MIT; the tiles are Copernicus Sentinel data, free to reproduce and adapt, with the notice 'Contains "
           "modified Copernicus Sentinel data [year]' wherever one is shown")
SOURCE = 'https://github.com/phelber/EuroSAT'
URL = 'https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1'
SIZE, SHA256 = 94658721, 'b4f5b234ecb7d7ff9c6cddb046543b4717c53fd6e9815be6c0e80cc614f51b90'
COUNT = 27000


def download(raw):
    """EuroSAT_RGB.zip -> raw/images, once; the zip is deleted when its tiles are out."""
    images = raw / 'images'
    if not (images / '.extracted').exists():
        archive = fetch.download(URL, raw / 'EuroSAT_RGB.zip', size=SIZE, sha256=SHA256)
        fetch.extract(archive, images)
        archive.unlink()
    return images / 'EuroSAT_RGB'


def rows(folder, task):
    """Labels: the folder names in lowercase snake_case (AnnualCrop -> annual_crop), the benchmark's convention."""
    out = [{'path': p, 'label': re.sub(r'(?<!^)(?=[A-Z])', '_', p.parent.name).lower()}
           for p in sorted(folder.glob('*/*.jpg'))]
    if len(out) != COUNT:
        raise RuntimeError(f'eurosat: {len(out)} tiles in {folder}, {COUNT} expected; delete {folder.parent} to '
                           f'download it again')
    return out
