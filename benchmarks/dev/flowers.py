# RP-CLASS - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""Development suite (never part of the benchmark): TensorFlow's flower_photos (TFDS tf_flowers), 5 kinds of flower in
Flickr photos, one task. It is here to show whether turning images (90-degree turns, vertical flips), which suits parts
seen from above, hurts natural photos, where up often means something.

Data: flower_photos.tgz from storage.googleapis.com/download.tensorflow.org/example_images (last modified 2016-02-10),
228,813,984 bytes. Its MD5 matched the one the host publishes (x-goog-hash, 6f87fb78e9cc9ab41eff2015b380011d) on
2026-11-06. The SHA-256 pinned below is that download's, and TFDS pins the same one. The tgz holds
flower_photos/<class>/<flickr id>_<secret>[_<size>].jpg and LICENSE.txt: 3,670 RGB baseline JPEGs,
    daisy 633   dandelion 898   roses 641   sunflowers 699   tulips 799
from 143 to 1,024 px wide and 159 to 442 px high (median 320 x 240; 345 sizes). All of them decode. None has an EXIF
orientation tag. 1,878 carry an ICC profile, which neither reader applies.
Licence: CC BY 2.0. LICENSE.txt opens with 'All images in this archive are licensed under the Creative Commons
By-Attribution License, available at: https://creativecommons.org/licenses/by/2.0/'. It then gives one line per photo,
3,670 in all, every one CC-BY, naming the photo's Flickr author (1,534 accounts). TensorFlow's tutorial says the same
('all images are licensed CC-BY'). The credits matter only where a photo is shown, and the suite shows none.

Choices:
    classes   the folder names, which are TFDS's names and already lowercase
    test      built, as there is no official split (TFDS has a train split only): round(0.4 x n_c) of each class
    groups    none given. LICENSE.txt names each photo's author, but an author is not a capture session. Two photos by
              one author of one kind differ nearly as much as two by different authors: their 32 x 32 grey thumbnails
              are 57.6 / 255 per pixel apart at the median, against 63.1, and 0.4 % of the 8,756 such pairs are under
              20. So finalize's copy check is what groups this set
    disk      the tgz is deleted once extracted: 0.46 GB at the peak (the tgz and 233 MB of photos), 0.23 GB after

Counts (prepare.py, 2026-11-06): 3,670 listed. Dedup kept one copy of each of 2 byte-identical sunflower pairs, which
have different Flickr ids. It dropped the photo filed in both roses and tulips (15922772266_1167a06620.jpg, the same
bytes). It joined 3 near-copy pairs, each one picture under two Flickr ids (2 daisy, 1 rose, 0.2 to 1.0 / 255 apart),
and a rose photo with its grey version (2.6). Of the 3,666 left:
    test    daisy 253, dandelion 359, roses 256, sunflowers 279, tulips 319: 1,466
    val     17, 24, 17, 19, 22: 99
    pool    363, 515, 367, 399, 457: 2,101
    train   52, 73, 52, 57, 65 per seed: 299

Orientation, from a look at 40 test photos (8 of each class, at random):
    19 are upright scenes: a stem, the ground or the sky, a vase, a person. Turned 90 degrees or flipped, they look
       wrong.
    17 are close-ups of flower heads seen face-on or from above, with no up. A turned copy looks just as natural.
     4 are in between.
The upright share depends on the class: tulips 6 of 8 (they stand on their stems), dandelions and sunflowers 4, roses
3, daisies 2 (mostly face-on heads). Turning can therefore remove a real cue, an up direction tied to the class, as
well as add variety. With about half of the photos upright, the set measures the net effect.

Caveats: the labels are loose, as in any web set. The 40 hold a gerbera among the daisies, a rose tattoo and a parade
float among the roses, and a tractor in a tulip field. A training draw holds trees seen from below through a dark rim
(tulips/5687705933_55a8c2dbac.jpg).
Retakes stay ungrouped. Of the 31 pairs of photos within 15 / 255 of each other (32 x 32 grey thumbnails), finalize
joins or drops the 7 closest. Of the other 24, looked at one by one, 12 (6.2 to 14.7 apart, each within one class) are
one scene shot twice or one photo edited (a grey or recoloured copy), and more lie further apart. Such a pair can
straddle test and pool, and a training draw takes 14 % of the pool. A seed's draw therefore holds the twin of a test
photo for about 0.8 of these 12 pairs on average (2 x 0.40 x 0.57 x 0.14 each), against 1,466 test photos.
These are common flowers in Flickr photos, and Oxford Flowers-102 was a DINOv2 retrieval seed (Table 15: 1,000,000
images kept from a retrieval on its train split). DINO backbones have likely seen many such photos, so the scores may
sit near the ceiling.
"""
import fetch

NAME = 'flowers'
TASKS = (NAME,)
IMGSZ = 224
LICENCE = ("CC BY 2.0, as flower_photos/LICENSE.txt states: 'All images in this archive are licensed under the "
           "Creative Commons By-Attribution License, available at: https://creativecommons.org/licenses/by/2.0/'; it "
           "credits each photo's Flickr author")
SOURCE = 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'
URL = SOURCE
TGZ, SIZE = 'flower_photos.tgz', 228813984
MD5 = '6f87fb78e9cc9ab41eff2015b380011d'                                    # as the host gives it (x-goog-hash)
SHA256 = '4c54ace7911aaffe13a365c34f650e71dd5bf1be0a58b464e5a7183e3e595d9c'  # that download's, and TFDS's
CLASSES = ('daisy', 'dandelion', 'roses', 'sunflowers', 'tulips')
COUNT = 3670


def download(raw):
    """flower_photos.tgz -> raw/images, once; the tgz is deleted when its photos are out."""
    images = raw / 'images'
    if not (images / '.extracted').exists():
        fetch.extract(fetch.download(URL, raw / TGZ, size=SIZE, sha256=SHA256, md5=MD5), images)
    (raw / TGZ).unlink(missing_ok=True)
    return images / 'flower_photos'


def rows(folder, task):
    out = [{'path': p, 'label': c} for c in CLASSES for p in sorted((folder / c).glob('*.jpg'))]
    if len(out) != COUNT:
        raise RuntimeError(f'flowers: {len(out)} photos in {folder}, {COUNT} expected; delete {folder.parent} to '
                           f'download it again')
    return out
