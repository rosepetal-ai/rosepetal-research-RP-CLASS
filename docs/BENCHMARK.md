# RP-CLASS public benchmark

A comparison of RP-CLASS with other classifiers on public datasets, cut to the size of a typical inspection dataset
(about 300 training images), so that the numbers say how each method does with little data. The protocol and the dataset
list were fixed on 2026-11-06, before any method ran on them. What gets published: the preparation scripts
(`benchmarks/public/`), the split manifests (file ids and SHA-256) and the metrics. No image is ever redistributed.

## Datasets

14 datasets, 7 general and 7 industrial, 34 tasks: VisA (12 objects) and DAGM (10 textures) run one OK / NOK task per
category and count once, as the mean of their tasks. Every licence allows commercial use; each was read at its source on
2026-11-05. Their authors, sources and licences, with links: `docs/DATASETS.md`.

| | Dataset | Task as used | Test | Licence |
|---|---|---|---|---|
| G1 | EuroSAT RGB (satellite land cover, 64 px) | 10 classes | built | MIT; Copernicus Sentinel data |
| G2 | Caltech-101 (web photos of objects) | 100 classes: BACKGROUND_Google and Faces_easy dropped, as CoOp does | built | CC BY 4.0 |
| G3 | NCT-CRC-HE (H&E histology, Kather 2019) | 9 tissue classes | CRC-VAL-HE-7K, patient-disjoint from the 100K training set | CC BY 4.0 |
| G4 | BloodMNIST, 224 px (blood-cell microscopy) | 8 classes | official | CC BY 4.0 |
| G5 | DeepWeeds (field agriculture) | 9 classes: 8 species + negative | built | CC BY 4.0 |
| G6 | Beans (leaf disease) | 3 classes, the 1,295 images pooled and split again | built | MIT |
| G7 | CLEVR v1.0, count_all (synthetic scenes) | 8 classes: 3 to 10 objects, as in VTAB | the official val, as in VTAB | CC BY 4.0 |
| I1 | VisA (PCBs, capsules, food, candles) | 12 x OK / NOK | the authors' 2cls_highshot split | CC BY 4.0 |
| I2 | DAGM 2007 (synthetic grey textures) | 10 x OK / NOK | the official test half | CC BY 4.0 |
| I3 | X-SDD (hot-rolled steel) | 7 defect types | built | Apache-2.0 |
| I4 | RAW-FABRID (textile, line-scan grey) | OK / NOK on 256 px patches, grouped by source frame | built | CC BY 4.0 |
| I5 | Wood surface defects (VSB-TU Ostrava) | 8 defect types from box crops, grouped by board | built | CC BY 4.0 |
| I6 | Tyres, good and defective | OK / NOK | built | CC BY 4.0 |
| I7 | SDNET2018 (concrete decks, walls, pavements) | cracked / uncracked, stratified by structure, grouped by photo | built | CC BY 4.0 |

Every binary inspection task names its classes ok and nok (nok: the defect, the positive class of the AUROC). Caveats,
each written in its script: EuroSAT and CLEVR are not photos of parts; Caltech-101 was a retrieval seed of DINOv2 / v3
(it may favour DINOv3 backbones); DeepWeeds species are tied to their location (a scene shortcut); DAGM and CLEVR are
synthetic; X-SDD's licence is a software licence with a blank copyright line and the image rights are not documented;
RAW-FABRID is a first version with no paper yet; the tyres are used tyres photographed by phone, several shots of a tyre
seconds apart that no id groups, and most defective ones outdoors with one phone (a capture cue every method can
use); BloodMNIST and NCT-CRC-HE have no patient ids inside their training sets.

Not included: sets whose licence forbids commercial use (MVTec AD, LOCO, AD 2 and 3D-AD, KolektorSDD and SDD2, VAD,
ELPV, MPDD, Real-IAD, AutoVI and others), sets with no licence at their source (NEU-CLS, Magnetic tile, GC10-DET and
others), and sets whose data are flawed (augmented before release, size shortcuts, leaking capture sessions).

## Protocol

**Splits, built once and frozen** (`benchmarks/public/common.py`, the same code for every dataset):

1. **Duplicates, in two tiers**, measured on real data (docs/EXPERIMENTS.md, 2026-11-06). Exact copies (the same
   bytes) are kept once; copies that disagree on the label are all dropped. Look-alikes (64-bit dHashes at most 4 bits
   apart and 32 x 32 grey thumbnails within 3 / 255 on average: the same photo cropped or shot again, but also distinct
   pieces on one plain background) only join a group. Near-copies, look-alikes whose 128 x 128 thumbnails are also
   within 1.5 / 255 (the same picture re-encoded or resized), that disagree on the label are dropped as label noise,
   the pair only, except from an official test. Featureless images (open water, empty background: a thumbnail
   standard deviation under 2 grey levels) are neither: they all look alike without being the same picture.
2. **Groups never cross the splits we build.** Groups are what the data gives (the source frame, board or photo; a
   group may hold several classes) joined with the look-alikes and near-copies.
3. **Test.** An official or author split is used whole, and pool images tied to it (the same source group, or a
   near-copy) are dropped; look-alikes are not: the authors' split is theirs. Otherwise the test is drawn by group:
   40 % of each class, at most 500 images, in proportion to the strata. Whole groups are taken at random while they
   hold a class (and stratum) still short of images, then exactly the wanted number is drawn from them; their other
   images go nowhere.
4. **Val**, from what remains, the same way, about 100 images: max(5, round(100 x share_c)) images of class c,
   where share_c is the class share of the images train and val come from (with strata, each class and stratum is
   rounded on its own: a class may get one image more or less).
5. **Pool**: everything else. Huge pools (the 100K histology set, CLEVR's 70,000 scenes) and SDNET (whose photos
   hold 240 tiles each: a test of whole photos would come from a handful of them) are first cut at random to a few
   hundred or thousand images per class, keeping the true class shares: the draws below never need more.
6. **A class without test, val or training images** is an error: such a task cannot be scored.

**Training sets, 3 seeds.** For seed s in {0, 1, 2}, train is a random draw from the pool of
min(pool_c, max(20, round(300 x share_c))) images of each class: about 300 images with the natural imbalance, at
least 20 per class (or all there are). The seed also sets the training run's seed. Val and test are the same for
every seed and every method.

**Casting the anomaly sets.** One task per category; OK against NOK, every defect type and every image with several
defects being NOK; defect-type classes only where each image or crop has a single type (X-SDD, wood). One-class
layouts (OK-only training sets) are pooled and split again so that NOK appears in training, unless the authors give
a supervised split (VisA 2cls_highshot, the DAGM halves). Only real or author-staged defects. These are supervised
few-shot results: they do not compare with one-class anomaly-detection leaderboards. Wood crops: each box with a
25 % margin per side; boxes under 24 px and crops that overlap a box of another class are dropped. A defect type
(X-SDD's, wood's) with fewer than 50 images is dropped from every split; none is. RAW-FABRID keeps the frame's kind (clean or defective) as a stratum, so
that the test's OK patches are not mostly the neighbours of defects.

**Changes the data forced**, each in its script and in the task's info.json (notes): X-SDD's 142 PNG images, all
of one class, are written as JPEG like the other 1,218 (the format alone gave the class away); SDNET's pool is cut to
2,000 tiles per class (step 5); in four VisA objects (cashew, chewing gum, fryum, pipe fryum) the training draw is the
whole pool, so their three seeds differ only by the run seed; BloodMNIST's official test loses one image whose byte
copy in its training set has another label; JPEG files whose EXIF orientation tag turns the image are written with the
tag set to 1, so that every reader sees the same pixels (OpenCV, in YOLO, applies the tag; PIL, in RP-CLASS, does
not).

**Running.** Every method sees the same training and val images at the same input size: 224, or 448 for VisA, DAGM
and the tyres (RP-CLASS choosing its own size is reported apart, as "RP-CLASS (auto)"). Each method runs its
documented default recipe, with the same epoch cap and patience; the only adaptation allowed is what a method derives
by itself from train and val (early stopping, the checkpoint, a threshold). We run every baseline ourselves on the
same manifests and never mix in published numbers.

**Metrics.** Macro-F1 at the method's own decision rule is the primary metric; also balanced accuracy, top-1 and, for
binary tasks, AUROC. Per dataset: mean and standard deviation over the 3 seeds. The headline is the unweighted mean
over the 14 datasets, with general and industrial means apart; per-category results in an appendix.

## Results, version 1 (2026-11-09)

Every method ran the 34 tasks with all 3 seeds (102 runs each) on one RTX 4090. RP-CLASS m is HGNetv2-B2 and lt is
the DINOv3 ViT-S/16, both with the library's recipe and its default augmentation (the 8 orientations). YOLO11s-cls runs
Ultralytics' default recipe from its ImageNet weights. Everything ran at the protocol's sizes; "RP-CLASS (auto)", each
size at its own default input size, was not run. The tables are in `benchmarks/public/results/v1/report.md` and every
run is in `runs.csv` there; how it was run is in docs/EXPERIMENTS.md (2026-11-09).

Headline: macro-F1, the unweighted mean over the datasets (mean ± std over the seeds).

| method | all 14 | general (7) | industrial (7) |
|---|---|---|---|
| RP-CLASS lt (DINOv3 ViT-S/16) | **88.8 ± 0.3** | **89.8 ± 0.3** | **87.8 ± 0.8** |
| RP-CLASS m (HGNetv2-B2) | 85.0 ± 0.6 | 82.4 ± 0.7 | 87.6 ± 0.7 |
| YOLO11s-cls | 77.1 ± 1.2 | 77.8 ± 0.8 | 76.3 ± 2.5 |

Per dataset: macro-F1, mean over the seeds (the spreads are in the report).

| | EuroSAT | Caltech-101 | NCT-CRC-HE | BloodMNIST | DeepWeeds | Beans | CLEVR count | VisA | DAGM | X-SDD | RAW-FABRID | Wood | Tyres | SDNET2018 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lt | **93.8** | 93.8 | **90.6** | 97.1 | **78.2** | 95.6 | **79.1** | 82.4 | **99.8** | **99.4** | **84.6** | **72.7** | **97.2** | **78.7** |
| m | 92.3 | **94.0** | 87.9 | **97.3** | 69.1 | **96.2** | 39.8 | **94.0** | **99.8** | 98.3 | 80.0 | 70.3 | 95.7 | 75.3 |
| YOLO11s-cls | 87.5 | 89.9 | 81.3 | 90.6 | 67.8 | 92.4 | 35.0 | 84.8 | 97.2 | 94.1 | 47.1 | 66.6 | 90.0 | 54.3 |

What the results show:

- **Against YOLO11s-cls.** m leads YOLO11s-cls on all 14 datasets, and lt on 13 of them (all but VisA). On the
  headline, m is 7.9 points ahead and lt 11.7.
- **lt against m.** lt leads m by 3.8 points overall, mostly on the general sets. CLEVR's counting alone (79.1
  against 39.8) accounts for 2.8 of those points. Without CLEVR, lt leads by 1.0 point overall and by 2.0 on the
  general sets.
- **Industrial sets.** lt and m tie on average (87.8 and 87.6). m leads on VisA (94.0 against 82.4; AUROC 98.9 against
  94.8), and lt leads on the other industrial sets except DAGM, where they tie.
- **Where lt's VisA score comes from.** In 5 of its 36 VisA runs, lt never left the majority class: macaroni1 seeds 0
  and 2, macaroni2 seeds 1 and 2, pcb3 seed 0. Its validation macro-F1 never rose above 47.6, the score of calling
  every image OK, until patience ended the run at epoch 41. m never did this in any of its 75 binary runs. YOLO11s-cls
  ended 7 of its 75 binary runs near a single class, with a test macro-F1 under 56. This shows on validation, so the
  library can notice it; reproducing it and dealing with it belongs on development data, not on this benchmark.

Caveats on reading these numbers:

- Both RP-CLASS sizes ran at 224 px on 11 of the 14 datasets, below their own defaults (384 px for m, 448 for lt). On
  private sets, 224 cost every model 1.8 to 2.4 points.
- Caltech-101 was a retrieval seed of DINOv3, yet lt shows no advantage on it.
- These are one-time test numbers. They change nothing in the library (see Never).

## Never

- Use the test for any decision: hyperparameters, epochs, early stopping, the checkpoint, thresholds, augmentation,
  input size, backbone, class names, which seeds or datasets to report. The test is scored exactly once per method,
  dataset and seed, after training ends.
- Develop the library on these datasets: development uses val, private sets and a separate dev suite (Kather-2016, PCam,
  tf_flowers, Jia's coated wood). DAGM classes 7 to 10 are never looked at during development, as
  its authors ask.
- Edit a frozen manifest without a new version and a full rerun of every method.
- Let a group cross splits, or train on anything outside the seeded draw (no pseudo-labels, no self-supervision on
  the pool or the test, no test-time adaptation).
- Tune one method more than another, or draw an unlucky seed again.
- Redistribute images, crops, features or trained weights, or show Caltech images.

## Manifests

`benchmarks/public/manifests/<dataset>/<task>.csv` and `.json`, frozen on 2026-11-06 (version 1) before any method ran
on them: every image used (file, class, group, split, which seeds train on it, SHA-256, its path in the original
download) and each task's counts, provenance and dedup record. 34 tasks: 25 official tests used whole (VisA's 12
products, 240-442 images each, DAGM's 10 textures, 575 or 1,150, CLEVR's 15,000, NCT-CRC-HE's 7,180 and BloodMNIST's
3,420), 9 tests built, from 518 (Beans) to 5,000 (EuroSAT) images; val 96-111 images, except Caltech-101's 510 (5 a
class); about 300 training images per seed (Caltech-101: 1,955). Changing one
needs a new version and a rerun of every method. The development suite's are in `benchmarks/dev/manifests/`.

## Reproducing

    python3 -I benchmarks/public/prepare.py eurosat caltech101 ... --raw <downloads> --out <prepared> \
        --manifests benchmarks/public/manifests

downloads each dataset, checks every file against its pinned size and the checksum its host publishes (SHA-256, or
MD5 where the host gives only that: Zenodo, CaltechDATA, CLEVR's host), records it (URL, size, SHA-256), writes
`<prepared>/<dataset>/<task>/s<seed>/{train,val,test}/<class>/` and copies each task's manifest and info to
`benchmarks/public/manifests/`, then deletes the download. A prepared dataset is not downloaded again. SDNET2018 sits
behind a browser check: download the record by hand from https://digitalcommons.usu.edu/all_datasets/48/ (its
Download button gives DATA_Maguire_20180517_ALL.zip, checked against the MD5 the record prints) into
`<downloads>/sdnet/`; it is kept. Run with `python3 -I`, so numpy, Pillow, requests and (for Beans) pyarrow must be
installed outside `~/.local`.

The YOLO baseline needs ultralytics (AGPL-3.0), installed separately; the rpclass library does not use it.
