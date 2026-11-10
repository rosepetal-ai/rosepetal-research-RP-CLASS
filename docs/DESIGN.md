# RP-CLASS design

What RP-CLASS is and the decisions behind it, taken on 2026-11-05 after a review of RP-OD. When building shows that a
decision has to change, the change goes to "As built" at the end, next to the number of the decision it changes.

## What it is

A single-label image classifier for industrial vision, built and used like RP-OD: one object (`RPCLASS`), a few sizes,
one fixed recipe, a time budget, ONNX export. It is an alternative to common classifiers such as Ultralytics YOLO-cls,
for any classification dataset. Anomaly detection is out of scope.

## Locked design decisions

| # | Decision | Why |
|---|---|---|
| 1 | Generic single-label classifier for any classification dataset. The output is the most probable class and every class score; no decision threshold and no OK/NOK operating point in the library. | A general classifier: the threshold belongs to the application, which gets the top class and every score. |
| 2 | The backbone is chosen by a benchmark: HGNetv2 (RP-OD's) against DINOv3 ViT / ConvNeXt, including an FP16 export check. On a tie, HGNetv2. | RP-OD's recipe was fixed by measurement. DINOv3 backbones do well on small datasets, but can overflow FP16 and have their own licence. |
| 3 | Images are letterboxed exactly as in RP-OD: aspect ratio kept, gray 114 padding, antialiased bilinear, float32 scale; the arithmetic of a CUDA letterbox kernel, so a GPU pre-processing at serving time gives the same pixels. | Nothing at the borders is lost, and training and serving see the same pixels. |
| 4 | The size given to `train` is used, rounded up to multiples of 32, as in RP-OD. Without one, each size uses its own measured default (As built 4). | Respect the user's choice; a 2048 px default makes classification slow for nothing. |
| 5 | v1 is a standalone classifier. The export has a dynamic batch, so many crops can be classified in one call. | First show that it beats YOLO-cls on its own. |
| 6 | No recipe knobs: model size, image size, batch and time only. A knob is added only if data demands it. | A fixed, measured recipe makes fewer mistakes; every exposed knob has to be priced and tested. |
| 7 | The best epoch has the highest macro-F1 on val; equal macro-F1 goes to the lower val loss, an exact tie to the latest epoch; a tie does not reset patience. Top-1 accuracy is logged too. | Rare classes are the norm in inspection; small validation sets saturate (RP-OD's tie rule). |
| 8 | The default augmentation is chosen with data. Never anything that erases or invents content: no random erasing, mixup or cutmix. Replaceable from Python (`augment=`), as in RP-OD. | Those transforms can remove the defect or create one. |
| 9 | Validation is required and must not be empty (a clear error otherwise, as in RP-OD). A class missing from val is left out of the averages and shown as '-'. | Same behaviour as RP-OD. |
| 10 | Package `rpclass`, class `RPCLASS`, model name `rpclass`, checkpoints `rpclass-<size>.pt`, environment `RPCLASS_CACHE` / `RPCLASS_WORKERS`, cache `~/.cache/rpclass`. | One name everywhere. |
| 11 | RP-OD's code is copied and adapted (trainer and time budget, letterbox, dataset check, export); no dependency on `rpod`. | Each library can change without breaking the other. |
| 12 | Public from version 0.1.0, under the dual licence, with the licence header in every source file. | Publishing cannot be undone. |
| 13 | `main` gets the releases; changes go through `feat/` and `fix/` pull requests. Tests and gitleaks run in CI. | A safety net on every change. |
| 14 | Export: `model.onnx` with input `images` float32 [B, 3, H, W] in 0-1 (letterboxed; the normalization is inside), first output `logits` [B, C] (2-D, C >= 2), a dynamic batch with its TensorRT profile embedded, opset 17; and `config.json` with the model name, task, input shape, maximum batch, the pre-processing and the classes, no post-processing (contract in rpclass/engine/exporter.py). | What a runtime needs to feed the model; logits, so FP16 cannot normalize probabilities twice. |
| 15 | A contract a training service can drive unchanged: `train(time=, reserve=, patience=function, final_eval=)` with RP-OD's return keys, per-epoch log lines with a phase token pinned by tests, top-1 as a 0-1 fraction. | A service can budget, stop and follow a training with the library as it is. |
| 16 | Data: a folder with `train/`, `val/` (or `validation/`) and an optional `test/`, one subfolder per class. Class ids are the sorted folder names; `unlabeled/` is ignored; empty class folders are fine. | The standard layout (torchvision's ImageFolder, YOLO-cls), and what dataset export tools usually write. |
| 17 | Pretrained weights through timm (Hugging Face), the exact tag pinned in the code (HGNetv2: ImageNet; lt: DINOv3 LVD-1689M); their licences in THIRD_PARTY.md. Own weight releases only if there is ever an own pretraining. | What RP-OD does for its backbone; no weights of our own to host. |
| 18 | Benchmark: three small private inspection datasets (not published) to tune the recipe; public datasets (general and industrial, subsampled to few images per class, as RF100-VL does for detection) for the published comparison. Baselines: YOLO-cls and a DINOv3 fine-tune. The public list is used only after its licences are checked. | The real datasets decide the recipe; the public ones make the comparison repeatable. |
| 19 | Experiments run on one RTX 4090, in Docker (Python 3.11, torch 2.6, timm 1.0.30). | Every training in docs/EXPERIMENTS.md ran in the same environment. |
| 20 | `docs/EXPERIMENTS.md` keeps every experiment and its numbers. | The numbers behind every decision stay in the repository. |

## As built

- **2 (backbones), provisional.** Until the benchmark, the sizes are RP-OD's HGNetv2 B0 / B2 / B4 with RP-OD's
  pretrained tags (`ssld_stage1_in22k_in1k`); the DINOv3 candidates (`convnext-t-dinov3`, `vit-s16-dinov3`) are
  registered in `BACKBONES` and build from the same code (ViTs at the training size). First benchmark (private sets,
  docs/EXPERIMENTS.md 2026-11-06): the DINOv3 ViT-S leads HGNetv2-B2 by about one point, serves in float16 on TensorRT,
  is 1.8x slower. Decision (2026-11-06): two ranges, the CNN sizes s / m / l (HGNetv2 B0 / B2 / B4) and a transformer
  size lt (DINOv3 ViT-S/16); the backbone of each size can still change with the recipe experiments and the public benchmark.
- **4 (image size).** Measured (docs/EXPERIMENTS.md, image size): HGNetv2-B2 peaks at 384 px, the ViT still gains at
  448, and 224 costs every model 1.8 to 2.4 points. `train(imgsz=None)`, the default, is now the size's own: 384 for s,
  m and l, 448 for lt (BACKBONES in rpclass/cfg/schema.py), or a checkpoint's training size.
- **6 (recipe), measured in part.** RP-OD's structure (warmup, flat, cosine over the last 20 %, EMA, restore the best
  for the decay and the clean last 4 %, BatchNorm frozen below 8 images per step) with the usual fine-tuning values: the
  new head at 1e-3, the backbone at 1e-4, weight decay 0.05, clipping 1.0; batch 32, 384 px (448 for lt, see 4),
  100 epochs, patience 20. Measured in the recipe experiments (docs/EXPERIMENTS.md, 2026-11-06): the learning rates stay
  (doubled: -1.1 points; halved: nothing consistent); label smoothing 0.1 is added (+0.3 points on average, and on set 1
  the OK images rejected to catch every NOK fall from 81 and 95 to 23 and 28 of 160; a confidence then stops at 0.95 for
  two classes).
- **8 (augmentation).** Measured (docs/EXPERIMENTS.md, 2026-11-06): the 8 orientations of the square (flips and turns by
  90 degrees) gained 1.3 to 3.6 points on the private sets and 1.1 to 1.8 on histology, and changed nothing on natural
  photos. They are the default (photometric distortion 0.5, horizontal and vertical flips 0.5, a random turn by a
  multiple of 90 degrees), and `train(keep_orientation=True)` removes every flip and turn for a task whose class is the
  orientation (a part upside down, a mirrored label). It is the first knob decision 6 allows, because the data asked for
  it, and a user understands it ("the orientation matters"); the gain is too large to leave off, and its cost on such a
  task too. A checkpoint keeps the switch, and training it again inherits it.
- **14 (export).** The logits leave the model shifted so that each image's highest is 0: the same softmax and class, and
  never a row that looks like probabilities, which a runtime that applies the softmax only to rows that do not already
  sum to 1 would pass through untouched. The parity check compares the ONNX Runtime logits with PyTorch's relative to
  their scale (1e-3), not the probabilities (near uniform in an untrained network, they hide a broken graph) nor the
  predicted class (a near tie swaps it under rounding, RP-OD's problem). A second check runs the model once with every
  internal tensor exposed and warns when a value exceeds float16 (float16 engines are the usual choice for serving; the
  DINOv3 ViT's attention does exceed it). Any failure of a check, ONNX Runtime's own errors included, prints a warning
  and keeps the export.
- **15 (contract).** The epoch and validation lines are pinned whole by `tests/test_budget.py`, train and val times with
  one decimal (a parser can sum them); the GPU-memory error says 'out of memory', so that a job runner can recognise it
  (to retry on a larger GPU, say); RP-OD's message lacks the words. A time-budget call (time, a reserve function,
  patience as a function, final_eval=False) has its own test.
- **Review 2026-11-05.** An adversarial review (5 lenses, each finding re-checked by a refuter) confirmed 24 findings
  and 8 in part, all fixed with a test each. The ones that mattered: in time mode the warmup was sized from the steps
  the whole budget holds, ignoring the epochs cap and validation, and could outlast the run (inherited from RP-OD);
  truncated JPEG / BMP / TIFF passed the dataset check (now every image is decoded once, JPEG at 1/8 scale); the
  prediction grid read every misclassified image instead of 16; equal macro-F1 values could differ in the last bit and
  defeat the val-loss tie-break (now computed exactly). Also: resume keeps the clean stretch and the early stop,
  BatchNorm freezes on the real images per step, the sample order and augmentation no longer depend on the number of
  workers, non-finite scores count as wrong, `time=` includes the final evaluation.
- **GPU kept busy, 2026-11-09.** On the public benchmark the GPU sat idle for long stretches. Four changes help, and
  none of them changes a number (each has a test that compares every number of a training):
  - Validation reads its split once into memory (up to 1 GB of letterboxed images) instead of spawning workers and
    decoding the same images every epoch. On small-image sets validation was 40-60 % of an epoch; on VisA at 448 px it
    went from 0.8 to 0.2 s.
  - Every training micro-batch is prepared by all the workers at once, in chunks joined in pinned memory
    (`rpclass/data/loader.py`), instead of one worker per micro-batch. With the ~10 steps an epoch of a 300-image
    dataset has, the GPU no longer waits for all of them at once. B2 on VisA at 448 px went from 2.0 to 1.6 s of
    training per epoch.
  - While it trains on a GPU with workers, the main process keeps 4 CPU threads (`MAIN_THREADS`). All it does on the CPU
    is copies and bookkeeping, and its default pool of one thread per core stalled once the workers filled the cores. On
    a 24-core machine running another training, a validation of 100 images held in memory took 0.3-2.4 s with 24 threads
    and 0.2-0.4 s with 4.
  - The benchmark decodes test images in workers.

  On large photos the CPU itself was the limit. A VisA image cost ~40 ms to decode, augment and letterbox at full
  resolution (6 ms to decode, 21 to augment, 11 to letterbox), and the i9-13900K tops out at ~250 images/s however
  many workers there are.

  Since then, large training images are read shrunk (docs/EXPERIMENTS.md, 2026-11-09; rpclass/data/dataset.py). An image
  goes to exactly 3x the size it is letterboxed to when that at least halves its pixels, a JPEG decoded at a reduced DCT
  scale. It is then augmented and letterboxed with the full image's geometry. At an exact odd factor the two resizes
  make the single full-resolution one: the tensor is the full-resolution one up to the 8-bit rounding of the
  intermediate image (0.04-0.08 / 255 on average). Reading at 2x, the first version, lost 5-24 % of the contrast of fine
  detail. This applies to the library's augmentations and square input sizes. The loader then runs 2.3-3.1x faster on
  large photos (tyres at 448 px: 4.5 s to 1.6 s an epoch).
