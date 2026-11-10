# Experiments

Every experiment that informs a decision, newest first. Each entry says what it asked, how it was run (setup,
data, GPU, seeds, command), what came out, and which decision of `docs/DESIGN.md` it settles or changes. Numbers are on
validation unless written otherwise; a test split is evaluated once at the end and never used for a decision.

Template:

```
## YYYY-MM-DD · short title

Question: what this has to decide.
Setup: datasets, GPU, seeds, command.
Result: table.
Conclusion: what it decides (DESIGN.md decision N), and what it does not show.
```

## 2026-11-09 · Serving parity: RP-CLASS in a float16 TensorRT engine behind Triton

Question: does a trained RP-CLASS model, exported and served in a float16 TensorRT engine, give the answers of
RPCLASS.predict()? And at what latency?

Setup: one fresh model of each range with the final recipe (library defaults) on private set 1, fold 0 (m: HGNetv2-B2 at
384 px, best epoch 48; lt: DINOv3 ViT-S/16 at 448 px, best epoch 43; both val and test macro-F1 1.0). Each model was
exported with model.export(), built with trtexec (--fp16 and the batch profile 1-16 stored in the ONNX) and served by
NVIDIA Triton 2.56 (TensorRT 10.9), the letterbox done on the GPU in front of the engine, the images sent over gRPC
through system shared memory.

The test split has 64 images. It was sent as RGB bitmaps one at a time (P1), as BGR bitmaps (P2), five in flight (P3)
and as PNG files decoded by the client (P4), plus a float32 engine without TF32 as a control (C). RTX 4090, nothing else
on the GPU.

Result:

| | m (HGNetv2-B2) | lt (DINOv3 ViT-S/16) |
|---|---|---|
| same class as predict(), 64 images | 64 in each of P1-P4 | 0 with the float16 engine (NaN scores), 64 with float32 |
| largest probability difference | 5.5e-3 (float16) | 6.0e-6 (float32) |
| float32 control against predict() on the CPU | 1.5e-6 | 3.8e-6 |
| P2, P3, P4 against P1 | identical, to the last digit | identical |
| round trip, one image at a time, p50 / p99 | 0.94 / 1.30 ms (float16); 1.20 ms (float32) | 3.03 ms (float32) |
| images per second, five in flight | 1,855 | - |
| predict() per image, for reference | 3.3 ms on the GPU | 4.9 ms |

The float32 controls match predict() on the CPU to a few millionths: the GPU letterbox in front of the engine is the
library's. The lt float16 engine gives NaN for every image, and a second build gives the same. Its attention in the
first block reaches 2.2e6 (the export warns of it), beyond float16's 65,504. On 2026-11-06 a float16 engine of the ViT
gave PyTorch's answer, with another checkpoint at 384 px, so whether it survives depends on the model.

Conclusion:
- m is ready to serve. In a float16 TensorRT engine it gives predict()'s answers to 0.6 % of probability, in under a
  millisecond.
- lt cannot be served in float16 as exported. Next: make the export safe for float16; failing that, serve lt in float32
  (about 3 ms an image).

Follow-up, the same evening: lt's export in four variants, each built with trtexec --fp16 and the batch profile and run
on 16 of the test images against PyTorch float32.

| variant | float32, against the original | float16 engine: NaN images | same class | largest probability difference |
|---|---|---|---|---|
| as it is | 2.7e-6 | 16 of 16 | - | - |
| attention logits clamped to +-3e4 | 0.12 | 16 of 16 | - | - |
| keys centred over the tokens | 1.1e-6 | 0 of 16 | 16 of 16 | 0.020 |
| both | 0.078 | 0 of 16 | 16 of 16 | 0.098 |

Centring the keys subtracts the same amount from every logit of a query row, so the softmax is unchanged. It brings the
first block's largest attention value from 2.2e6 to 5.9e5. In the float16 engine, lt then gives no NaN and the same
class for all 16 images. Clamping the logits is not exact, and on its own it does not stop the NaN.

Then, on 2026-11-10, the centring went into the export of every timm attention (rpclass/engine/exporter.py,
_float16_safe_attention; the export stops if it no longer applies). The whole check was run again for both models on
450 images: set 1 fold 0 (its train, val and test), plus the tests of sets 2 and 3 as images the set-1 models never saw.

| | m (HGNetv2-B2) | lt (DINOv3 ViT-S/16, keys centred) |
|---|---|---|
| same class as predict(), float16 engine | 450 of 450 in each of P1-P4 | 449 of 450 |
| probability difference: median / p99 / largest | 2.1e-4 / 3.5e-3 / 6.0e-3 | 1.2e-4 / 9.3e-3 / 5.1e-2 |
| float32 control against predict() on the CPU | 1.7e-6 | 4.4e-5 |
| P2 and P4 against P1 | identical | identical |
| round trip, one image at a time, p50 / p99 | 0.72 / 1.11 ms | 1.38 / 1.76 ms |
| images per second, five in flight | 2,213 | 1,027 |

lt's one disagreement is an image of set 2 that predict() calls nearly even (0.531 against 0.469, a logit margin of
0.12). It is the only image with a margin under 0.25 for lt; float16 moves lt's margins by up to 0.22. All 320 images of
set 1 agree. Five in flight, the engine runs batches, and float16 gives other last digits for 130 of 450 images, but the
same classes. In float16, lt serves 2.2x faster than in float32 (1.38 against 3.03 ms).

Conclusion, revised: both ranges serve in a float16 TensorRT engine with predict()'s answers. m agrees to 0.6 % of
probability; lt with centred keys agrees to 5 %, and differs only on a near tie.

## 2026-11-09 · Large training images read shrunk

Question: can training read large photos at a reduced size, so that the GPU stops waiting for the CPU, without
changing what the model learns?

Setup:
- First version: images read at about 2x the size their letterbox needs, with the letterbox geometry kept.
- Validation of that version: HGNetv2-B2, which is bit-for-bit reproducible, and the ViT, each run with and without the
  shrink. It used sizes where private and development images exceed twice their letterbox: 224 px, set 2 at 160 px and
  set 1 at 192 px. At 160 and 192 px every image of those sets is shrunk. 3 to 5 seeds (benchmarks/run.py and
  benchmarks/public/benchmark.py with --shrink-factor, each under its own --tag). The controls (set 2 at 224,
  Kather-2016, PCam) hold no image large enough to shrink.
- Then an adversarial review of the change: three lenses, each finding re-checked by a refuter.
- Second version: exactly 3x, only when that at least halves the pixels.

Result 1, the 2x version: macro-F1, mean ± std over the seeds (private sets: out of fold; development suite: test).

| set | model | seeds | without | with | per seed |
|---|---|---|---|---|---|
| set 2, 160 px, every image shrunk | B2 | 5 | 93.5 ± 0.4 | 94.3 ± 1.4 | -2.0 +0.4 +1.9 +2.0 +2.0 |
| set 1, 192 px, every image shrunk | B2 | 5 | 94.8 ± 0.5 | 94.9 ± 0.7 | +0.3 +1.3 +1.3 -0.6 -1.9 |
| set 1, 224 px | B2 | 3 | 96.5 ± 0.3 | 96.1 ± 0.9 | -1.9 +0.9 -0.0 |
| set 3, 224 px | B2 / ViT | 3 | 99.9 / 99.3 | 100.0 / 99.5 | within 0.3 |
| flowers, 224 px | B2 / ViT | 3 | 88.8 / 92.5 | 87.7 / 92.3 | B2 +1.4 -0.4 -4.3, ViT -0.7 -1.6 +1.8 |
| jiawood, 224 px | B2 / ViT | 3 | 82.4 / 85.8 | 80.3 / 87.4 | B2 -1.5 +1.7 -6.5, ViT +1.2 +3.0 +0.6 |
| controls: set 2, Kather-2016, PCam | B2 | 1-3 | | | identical, every run to the last digit |

Result 2, the review. At 2x, the two resizes make a wider filter than the single full-resolution one: on a grating at
448 px, the contrast kept at 0.2 / 0.3 / 0.4 / 0.45 cycles per pixel was 0.95 / 0.89 / 0.81 / 0.76 of the
full-resolution letterbox's. At an exact odd factor every output pixel is centred on an intermediate one, and the two
triangles make the full-resolution one: 1.000 at 3x and at 5x. The other confirmed findings, all fixed:
- the size search missed 10-14 % of large images at 256 and 512 px;
- a JPEG decoded at a reduced scale shifted its content by up to 0.43 px;
- the benchmarks' `upright` read shrunk without being asked;
- workers started by spawn or forkserver ignored the setting;
- functools.wraps passed the opt-in on to a wrapper;
- args.yaml named the wrapped augmentation `null`.

Result 3, the 3x version: the training loader alone (12 workers, d4, no GPU), and how far a shrunk training tensor is
from the full-resolution one (identity augmentation; training pixels only):

| images | input | loader, in full | loader, shrunk | speed-up | mean distance from full resolution |
|---|---|---|---|---|---|
| VisA macaroni2, 2.3 MP | 448 px | 1.33 s an epoch | 1.37 s | none: not shrunk (3x keeps more than half the pixels) | 0 |
| VisA macaroni2 | 224 px | 1.29 s | 0.55 s | 2.3x | 0.084 / 255 |
| tyres, up to 12 MP | 448 px | 4.54 s | 1.60 s | 2.8x | 0.039 / 255 |
| tyres | 384 px | 4.36 s | 1.42 s | 3.1x | 0.049 / 255 |

Conclusion:
- The 2x version changed no accuracy measurably. With every training image shrunk, B2 went +0.9 on set 2 and +0.1 on
  set 1, within the seed spread, and the controls were identical. Even so, it trained on images 5-24 % softer at fine
  detail than the full-resolution ones.
- The 3x version gives the full-resolution tensor, up to the 8-bit rounding of the intermediate image, and cuts the
  loader's time 2.3-3.1x on large photos. It applies to images it shrinks to half their pixels or fewer, with square
  input sizes and the library's augmentations.
- Not run: accuracy at 3x. Its tensors equal the full-resolution ones up to that rounding. The 2x runs, whose tensors
  were 3 to 15 times farther from the full-resolution ones and up to 24 % softer at fine detail, already show no
  measurable effect.

## 2026-11-09 · Public benchmark, version 1: RP-CLASS m and lt against YOLO11s-cls

Question: how does RP-CLASS do against a common classifier on public datasets cut to the size of a typical inspection
dataset? This is the benchmark's one evaluation of each test. It reports, and decides nothing in the library.

Setup: the protocol in docs/BENCHMARK.md: 14 datasets, 34 tasks, manifests v1, seeds 0-2, 224 px, or 448 px for
VisA, DAGM and the tyres; batch 32, at most 100 epochs, patience 20, DAGM classes 07-10 included (`--final`).
- RP-CLASS m (hgnetv2-b2) and lt (vit-s16-dinov3) used the default augmentation: benchmarks/augments.py's d4, which
  draws exactly as the library's does. They used the recipe with label smoothing 0.1, which every run's args.yaml
  records.
- The runs started before 10:47 used the library without three changes that only make training faster; later runs had
  them: validation from memory, micro-batches prepared by every worker, 4 main threads. Each of these has a test that
  compares every number of a training. On the GPU, 8 epochs of B2 on VisA gave the same losses and macro-F1.
- YOLO11s-cls used Ultralytics' defaults (benchmarks/yolo_cls.py).
- Hardware: one RTX 4090, with up to four runs at a time. Summed over the runs, the time was 4.3 h for m, 4.6 h for lt
  and 1.9 h for YOLO11s-cls.

Commands:

    python3 benchmarks/public/benchmark.py <prepared> <out> --datasets <the 14> --models hgnetv2-b2 \
        --seeds 0 1 2 --augment default --workers 12 --final       # the same for vit-s16-dinov3 and yolo11s-cls
    python3 benchmarks/public/report.py <prepared> <out>

Result: benchmarks/public/results/v1 (report.md, runs.csv). The per-dataset table is in docs/BENCHMARK.md, under
Results. Headline: macro-F1, the unweighted mean over the datasets (mean ± std over the seeds):

| method | all 14 | general | industrial |
|---|---|---|---|
| RP-CLASS lt | 88.8 ± 0.3 | 89.8 ± 0.3 | 87.8 ± 0.8 |
| RP-CLASS m | 85.0 ± 0.6 | 82.4 ± 0.7 | 87.6 ± 0.7 |
| YOLO11s-cls | 77.1 ± 1.2 | 77.8 ± 0.8 | 76.3 ± 2.5 |

Binary runs that never left the majority class. For RP-CLASS, validation macro-F1 never rose above 0.476, the score
of calling every image OK. For YOLO11s-cls, which records no validation score here, test macro-F1 stayed under 56:

| method | runs | which |
|---|---|---|
| m | 0 of 75 | |
| lt | 5 of 75 | VisA macaroni1 s0 and s2, macaroni2 s1 and s2, pcb3 s0; each stopped by patience at epoch 41 |
| YOLO11s-cls | 7 of 75 | VisA candle s2, pcb3 s1 and s2; DAGM class03 s0; RAW-FABRID s1 and s2; SDNET2018 s2 |

Conclusion: m and lt are both well ahead of YOLO11s-cls, by 7.9 and 11.7 points, and m is ahead on all 14 datasets.
lt leads m overall, mostly through the general sets: CLEVR's counting gives 2.8 of its 3.8 points. On the industrial
sets they tie on average; m is clearly ahead on VisA and lt ahead on the rest.

lt's VisA deficit is mostly the 5 runs that never learned the defect class. That failure shows on validation, which
makes it a development candidate: detect it, and find out why the ViT stays at the majority class. It has to be
reproduced on development data first.

What it does not show:
- RP-CLASS at its own input sizes: both models ran at 224 px on 11 datasets, below their defaults.
- A DINOv3 fine-tune baseline, which was not run.
- Any decision: none of these test numbers changes the library.

## 2026-11-06 · Learning rate and label smoothing; the ViT at 512 px; what is noise

Question: does the recipe gain from another learning rate (both halved or doubled: head 5e-4 / 2e-3, backbone 5e-5 /
2e-4) or from label smoothing 0.1? And does lt gain from 512 px over 448?

Setup: HGNetv2-B2 and ViT-S DINOv3, d4, 384 px, validation in float32; this batch's own baseline (base, the recipe as it
is). Private sets: 5 folds x 3 seeds, OOF macro-F1. Development suite: 3 seeds, test macro-F1. 336 runs
(benchmarks/run.py and benchmarks/public/benchmark.py with --recipe and --tag; benchmarks/recipe.py). The ViT at 512 px:
sets 1 and 2, 3 seeds x 5 folds.

Result on the private sets, OOF macro-F1 (mean ± std over 3 seeds); OK images rejected to catch every NOK in brackets
(of 160 / 153):

| recipe | set 1, B2 | set 1, ViT | set 2, B2 | set 2, ViT |
|---|---|---|---|---|
| base | 97.8 ± 0.4 (81) | 96.1 ± 0.3 (95) | 95.5 ± 1.0 (133) | 96.7 ± 0.7 (111) |
| learning rates x0.5 | 96.9 ± 0.5 (76) | 96.8 ± 0.6 (92) | 96.0 ± 0.7 (112) | 96.7 ± 0.4 (125) |
| learning rates x2 | 97.1 ± 0.3 (79) | 95.0 ± 0.7 (72) | 94.9 ± 0.5 (126) | 94.2 ± 1.4 (126) |
| label smoothing 0.1 | 97.8 ± 0.3 (**23**) | 97.2 ± 0.3 (**28**) | 96.1 ± 0.6 (118) | 96.6 ± 0.9 (130) |

Development suite, test macro-F1, B2 / ViT:

| recipe | flowers | jiawood | kather2016 | pcam |
|---|---|---|---|---|
| base | 87.9 / 92.3 | 82.1 / 85.4 | 85.9 / 88.8 | 83.9 / 83.7 |
| learning rates x0.5 | 87.8 / 93.0 | 86.2 / 77.8 | 86.3 / 88.3 | 83.8 / 83.7 |
| learning rates x2 | 87.9 / 90.4 | 82.6 / 80.6 | 86.8 / 86.9 | 83.8 / 83.2 |
| label smoothing 0.1 | 88.8 / 92.5 | 82.4 / 85.8 | 87.1 / 87.8 | 82.5 / 85.1 |

Mean change over base across the 12 numbers: x0.5 -0.2, x2 -1.1 (worse in 9 of 12), label smoothing +0.3. The ViT at
512 px: 97.1 ± 0.4 on set 1 and 96.8 ± 0.9 on set 2, against 97.7 ± 0.1 and 96.4 ± 0.7 at 448 (image-size batch).

What is noise: HGNetv2-B2 runs are bit-for-bit reproducible. This batch's B2 base repeats the image-size batch's 384 px
runs to the last digit (97.8 ± 0.4 and 95.5 ± 1.0, the same rejections), so the earlier gap between batches for B2 came
from the validation precision alone. The ViT is not: the same setting gave 96.1 here and 96.6 there (set 1), 96.7
against 95.9 (set 2), from non-deterministic attention kernels. Below about 1 point, a ViT difference says nothing.

Conclusion: the learning rates stay (doubling them loses 1.1 points on average; halving them changes nothing
consistent). Label smoothing 0.1 adds 0.3 points of macro-F1 on average, within the noise, but on set 1 the OK images a
threshold must reject to catch every NOK fall from 81 and 95 to 23 and 28 (of 160) for both models, the operating point
inspection cares about; on set 2 they barely move. Its cost: confidences stop at 0.95 for two classes
(1 - 0.1 + 0.1 / C), which a user sees. Decision (2026-11-09): label smoothing 0.1 is in the recipe. The ViT gains
nothing at 512 px over 448: lt stays at 448.

## 2026-11-06 · Image size

Question: which input size should each model use by default (decision 4: a size given to train is always used)?

Setup: HGNetv2-B2 (m) and ViT-S/16 DINOv3 (lt), d4 augmentation, validation in float32, sets 1 and 2, 5 folds, 3 seeds,
at 224, 320, 384 and 448 px. 240 runs (benchmarks/run.py --imgsz, two or four at a time; every run used the full batch
of 32).

Result, OOF macro-F1 (mean ± std over the 3 seeds); AUC of P(NOK) in brackets:

| size | set 1, B2 | set 1, ViT | set 2, B2 | set 2, ViT |
|---|---|---|---|---|
| 224 | 95.7 ± 0.6 (99.0) | 94.2 ± 0.5 (97.3) | 93.7 ± 1.0 (97.7) | 94.1 ± 0.7 (97.6) |
| 320 | 97.0 ± 0.7 (99.3) | 96.4 ± 0.9 (98.8) | 94.7 ± 1.0 (97.9) | 95.4 ± 0.6 (97.5) |
| 384 | **97.8 ± 0.4** (99.4) | 96.6 ± 0.4 (98.9) | **95.5 ± 1.0** (98.3) | 95.9 ± 0.4 (97.7) |
| 448 | 97.7 ± 0.4 (99.7) | **97.7 ± 0.1** (99.4) | 94.7 ± 0.6 (97.8) | **96.4 ± 0.7** (98.2) |

Conclusion: B2 peaks at 384 (448 is no better: set 1 -0.1, set 2 -0.8); the ViT still gains at 448 (+1.1 and +0.5 over
384), as a 16 px patch then covers less of the part. 224 costs 1.8 to 2.4 points to every model. Proposed defaults:
384 px for the CNN sizes (s, m, l), 448 px for lt; whether lt gains further at 512 is open.

Noise between batches: the same setting, d4 at 384, gave the ViT 97.4 ± 0.2 on set 2 in the augmentation batch and
95.9 ± 0.4 here, B2 97.5 against 97.8 on set 1 and 96.0 against 95.5 on set 2. The one change in between is the
validation precision (bfloat16 then, float32 now), which moves the best epoch and the weights the final decay restarts
from; GPU kernels are not deterministic either. So a difference between batches is not evidence below about 1.5 points:
every comparison is made within one batch, with its own baseline.

## 2026-11-06 · The 8 orientations beyond the private sets: the development suite

Question: the 8 orientations (d4) gained 1.3 to 3.6 points on the private sets, whose parts are seen from above. Do they
cost anything elsewhere: on natural photos, half of them upright (flowers), on histology (Kather-2016, PCam), on
coated-wood defect crops (Jia)?

Setup: the development suite (benchmarks/dev, docs/BENCHMARK.md), prepared with the benchmark's protocol (built test,
val of about 100, training draws of about 300 images per seed; Jia's whole pool, 152 crops); HGNetv2-B2 and ViT-S
DINOv3, the default augmentation against d4, 224 px, 3 seeds, validation in float32. 48 runs,
benchmarks/public/benchmark.py and report.py; scores on each set's test, used for development as the suite is meant to.

Result, test macro-F1 (mean ± std over the 3 seeds); PCam's AUROC of P(tumour) in brackets:

| model | flowers | jiawood | kather2016 | pcam |
|---|---|---|---|---|
| B2, default | 87.6 ± 1.3 | 84.1 ± 1.7 | 84.6 ± 0.6 | 82.6 ± 0.9 (91.1) |
| B2, d4 | 87.9 ± 1.2 | 82.1 ± 1.2 | 85.9 ± 1.2 | 83.9 ± 1.9 (93.2) |
| ViT, default | 92.4 ± 0.9 | 83.1 ± 2.1 | 87.7 ± 1.1 | 81.9 ± 1.8 (91.8) |
| ViT, d4 | 92.3 ± 0.7 | 85.4 ± 3.4 | 88.8 ± 0.8 | 83.7 ± 3.8 (93.2) |

Conclusion: no measurable cost anywhere. The flower photos, about half of them upright, are unchanged (+0.3, -0.1);
histology, which has no up, gains 1.1 to 1.8 points of macro-F1 and 1.4 to 2.1 of AUROC; Jia's crops move both ways
(-2.0 for B2, +2.3 for the ViT) within their noise (152 training crops, std up to 3.4). With the private sets, d4 is a
gain or neutral on six sets of four kinds. What no set here can show is a task whose class is the orientation (a part
mounted upside down, a mirrored label): d4 would teach a model to ignore exactly that, as the horizontal flip of the
current default already does for mirrored parts.

Decision (2026-11-06): d4 is the library's default, and train(keep_orientation=True) removes every flip and turn for a
task whose class is the orientation (docs/DESIGN.md, As built 8). In this entry and the two before it, "default" is the
provisional one, today's --augment hflip.

## 2026-11-06 · The default augmentation, on the private sets

Question: which augmentation becomes the default (decision 8)? The candidates (benchmarks/augments.py) never erase or
invent content: default, the provisional one (photometric distortion 0.5, horizontal flip 0.5); none; d4, default plus
the 8 orientations of the square (vertical flip 0.5, then a random multiple of 90 degrees); rotate, default plus a
free rotation of 0-360 degrees on a canvas that holds the whole image; d4-zoom, d4 plus a zoom-out 0.5 (the image on a
grey canvas up to 1.3x its size).

Setup: as the backbone benchmark (5 folds, out-of-fold scores, 3 seeds, 384 px, the provisional recipe, validation then
in bfloat16): HGNetv2-B2 (size m) and ViT-S/16 DINOv3 (size lt) on sets 1 and 2; set 3 is saturated. 240 runs
(benchmarks/run.py --augment).

Result, OOF macro-F1 (mean ± std over the 3 seeds); AUC of P(NOK) in brackets:

| augmentation | set 1, B2 | set 1, ViT | set 2, B2 | set 2, ViT |
|---|---|---|---|---|
| default | 93.9 ± 1.3 (98.5) | 94.9 ± 0.5 (98.3) | 94.7 ± 0.7 (97.9) | 95.9 ± 0.8 (97.9) |
| none | 92.2 ± 0.5 (97.3) | 93.9 ± 0.7 (97.9) | 93.7 ± 2.8 (97.8) | 95.3 ± 0.6 (97.2) |
| d4 | 97.5 ± 0.4 (99.4) | **96.8 ± 0.1** (99.3) | 96.0 ± 1.0 (98.5) | **97.4 ± 0.2** (98.7) |
| d4-zoom | **97.9 ± 0.4** (99.6) | 96.6 ± 0.4 (99.0) | **96.4 ± 0.7** (98.2) | 96.3 ± 0.7 (98.7) |
| rotate | 97.0 ± 0.8 (99.3) | 96.6 ± 0.5 (99.3) | 94.7 ± 0.5 (96.5) | 94.7 ± 0.2 (97.9) |

The d4 runs train longer (best epochs 41-53 against 31-40; 87-161 s a run against 63-103): the model keeps improving
under more varied views. The OK images a threshold must reject to catch every NOK stay many for every variant (55-153
of 153-160): that operating point hangs on the one hardest NOK of each set.

Conclusion: the 8 orientations are the lever, 1.3 to 3.6 points over the default and the best or within a standard
deviation of it in all four columns. The zoom-out adds nothing consistent (+0.4 for B2, -0.2 and -1.1 for the ViT); the
free rotation helps set 1 (round parts in round crops) but not set 2, whose crops are square and fill the frame (a
rotation shrinks the part inside a larger canvas and adds grey corners). Photometric distortion and the horizontal flip
are worth 0.6 to 1.7 points over nothing. Not yet the library's default: in both sets the parts are seen from above,
where every orientation is a valid view, while on a task where the orientation is the defect (a part upside down) d4
would teach the model to ignore it. Next: the development suite (natural photos, histology, coated wood) shows whether
d4 costs anything elsewhere. Meanwhile d4 is the augmentation of the recipe experiments.

## 2026-11-06 · Validation precision: the trainer's numbers against model.val()

Question: the runs.csv of the benchmark on the private sets (the trainer's final test evaluation) and its
predictions.csv (predict() afterwards, the same weights) disagree. By how much, and why?

Setup: the 135 RP-CLASS runs of the benchmark on the private sets (HGNetv2-B0, B2, ViT-S DINOv3; sets 1, 2 and 3;
3 seeds x 5 folds), test macro-F1 and top-1 from runs.csv against the same metrics computed from predictions.csv.

Result: 78 of 135 runs differ, by 1 or 2 test images of 53-54 (top-1 up to 3.7 points, macro-F1 up to 3.7; mean
difference +0.03 points, no direction). The trainer validated under autocast (bfloat16 on the RTX 4090), predict() in
float32: borderline images move.

Conclusion: validation runs in float32 from now on (rpclass/engine/trainer.py), per epoch and at the end: the best
epoch no longer depends on rounding, and the numbers a training reports are the ones model.val() gives. Training stays
in reduced precision. The benchmark summaries were already computed from predictions.csv (float32) for every method,
so no reported number changes.

## 2026-11-06 · What counts as a duplicate in the public benchmark

Question: the public splits group near-copies (docs/BENCHMARK.md, step 1), but the first rule (64-bit dHashes within
4 bits and 32 x 32 grey thumbnails within 3 / 255) trimmed official tests (VisA pipe_fryum 240 -> 174 and an empty
training pool, candle, pcb3, macaroni1, DAGM class04) by joining images that are not copies and dropping them as label
noise. Which rule tells the same picture from a picture that looks alike?

Setup: every candidate pair under the first rule (each image against the first image of every cluster so far, as
finalize matched them), on the real data of the dataset modules: EuroSAT, DAGM class01 and class04, DeepWeeds, VisA
(6 objects). For each pair, the mean thumbnail difference at 32, 64 and 128 px, and whether the labels agree. Then
real copies made on purpose: 60 real images (DeepWeeds, VisA, EuroSAT) re-encoded as JPEG 95 / 85 / 75 and resized to
0.8 then JPEG 90. Script: scratch calibrate_twins.py (not kept: a one-off measurement).

Result, mean thumbnail difference at 128 px (min / median / max):

| pairs under the first rule | count | same label | other label |
|---|---|---|---|
| VisA pipe_fryum (distinct pieces, one background) | 438 | 1.89 / 2.97 / 3.80 | 2.38 / 3.05 / 3.62 |
| VisA candle | 249 | 0.37 (4 true repeats) / 3.27 / 3.58 | 2.78 / 3.29 / 3.64 |
| VisA pcb3, pcb4, macaroni1 | 232 | 3.78 and up | 4.27 and up |
| DAGM class04 (independent textures) | 99 | 7.65 / 8.36 / 9.10 | 7.95 / 8.57 / 9.01 |
| DeepWeeds (the same scene shot twice) | 57 | 2.05 / 5.64 / 12.75 | none |
| EuroSAT (707 of 762 flat open water) | 762 | 0.45 / 0.91 / 3.68 | 1.38 / 2.06 / 3.01 |

| copies made on purpose (60 real images) | 32 px, 90th pct / max | 128 px, 90th pct / max |
|---|---|---|
| JPEG 95 | 0.23 / 0.30 | 0.55 / 0.65 |
| JPEG 85 | 0.40 / 0.49 | 0.90 / 1.18 |
| JPEG 75 | 0.59 / 0.67 | 1.20 / 1.59 |
| resized 0.8, JPEG 90 | 0.53 / 0.75 | 1.28 / 1.77 |

The review of the adapters added Caltech's 41 grouped pairs, each the same photo or drawing (cropped or shifted):
0.18-2.71 at 32 px, 0.43-5.61 at 128 px.

Conclusion: no single threshold separates them. At 32 px, copies (under 0.8), re-shots of one scene (Caltech, DeepWeeds:
up to 3) and distinct look-alikes (VisA: from 1.1) overlap; at 128 px, copies stay under 1.8 while distinct VisA pieces
start at 1.9 against the cluster leaders the rule compares with (1.57 over all pairs, found by the docstring check
of the frozen data: a thin margin, the frozen near-copies reaching 1.43) and textures at 7.6, but re-shots spread to
13. So two tiers (benchmarks/public/common.py): look-alikes
(the first rule) only join a group, which keeps re-shots on one side of the splits we build; near-copies (look-alikes
within 1.5 at 128 px) are the same picture, and only they count as label noise or tie a pool image to an official
test. Two refinements came from running it on EuroSAT: label noise is a pair, not a cluster (near-copies chain, and
one dark forest tile among 867 flat water tiles dropped 67 of them), and featureless images (32 x 32 thumbnail
standard deviation under 2 grey levels: 837 of the 895 EuroSAT tiles the first rule grouped, against 14.2 for the
rest) are neither look-alikes nor near-copies: chained into one group, the flat water would have made almost all of
sea_lake's test when drawn. Official tests now stay whole (VisA: 12 of 12, DAGM class01 and class04), pipe_fryum
trains on 237 images instead of none, and DeepWeeds' 57 retakes and Caltech's 45 repeated photos stay on one side of
the splits.

## 2026-11-06 · Backbones on the three private inspection datasets

Question: which backbone makes the sizes (decision 2): accuracy on small inspection datasets, speed, float16.

Setup: three private OK / NOK sets of 320, 266 and 383 images (sets 1, 2, 3), in 5 folds (`benchmarks/folds.py`): for
fold k, test = fold k, val = fold k+1, train = the rest; the 5 test folds pooled give out-of-fold (OOF) scores over
every image. RTX 4090, Python 3.11, torch 2.6, bfloat16. The provisional recipe of every RP-CLASS run: head lr 1e-3,
backbone 1e-4, weight decay 0.05, batch 32, 384 px letterbox, 100 epochs, patience 20, the default augmentation
(photometric, horizontal flip). Seeds 0, 1, 2. Baseline: Ultralytics YOLO11s-cls 8.4.93 with its own defaults at the
same size, epochs, patience and batch (`benchmarks/yolo_cls.py`). 270 runs (`benchmarks/run.py`,
`benchmarks/yolo_cls.py`), summarized by `benchmarks/summary.py`.

Result, OOF macro-F1 (mean ± std over the 3 seeds); AUC of P(NOK) in brackets:

| model | set 1 | set 2 | set 3 | s / run | params |
|---|---|---|---|---|---|
| vit-s16-dinov3 | **94.9 ± 0.5** (98.3) | **95.9 ± 0.8** (97.9) | 99.9 ± 0.1 | 87-95 | 21.6 M |
| hgnetv2-b2 | 93.9 ± 1.3 (98.5) | 94.7 ± 0.7 (97.9) | 100.0 | 63-103 | 9.2 M |
| hgnetv2-b4 | 94.7 ± 1.1 (98.9) | 91.2 ± 1.4 (96.5) | 100.0 | 72-125 | 17.8 M |
| hgnetv2-b0 | 92.6 ± 0.4 (97.5) | 91.8 ± 0.8 (97.4) | 100.0 | 47-90 | 3.9 M |
| convnext-t-dinov3 | 92.5 ± 0.7 (96.2) | 92.0 ± 2.1 (95.4) | 99.6 ± 0.1 | 103-111 | 27.8 M |
| YOLO11s-cls | 88.4 ± 0.9 (95.2) | 88.0 ± 1.7 (93.2) | 99.8 ± 0.1 | 26-39 | 5.5 M |

Every RP-CLASS run ended by patience (best epoch 29-45, 44-67 epochs run). Set 3 is saturated (every model 99.6-100) and
decides nothing.

Float16 (`benchmarks/fp16.py`, set 1, seed 0, fold 0, 8 test images): only the ViT holds values float16 cannot (block 0
attention, 2.3e6); the four CNNs never pass 65504. TensorRT (`benchmarks/tensorrt.py`, TensorRT 10.9, the same model,
one real test image): every float16 engine, the ViT's included, gives the PyTorch answer (probabilities within 3e-5, the
same class): TensorRT computes the attention safely; a plain float16 runtime (ONNX Runtime float16) would not. GPU time
per image, batch 1, 384 x 384, median of 500 runs, float16 / float32:

| model | RTX 4090 |
|---|---|
| hgnetv2-b0 | 0.22 / 0.35 ms |
| hgnetv2-b2 | 0.34 / 0.59 ms |
| hgnetv2-b4 | 0.44 / 0.75 ms |
| convnext-t-dinov3 | 0.68 / 1.10 ms |
| vit-s16-dinov3 | 0.62 / 1.37 ms |

Conclusion: RP-CLASS beats YOLO-cls by 5.5-8 points of macro-F1 with any of its backbones. The DINOv3 ViT-S is the most
accurate (1.0-1.2 points over HGNetv2-B2, about one standard deviation), and serves in float16 on TensorRT, but is 1.8x
slower than B2 and comes under Meta's DINOv3 licence; B4 is not better than B2 on data this small. Not decided yet
(decision 2): two datasets of a few hundred images, with a provisional recipe whose augmentation ignores what matters
for parts seen from above (rotation); the recipe experiments and the public benchmark come first. To catch every NOK of set 1,
every model still rejects 83-106 of its 160 OK images, so the recipe has a lot to give.
