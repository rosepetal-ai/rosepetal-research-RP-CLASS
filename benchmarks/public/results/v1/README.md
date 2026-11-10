# Public benchmark, version 1 (2026-11-09)

The results of the protocol in `docs/BENCHMARK.md` on manifests v1: RP-CLASS m (HGNetv2-B2) and lt (DINOv3
ViT-S/16), both with the library's recipe and its default augmentation (the 8 orientations), and YOLO11s-cls with
Ultralytics' default recipe. Every method ran the 34 tasks with 3 seeds (102 runs each) on one RTX 4090.

- `report.md`: the tables, written by `benchmarks/public/report.py` from every run's `predictions.csv` (macro-F1,
  AUROC of the binary tasks, VisA and DAGM per task).
- `runs.csv`: one line per run, from `benchmarks/public/benchmark.py`: best and planned epochs, why it stopped, the
  test metrics, the best validation macro-F1 (RP-CLASS), wall time, input size.

How it was run, the command and what the numbers show: `docs/EXPERIMENTS.md`, 2026-11-09.
