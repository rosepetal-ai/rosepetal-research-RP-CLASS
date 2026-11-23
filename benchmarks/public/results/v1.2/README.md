# Public benchmark, version 1.2 (2026-11-17)

Version 1's protocol and manifests with RP-CLASS lt rerun at 0.1.2 (an lt run that still answers one class after its
warmup starts over once with a linear-probe head). RP-CLASS m and YOLO11s-cls are version 1's runs, copied here
unchanged: their code did not change, and m trains with 0.1.2 exactly as before. lt ran the 34 tasks with 3 seeds (102
runs) on one RTX 4090, with the same command and flags as version 1.

- `report.md`: the tables, written by `benchmarks/public/report.py` from every run's `predictions.csv`.
- `runs.csv`: one line per run, from `benchmarks/public/benchmark.py` (version 1's lines for m and YOLO11s-cls).
- `scores.csv`: one line per run, the metrics of its `predictions.csv` (`benchmarks/public/report.py --scores`);
  `benchmarks/public/plots.py` draws the README's charts from it.

What changed against version 1: `docs/BENCHMARK.md`, Results, version 1.2; how lt was run: `docs/EXPERIMENTS.md`,
2026-11-17.
