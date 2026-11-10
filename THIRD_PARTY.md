# Third-party components

Pretrained weights are not part of RP-CLASS and are not distributed with it. timm downloads them from Hugging Face
when a model is first built, and each comes under its own licence, not under RP-CLASS's (LICENSE, section 3).

| Component | Used for | Publisher | Licence |
|---|---|---|---|
| HGNetv2 weights `hgnetv2_b0.ssld_stage1_in22k_in1k`, `hgnetv2_b2.ssld_stage1_in22k_in1k`, `hgnetv2_b4.ssld_stage1_in22k_in1k` (ImageNet) | sizes `s`, `m` and `l` | PaddlePaddle Authors (PaddleClas) | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| DINOv3 weights `vit_small_patch16_dinov3.lvd1689m` (LVD-1689M) | size `lt` | Meta | [DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license/) |
| DINOv3 weights `convnext_tiny.dinov3_lvd1689m` (LVD-1689M) | the `convnext-t-dinov3` backbone, a benchmark candidate, not a size | Meta | [DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license/) |
| Ultralytics | only `benchmarks/yolo_cls.py`, the YOLO baseline, which `benchmarks/public/benchmark.py` also calls; installed separately, not a dependency of rpclass | Ultralytics | [AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.html) |

The DINOv3 License is not an open-source licence. Using `lt` or `convnext-t-dinov3` means accepting it, including its
end-use restrictions, and a model trained from those weights carries its terms.

The Python packages RP-CLASS depends on (PyTorch, torchvision, timm, NumPy, PyYAML, Pillow, Matplotlib; ONNX and ONNX
Runtime for the export) are installed by pip, each under its own licence.

Datasets: `docs/DATASETS.md`.
