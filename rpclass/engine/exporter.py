# RP-CLASS - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""Export for serving: model.onnx + config.json in one folder.

ONNX contract:
  input   images  float32 [B, 3, H, W], values 0-1, letterboxed as predict() does (the model normalizes inside).
                  B is dynamic, 1 to MAX_BATCH; the range is written in the model's metadata (TRT_PROFILE_KEY), so a
                  TensorRT engine can be built for it without guessing
  output  logits  float32 [B, C]  class scores before the softmax, shifted so that the highest of each image is 0: the
                  class is the one at 0, a softmax gives the probabilities (the shift does not change them). Scores,
                  not probabilities, and never a row that looks like probabilities (all <= 0): a runtime that guesses
                  whether to apply the softmax (some check whether a row sums to 1) always applies it, and float16
                  rounding cannot make it normalize twice
config.json: model name, task, input shape, maximum batch, the pre-processing the model expects, and the classes.
A TensorRT engine (or any other runtime's) is built from the ONNX on the machine that serves it; float16 engines are
the usual choice for serving, so the export warns when a value inside the model does not fit in float16.

A transformer's attention is exported with its keys centred over the tokens (_CentredKeys). That subtracts the same
amount from every logit of a query's row, so the softmax and the answer are exactly the same, but the logits shrink. A
DINOv3 ViT's first attention reached 2.2e6, past float16's 65504, and a float16 TensorRT engine (trtexec --fp16) gave
NaN for every image; centred, the same engine gives the model's answers (docs/EXPERIMENTS.md, 2026-11-09).
"""
import contextlib
import copy
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OPSET = 17
MAX_BATCH = 16
TRT_PROFILE_KEY = 'rosepetal.trt_profiles'
PREPROCESS = {'resize': 'letterbox', 'pad_value': 114, 'antialias': True, 'scale': 255, 'mean': None, 'std': None}
FP16_MAX = 65504.0
ATTENTION_MODULES = ('timm.models.eva', 'timm.models.vision_transformer')    # their attentions call F.scaled_dot_...


class _Exportable(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, images):
        return _shifted(self.net(images))


def _shifted(logits):
    """The logits with the highest of each row at 0: the same softmax and the same class."""
    return logits - logits.amax(1, keepdim=True)


class _CentredKeys:
    """torch.nn.functional as timm's attention modules see it during an export, its scaled_dot_product_attention
    computing on keys centred over the tokens: exact, and the logits shrink (module docstring). Counts its calls, so
    that an export can tell that it was used."""

    def __init__(self):
        self.calls = 0

    def __getattr__(self, name):
        return getattr(F, name)

    def scaled_dot_product_attention(self, q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None,
                                     **kwargs):
        self.calls += 1
        k = k - k.mean(dim=-2, keepdim=True)
        if attn_mask is not None or is_causal or dropout_p or kwargs:
            return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                                                  is_causal=is_causal, scale=scale, **kwargs)
        attn = (q * (q.shape[-1] ** -0.5 if scale is None else scale)) @ k.transpose(-2, -1)
        return attn.softmax(dim=-1) @ v


@contextlib.contextmanager
def _float16_safe_attention(net):
    """While exporting: every timm attention of `net` through _CentredKeys (yielded; None when there is none)."""
    attention = [m for m in net.modules() if hasattr(m, 'fused_attn') and type(m).__module__ in ATTENTION_MODULES]
    if not attention:
        yield None
        return
    centred = _CentredKeys()
    modules = [sys.modules[name] for name in ATTENTION_MODULES if hasattr(sys.modules.get(name), 'F')]
    functional, fused = [mod.F for mod in modules], [m.fused_attn for m in attention]
    try:
        for mod in modules:
            mod.F = centred
        for m in attention:
            m.fused_attn = True
        yield centred
    finally:
        for mod, f in zip(modules, functional):
            mod.F = f
        for m, f in zip(attention, fused):
            m.fused_attn = f


def serving_config(size, names, imgsz, max_batch=MAX_BATCH):
    return {
        'common': {'model_name': 'rpclass', 'task': 'classification',
                   'image_shape': {'height': int(imgsz[0]), 'width': int(imgsz[1]), 'channels': 3},
                   'max_batch': int(max_batch), 'preprocess': dict(PREPROCESS)},
        'task_specific': {'classes': {str(i): n for i, n in sorted(names.items())}, 'confidence': 0.0},
        'model_specific': {'size': size},
    }


def trt_profile(imgsz, max_batch=MAX_BATCH, stride=32):
    """The range of the input for TensorRT: any batch from 1 to max_batch, the trained size only (every image is
    letterboxed to it); granularity: what every dimension must stay a multiple of if a runtime narrows the range."""
    h, w = int(imgsz[0]), int(imgsz[1])
    return [{'name': 'images', 'min': [1, 3, h, w], 'opt': [1, 3, h, w], 'max': [int(max_batch), 3, h, w],
             'granularity': [1, 1, stride, stride]}]


def export_onnx(net, size, names, imgsz, output_dir, check=True):
    """size: the model size, written to config.json. check: compare ONNX Runtime with PyTorch on random images, and
    look for values that float16 cannot hold; both only warn: a failed check never removes the export."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / 'model.onnx'
    reference = copy.deepcopy(net).cpu().eval()
    fused = copy.deepcopy(reference).fuse()
    x = torch.rand(2, 3, int(imgsz[0]), int(imgsz[1]))
    with warnings.catch_warnings(), _float16_safe_attention(fused) as centred:
        warnings.simplefilter('ignore')   # tracer notices about constant shapes: expected for a fixed-size export
        torch.onnx.export(_Exportable(fused), (x,), str(onnx_path), input_names=['images'], output_names=['logits'],
                          dynamic_axes={'images': {0: 'batch'}, 'logits': {0: 'batch'}}, opset_version=OPSET,
                          dynamo=False)
    if centred is not None and not centred.calls:
        raise RuntimeError('the attention of this model was exported without its keys centred (this timm version '
                           'no longer calls F.scaled_dot_product_attention in it): a float16 engine may '
                           'give NaN. Fix rpclass.engine.exporter._float16_safe_attention before serving the model')
    _embed_profile(onnx_path, trt_profile(imgsz))
    (output_dir / 'config.json').write_text(json.dumps(serving_config(size, names, imgsz), indent=2))
    if check:
        for name, run in (('parity', lambda: _check(onnx_path, reference, torch.rand(3, 3, *map(int, imgsz)))),
                          ('float16', lambda: _fp16_warning(onnx_path, torch.rand(1, 3, *map(int, imgsz))))):
            try:
                run()
            except Exception as e:  # noqa: BLE001 - any failure of a check, ONNX Runtime's own errors included
                print(f'warning: export {name} check: {e}; the export is kept', flush=True)
    return output_dir


def _embed_profile(onnx_path, profiles):
    """Write the TensorRT range into the ONNX metadata (replacing any previous one); skipped without onnx."""
    try:
        import onnx
    except ImportError:
        return
    model = onnx.load(str(onnx_path))
    kept = [p for p in model.metadata_props if p.key != TRT_PROFILE_KEY]
    del model.metadata_props[:]
    model.metadata_props.extend(kept)
    entry = model.metadata_props.add()
    entry.key, entry.value = TRT_PROFILE_KEY, json.dumps(profiles)
    onnx.save(model, str(onnx_path))


# float32 rounding, accumulated in another order by ONNX Runtime, moves the logits by about 1e-6 of their scale; a real
# fault (a wrong weight, a lost normalization) by far more. The scale, not the probabilities: an untrained network
# gives near uniform probabilities that hide anything, and not the predicted class either, which a near tie can swap
# under rounding alone (what made RP-OD's first check reject correct exports).
PARITY_TOL = 1e-3


def _check(onnx_path, reference, x):
    """ONNX Runtime must reproduce PyTorch: the logits within PARITY_TOL of their largest value."""
    try:
        import onnxruntime as ort
    except ImportError:
        return
    sess = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    (logits,) = sess.run(None, {'images': x.numpy()})
    with torch.no_grad():
        ref = _shifted(reference(x).float()).numpy()
    worst = float(np.abs(logits - ref).max() / max(float(np.abs(ref).max()), 1e-6))
    if worst > PARITY_TOL:
        raise RuntimeError(f'exported ONNX differs from PyTorch (logits up to {worst:.2g} of their scale apart)')


def fp16_overflows(onnx_path, x):
    """The values inside the model that float16 cannot hold (above 65504), for one input: [(tensor, largest |value|)],
    the largest first. One ONNX Runtime run in float32 with every intermediate float tensor as an output."""
    import onnx
    import onnxruntime as ort
    model = onnx.shape_inference.infer_shapes(onnx.load(str(onnx_path)))
    floats = [v.name for v in model.graph.value_info if v.type.tensor_type.elem_type == onnx.TensorProto.FLOAT]
    known = {o.name for o in model.graph.output}
    model.graph.output.extend(v for v in model.graph.value_info if v.name in floats and v.name not in known)
    sess = ort.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])
    names = [o.name for o in sess.get_outputs()]
    big = []
    for name, value in zip(names, sess.run(None, {'images': x.numpy()})):
        top = float(np.abs(value).max()) if value.size else 0.0
        if top > FP16_MAX:
            big.append((name, top))
    return sorted(big, key=lambda t: -t[1])


def _fp16_warning(onnx_path, x):
    try:
        import onnx  # noqa: F401
        import onnxruntime  # noqa: F401
    except ImportError:
        return
    big = fp16_overflows(onnx_path, x)
    if big:
        worst = ', '.join(f'{n} {v:.3g}' for n, v in big[:3])
        print(f'warning: {len(big)} values inside the model exceed what float16 holds ({worst}): a float16 runtime '
              f'that does not keep them in float32 gives inf or NaN; check the float16 engine\'s answer before '
              f'serving it (benchmarks/tensorrt.py)', flush=True)
