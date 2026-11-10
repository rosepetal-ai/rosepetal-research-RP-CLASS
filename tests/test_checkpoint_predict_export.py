# RP-CLASS - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""A trained model out of its checkpoint: inspect, predict, export, and the export contract serving relies on."""
import json

import numpy as np
import pytest
import torch
from PIL import Image

from rpclass import RPCLASS
from rpclass.cfg.schema import SIZES, derive
from rpclass.engine.exporter import MAX_BATCH, TRT_PROFILE_KEY, _check
from rpclass.nn import Classifier
from rpclass.utils.checkpoint import load_weights

from .conftest import NAMES, make_net

onnx = pytest.importorskip('onnx')
ort = pytest.importorskip('onnxruntime')


def test_checkpoint_roundtrip(trained_ckpt):
    m = RPCLASS(trained_ckpt)
    assert m.names == NAMES and m.imgsz == [64, 96]
    assert 'size s' in repr(m) and 'trained, 3 classes' in repr(m)


def test_untrained_model_refuses_to_predict():
    with pytest.raises(RuntimeError, match='not trained'):
        RPCLASS().predict(np.zeros((8, 8, 3), np.uint8))


def test_info_reports_backbone_and_head():
    s = RPCLASS('s').info(imgsz=[250, 330], verbose=False)
    assert s['imgsz'] == [256, 352] and s['total'] == s['backbone'] + s['head'] and s['gflops'] > 0


def test_predict_inputs_and_probabilities(trained_ckpt, tmp_path):
    m = RPCLASS(trained_ckpt)
    img = np.random.default_rng(0).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    Image.fromarray(img).save(tmp_path / 'a.png')
    r = m.predict(img, color='BGR', device='cpu')[0]
    assert r.orig_shape == (480, 640) and r.probs.shape == (3,) and r.probs.sum() == pytest.approx(1, abs=1e-5)
    assert r.top1 == int(r.probs.argmax()) and r.name == NAMES[r.top1] and r.conf == pytest.approx(r.probs.max())
    assert [k for k, _ in r.top(2)] == list(np.argsort(-r.probs)[:2])
    assert len(m.predict([tmp_path / 'a.png', img], device='cpu')) == 2
    assert len(m.predict(tmp_path, device='cpu')) == 1
    assert r.plot().shape == (480, 640, 3)


def test_export_contract(trained_ckpt):
    out = RPCLASS(trained_ckpt).export()
    model = onnx.load(str(out / 'model.onnx'))
    inp, outp = model.graph.input[0], model.graph.output[0]
    assert [i.name for i in model.graph.input] == ['images'] and [o.name for o in model.graph.output] == ['logits']
    assert [d.dim_param or d.dim_value for d in inp.type.tensor_type.shape.dim] == ['batch', 3, 64, 96]
    assert [d.dim_param or d.dim_value for d in outp.type.tensor_type.shape.dim] == ['batch', 3]
    meta = {p.key: json.loads(p.value) for p in model.metadata_props}
    assert meta[TRT_PROFILE_KEY] == [{'name': 'images', 'min': [1, 3, 64, 96], 'opt': [1, 3, 64, 96],
                                      'max': [MAX_BATCH, 3, 64, 96], 'granularity': [1, 1, 32, 32]}]
    cfg = json.loads((out / 'config.json').read_text())
    assert cfg['common'] == {'model_name': 'rpclass', 'task': 'classification', 'max_batch': MAX_BATCH,
                             'image_shape': {'height': 64, 'width': 96, 'channels': 3},
                             'preprocess': {'resize': 'letterbox', 'pad_value': 114, 'antialias': True, 'scale': 255,
                                            'mean': None, 'std': None}}
    assert 'postprocess' not in cfg['common']                   # no postprocess entry: the output is plain logits
    assert cfg['task_specific']['classes'] == {'0': 'nok', '1': 'ok', '2': 'scratch'}
    assert cfg['model_specific'] == {'size': 's'}


@pytest.mark.parametrize('batch', [1, 5, MAX_BATCH])
def test_the_onnx_reproduces_pytorch_at_any_batch(trained_ckpt, tmp_path, batch):
    """Every image of a batch gets its own scores (served batches mix images), shifted so that the highest is 0."""
    m = RPCLASS(trained_ckpt)
    onnx_path = m.export(tmp_path / 'e') / 'model.onnx'
    x = torch.rand(batch, 3, *m.imgsz)
    session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    (logits,) = session.run(None, {'images': x.numpy()})
    with torch.no_grad():
        ref = m.net.cpu().eval()(x)
        ref = (ref - ref.amax(1, keepdim=True)).numpy()
    assert logits.shape == (batch, 3) and np.abs(logits - ref).max() <= 1e-3 * np.abs(ref).max()
    assert (logits.max(1) == 0).all()          # never a row a runtime could take for probabilities


def test_export_check_tolerates_rounding_but_not_a_broken_graph(trained_ckpt, tmp_path):
    from onnx import numpy_helper
    m = RPCLASS(trained_ckpt)
    onnx_path = m.export(tmp_path / 'ok') / 'model.onnx'
    x = torch.rand(3, 3, *m.imgsz)
    reference = m.net.cpu().eval()
    _check(onnx_path, reference, x)
    model = onnx.load(str(onnx_path))
    for init in model.graph.initializer:                         # every weight off by 10 %: a broken export
        w = numpy_helper.to_array(init)
        if w.dtype == np.float32 and w.size > 1:
            init.CopyFrom(numpy_helper.from_array(w * 1.1, init.name))
    broken = tmp_path / 'broken.onnx'
    onnx.save(model, str(broken))
    with pytest.raises(RuntimeError, match='differs from PyTorch'):
        _check(broken, reference, x)


def test_a_failed_check_keeps_the_export(trained_ckpt, tmp_path, monkeypatch, capsys):
    """RP-OD's lesson: a too strict check once left a correct training without any ONNX. Now a check only warns,
    whatever it raises, ONNX Runtime's own errors included (they are not RuntimeErrors)."""
    import rpclass.engine.exporter as E
    from onnxruntime.capi.onnxruntime_pybind11_state import Fail
    for error in (RuntimeError('exported ONNX differs from PyTorch'), Fail('the runtime gave up')):
        monkeypatch.setattr(E, '_check', lambda *args, e=error: (_ for _ in ()).throw(e))
        out = RPCLASS(trained_ckpt).export(tmp_path / type(error).__name__)
        assert (out / 'model.onnx').exists() and 'warning: export parity check' in capsys.readouterr().out


def test_an_intact_export_passes_its_checks_quietly(trained_ckpt, tmp_path, capsys):
    RPCLASS(trained_ckpt).export(tmp_path / 'e')
    assert 'warning' not in capsys.readouterr().out


def test_values_that_float16_cannot_hold_are_reported(trained_ckpt, tmp_path, capsys):
    """Float16 engines are the usual choice for serving; values above 65504 inside the model (a DINOv3 ViT's attention
    has them) give inf or NaN in a runtime that computes them in float16. The export says so and which tensors."""
    m = RPCLASS(trained_ckpt)
    with torch.no_grad():
        m.net.body.head.fc.weight.mul_(1e9)
    m.export(tmp_path / 'e')
    out = capsys.readouterr().out
    assert 'exceed what float16 holds' in out and "check the float16 engine's answer" in out


def test_a_trained_model_initializes_the_same_size_for_other_classes():
    src = make_net(num_classes=10)
    dst = Classifier(derive(SIZES['s'], 3, [64, 64]), pretrained=False)
    report = load_weights(dst, src.state_dict())
    assert report['loaded_fraction'] > 0.99
    assert report['reinitialized'] == ['body.head.fc']              # only the class layer starts fresh


def test_predict_on_arrays_is_predict_on_files(trained_ckpt, tmp_path):
    """benchmarks/run.py reads test images in DataLoader workers and predicts on the arrays: the same probabilities."""
    from rpclass.data.sources import read_image
    paths = []
    for i in range(3):
        path = tmp_path / f'{i}.jpg'
        Image.fromarray(np.random.default_rng(i).integers(0, 256, (70, 90, 3), dtype=np.uint8)).save(path)
        paths.append(str(path))
    m = RPCLASS(trained_ckpt)
    from_files = m.predict(paths)
    from_arrays = m.predict([read_image(p) for p in paths], color='RGB')
    assert all(np.array_equal(a.probs, b.probs) for a, b in zip(from_files, from_arrays))


def test_a_vit_exports_its_attention_with_centred_keys(tmp_path, capsys, monkeypatch):
    """A ViT's attention goes to the ONNX with its keys centred over the tokens (rpclass.engine.exporter), which
    float16 then holds: a float16 TensorRT engine gave NaN without it. ONNX Runtime still gives PyTorch's answer,
    every attention has its centring in the graph, and timm is left as it was."""
    import contextlib
    import onnx
    import timm.models.eva as eva
    import rpclass.engine.exporter as E
    from rpclass.cfg.schema import Architecture
    net = make_net(arch=Architecture('vit-s16-dinov3'))
    attention = [m for m in net.modules() if type(m).__name__ == 'EvaAttention']
    before = [m.fused_attn for m in attention]
    E.export_onnx(net, 'lt', {0: 'a', 1: 'b', 2: 'c'}, [64, 64], tmp_path / 'centred')
    assert 'export parity check' not in capsys.readouterr().out
    assert eva.F is torch.nn.functional and [m.fused_attn for m in attention] == before
    monkeypatch.setattr(E, '_float16_safe_attention', lambda net: contextlib.nullcontext())
    E.export_onnx(net, 'lt', {0: 'a', 1: 'b', 2: 'c'}, [64, 64], tmp_path / 'plain', check=False)

    def means(folder):
        return sum(n.op_type == 'ReduceMean' for n in onnx.load(tmp_path / folder / 'model.onnx').graph.node)
    assert len(attention) > 0 and means('centred') - means('plain') == len(attention)


def test_centred_keys_give_the_same_attention():
    """Centring the keys subtracts the same amount from every logit of a query's row: the same attention, from logits
    that a large part common to every key no longer blows up."""
    from rpclass.engine.exporter import _CentredKeys
    g = torch.Generator().manual_seed(0)
    q, v = torch.randn(2, 3, 17, 8, generator=g), torch.randn(2, 3, 17, 8, generator=g)
    k = torch.randn(2, 3, 17, 8, generator=g) + 300 * torch.randn(1, 1, 1, 8, generator=g)
    centred = _CentredKeys()
    got = centred.scaled_dot_product_attention(q, k, v)
    exact = torch.nn.functional.scaled_dot_product_attention(q.double(), k.double(), v.double()).float()
    assert torch.allclose(got, exact, atol=1e-4) and centred.calls == 1
    raw = (q @ k.transpose(-2, -1)).abs().max()
    assert (q @ (k - k.mean(-2, keepdim=True)).transpose(-2, -1)).abs().max() < raw / 10


def test_an_attention_that_cannot_be_centred_stops_the_export(tmp_path):
    """If a timm attention no longer goes through F.scaled_dot_product_attention, the export says so rather than
    write a model whose float16 engine may give NaN."""
    import timm.models.eva  # noqa: F401 - the module whose F the export replaces
    import rpclass.engine.exporter as E

    class Attention(torch.nn.Module):
        fused_attn = False

        def forward(self, x):
            return x * 2

    Attention.__module__ = 'timm.models.eva'

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn, self.fc = Attention(), torch.nn.Linear(3, 2)

        def forward(self, x):
            return self.fc(self.attn(x).mean((2, 3)))

        def fuse(self):
            return self

    with pytest.raises(RuntimeError, match='without its keys centred'):
        E.export_onnx(Net().eval(), 's', {0: 'a', 1: 'b'}, [32, 32], tmp_path / 'e', check=False)
