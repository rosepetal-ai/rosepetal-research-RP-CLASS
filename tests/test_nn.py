# RP-CLASS - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""The network: every size and every registered backbone builds, normalizes inside, and gives one score per class."""
import pytest
import torch

from rpclass.cfg.schema import BACKBONES, SIZES, Architecture, ConfigError, from_dict, input_size, to_dict

from .conftest import make_net


@pytest.mark.parametrize('size', list(SIZES))
def test_every_size_gives_one_score_per_class(size):
    net = make_net(num_classes=5, size=size, imgsz=(64, 96))
    with torch.no_grad():
        out = net(torch.rand(2, 3, 64, 96))
    assert out.shape == (2, 5) and torch.isfinite(out).all()


@pytest.mark.parametrize('backbone', [b for b in BACKBONES if not b.startswith('hgnetv2')])
def test_the_benchmark_candidates_build_at_the_training_size(backbone):
    """Vision transformers are built for their input size (not square here); CNNs take any size."""
    net = make_net(num_classes=3, imgsz=(64, 96), arch=Architecture(backbone))
    with torch.no_grad():
        assert net(torch.rand(1, 3, 64, 96)).shape == (1, 3)


def test_the_imagenet_normalization_is_inside_the_model():
    """Serving divides by 255 and nothing else: the model applies the mean and std of its pretrained weights."""
    net = make_net()
    x = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        mean = torch.tensor(net.body.pretrained_cfg['mean']).view(1, 3, 1, 1)
        std = torch.tensor(net.body.pretrained_cfg['std']).view(1, 3, 1, 1)
        assert torch.allclose(net(x), net.body((x - mean) / std))
    assert 'mean' not in net.state_dict()                      # not a weight: it comes from the backbone's config


def test_the_head_is_the_last_linear_layer():
    """What trains at the head's learning rate: the weight and bias of the new class layer, a tiny part of the net."""
    net = make_net(num_classes=4)
    head = net.head_parameters()
    assert len(head) == 2 and all(p.shape[0] == 4 for p in head)
    assert sum(p.numel() for p in head) < 0.01 * sum(p.numel() for p in net.parameters())


def test_input_sizes_round_up_to_multiples_of_32():
    assert input_size(250) == [256, 256] and input_size([200, 330]) == [224, 352]
    with pytest.raises(ValueError, match='imgsz'):
        input_size([0, 64])


def test_the_checkpoint_config_round_trips_and_refuses_unknown_backbones():
    assert from_dict(to_dict(SIZES['m'])) == SIZES['m']
    with pytest.raises(ConfigError, match='not part of this version'):
        from_dict({'backbone': 'resnet-1000'})


def test_every_registered_backbone_is_a_published_timm_checkpoint():
    """The pinned tags must exist (checked without downloading anything)."""
    import timm
    published = set(timm.list_pretrained())
    assert all(b['timm'] in published for b in BACKBONES.values())
