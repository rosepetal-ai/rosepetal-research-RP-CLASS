# RP-CLASS - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""The network: a timm classifier with pretrained weights, its head replaced by one for the dataset's classes.

Images come in as float tensors in [0, 1], shape [B, 3, H, W]. The normalization the pretrained weights expect
happens inside, as in RP-OD, so it is part of the exported model and the serving side only has to divide by 255. The
output is the class scores before the softmax (logits), [B, C].
"""
import timm
import torch
import torch.nn as nn

_create_model = timm.create_model      # one place to build timm models (the tests forbid downloads through it)


class Classifier(nn.Module):
    def __init__(self, d, pretrained=True):
        """d: rpclass.cfg.schema.derive(...). pretrained: start from the pretrained weights of the backbone (downloaded
        once by timm and cached); the head always starts fresh."""
        super().__init__()
        kwargs = dict(img_size=tuple(d['imgsz'])) if d['fixed_size'] else {}
        self.body = _create_model(d['timm_name'], pretrained=pretrained, num_classes=d['num_classes'], **kwargs)
        cfg = self.body.pretrained_cfg
        self.register_buffer('mean', torch.tensor(cfg.get('mean', (0.485, 0.456, 0.406))).view(1, 3, 1, 1),
                             persistent=False)
        self.register_buffer('std', torch.tensor(cfg.get('std', (0.229, 0.224, 0.225))).view(1, 3, 1, 1),
                             persistent=False)
        self.num_classes = d['num_classes']

    def forward(self, images):
        """images: [B, 3, H, W] in [0, 1] -> logits [B, C]."""
        return self.body((images - self.mean) / self.std)

    def head_parameters(self):
        """The parameters of the new classification head (the rest is the pretrained backbone)."""
        return list(self.body.get_classifier().parameters())

    def fuse(self):
        """Nothing to merge in the supported backbones (the runtimes fold BatchNorm themselves); kept so that every
        RP-CLASS network exports the same way."""
        return self
