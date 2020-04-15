# -*- coding: utf-8 -*-

from yacs.config import CfgNode

import torch
import torch.nn as nn

from .registry import BACKBONES

from ..utils import load_state_dict
from torchvision.models.utils import load_state_dict_from_url


# the urls for pre-trained models in torchvision.
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


def build_model(cfg: CfgNode) -> nn.Module:
    """
    Instantiate a backbone class.

    Args:
        cfg (CfgNode): the configuration tree.

    Returns:
        model (nn.Module): the model for extracting features.
    """
    name = cfg["name"]
    model = BACKBONES.get(name)()

    load_checkpoint = cfg[cfg.name]["load_checkpoint"]
    if 'torchvision' in load_checkpoint:
        arch = load_checkpoint.split('://')[-1]
        state_dict = load_state_dict_from_url(model_urls[arch], progress=True)
    else:
        state_dict = torch.load(load_checkpoint)

    try:
        model.load_state_dict(state_dict, strict=False)
    except:
        load_state_dict(model, state_dict)

    return model
