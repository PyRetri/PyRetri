# -*- coding: utf-8 -*-

from .backbone_impl.resnet import ResNet
from .backbone_impl.vgg import VGG
from .backbone_impl.reid_baseline import ft_net
from .backbone_base import BackboneBase


__all__ = [
    'BackboneBase',
    'ResNet', 'VGG',
    'ft_net',
]
