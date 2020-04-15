# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from ..registry import BACKBONES


class BackboneBase(nn.Module):
    """
    The base class of backbone.
    """
    def __init__(self):
        super(BackboneBase, self).__init__()

    def _forward(self, x: torch.tensor) -> torch.tensor:
        pass
