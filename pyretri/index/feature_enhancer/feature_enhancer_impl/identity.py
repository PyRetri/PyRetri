# -*- coding: utf-8 -*-

import torch

from ..feature_enhancer_base import EnhanceBase
from ...registry import ENHANCERS

from typing import Dict

@ENHANCERS.register
class Identity(EnhanceBase):
    """
    Directly return features without any feature enhance operations.
    """
    default_hyper_params = {}

    def __init__(self, hps: Dict or None = None):
        """
        Args:
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        super(Identity, self).__init__(hps)

    def __call__(self, feature: torch.tensor) -> torch.tensor:
        return feature
