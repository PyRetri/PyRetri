# -*- coding: utf-8 -*-

import torch
import numpy as np

from ..splitter_base import SplitterBase
from ...registry import SPLITTERS

from typing import Dict


@SPLITTERS.register
class Identity(SplitterBase):
    """
    Directly return feature maps without any operations.
    """
    default_hyper_params = dict()

    def __init__(self, hps: Dict or None = None):
        """
        Args:
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        super(Identity, self).__init__(hps)

    def __call__(self, features: torch.tensor) -> torch.tensor:
        return features
