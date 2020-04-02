# -*- coding: utf-8 -*-

from abc import abstractmethod

import torch

from ...utils import ModuleBase

from typing import Dict


class EnhanceBase(ModuleBase):
    """
    The base class of feature enhancer.
    """
    default_hyper_params = {}

    def __init__(self, hps: Dict or None = None):
        """
        Args:
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        super(EnhanceBase, self).__init__(hps)

    @abstractmethod
    def __call__(self, feature: torch.tensor) -> torch.tensor:
        pass
