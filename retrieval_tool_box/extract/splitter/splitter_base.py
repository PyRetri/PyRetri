# -*- coding: utf-8 -*-

from abc import abstractmethod

import torch

from ...utils import ModuleBase

from typing import Dict


class SplitterBase(ModuleBase):
    """
    The base class for splitter function.
    """
    default_hyper_params = dict()

    def __init__(self, hps: Dict or None = None):
        """
        Args:
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        super(SplitterBase, self).__init__(hps)

    @abstractmethod
    def __call__(self, features: torch.tensor) -> Dict:
        pass
