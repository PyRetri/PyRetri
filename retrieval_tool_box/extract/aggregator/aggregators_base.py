# -*- coding: utf-8 -*-

from abc import abstractmethod

import torch

from ...utils import ModuleBase

from typing import Dict


class AggregatorBase(ModuleBase):
    r"""
    The base class for feature aggregators.
    """
    default_hyper_params = dict()

    def __init__(self, hps: Dict or None = None):
        """
        Args:
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        super(AggregatorBase, self).__init__(hps)

    @abstractmethod
    def __call__(self, features: Dict[str, torch.tensor]) -> Dict[str, torch.tensor]:
        pass
