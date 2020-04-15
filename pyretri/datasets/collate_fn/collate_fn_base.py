# -*- coding: utf-8 -*-

from abc import abstractmethod

import torch

from ...utils import ModuleBase

from typing import Dict, List


class CollateFnBase(ModuleBase):
    """
    The base class of collate function.
    """
    default_hyper_params = dict()

    def __init__(self, hps: Dict or None = None):
        """
        Args:
            hps: default hyper parameters in a dict (keys, values).
        """
        super(CollateFnBase, self).__init__(hps)

    @abstractmethod
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.tensor]:
        pass
