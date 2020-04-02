# -*- coding: utf-8 -*-

from abc import abstractmethod

import torch

from ...utils import ModuleBase

from typing import Dict


class MetricBase(ModuleBase):
    """
    The base class for similarity metric.
    """
    default_hyper_params = dict()

    def __init__(self, hps: Dict or None = None):
        """
        Args:
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        super(MetricBase, self).__init__(hps)

    @abstractmethod
    def __call__(self, query_fea: torch.tensor, gallery_fea: torch.tensor):
        pass
