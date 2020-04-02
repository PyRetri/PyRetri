# -*- coding: utf-8 -*-

from abc import abstractmethod

import numpy as np

from ...utils import ModuleBase

from typing import Dict, List


class DimProcessorBase(ModuleBase):
    """
    The base class of dimension processor.
    """
    default_hyper_params = dict()

    def __init__(self, feature_names: List[str], hps: Dict or None = None):
        """
        Args:
            feature_names (list): a list of features names to be loaded.
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        ModuleBase.__init__(self, hps)
        self.feature_names = feature_names

    @abstractmethod
    def __call__(self, fea: np.ndarray) -> np.ndarray:
        pass
