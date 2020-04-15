# -*- coding: utf-8 -*-

import numpy as np

from ..dim_processors_base import DimProcessorBase
from ...registry import DIMPROCESSORS
from sklearn.preprocessing import normalize

from typing import Dict, List

@DIMPROCESSORS.register
class L2Normalize(DimProcessorBase):
    """
    L2 normalize the features.
    """
    default_hyper_params = dict()

    def __init__(self, feature_names: List[str], hps: Dict or None = None):
        """
        Args:
            feature_names (list): a list of features names to be loaded.
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        super(L2Normalize, self).__init__(feature_names, hps)

    def __call__(self, fea: np.ndarray) -> np.ndarray:
        return normalize(fea, norm="l2")
