# -*- coding: utf-8 -*-

import numpy as np

from ..dim_processors_base import DimProcessorBase
from ...registry import DIMPROCESSORS

from typing import Dict, List

@DIMPROCESSORS.register
class Identity(DimProcessorBase):
    """
    Directly return feature without any dimension process operations.
    """
    default_hyper_params = dict()

    def __init__(self, feature_names: List[str], hps: Dict or None = None):
        """
        Args:
            feature_names (list): a list of features names to be loaded.
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        super(Identity, self).__init__(feature_names, hps)

    def __call__(self, fea: np.ndarray) -> np.ndarray:
        return fea
