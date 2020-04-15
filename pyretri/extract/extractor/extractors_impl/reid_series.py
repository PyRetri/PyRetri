# -*- coding: utf-8 -*-

import torch.nn as nn

from ..extractors_base import ExtractorBase
from ...registry import EXTRACTORS

from typing import Dict

@EXTRACTORS.register
class ReIDSeries(ExtractorBase):
    """
    The extractors for reid baseline models.

    Hyper-Parameters
        extract_features (list): indicates which feature maps to output. See available_feas for available feature maps.
            If it is ["all"], then all available features will be output.
    """
    default_hyper_params = {
        "extract_features": list(),
    }

    available_feas = ["output"]

    def __init__(self, model: nn.Module, hps: Dict or None = None):
        """
        Args:
            model (nn.Module): the model for extracting features.
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        children = list(model.children())
        feature_modules = {
            "output": children[1].add_block,
        }
        super(ReIDSeries, self).__init__(model, feature_modules, hps)
