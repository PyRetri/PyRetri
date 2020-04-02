# -*- coding: utf-8 -*-

from ..extractors_base import ExtractorBase
from ...registry import EXTRACTORS

from typing import Dict

@EXTRACTORS.register
class VggSeries(ExtractorBase):
    """
    The extractors for VGG Net.

    Hyper-Parameters
        extract_features (list): indicates which feature maps to output. See available_feas for available feature maps.
            If it is ["all"], then all available features will be output.
    """
    default_hyper_params = {
        "extract_features": list(),
    }

    available_feas = ["fc", "pool5", "pool4"]

    def __init__(self, model, hps: Dict or None = None):
        """
        Args:
            model (nn.Module): the model for extracting features.
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        feature_modules = {
            "fc": model.classifier[4],
            "pool5": model.features[-1],
            "pool4": model.features[23]
        }
        super(VggSeries, self).__init__(model, feature_modules, hps)
