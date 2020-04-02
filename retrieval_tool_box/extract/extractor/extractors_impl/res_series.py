# -*- coding: utf-8 -*-

from ..extractors_base import ExtractorBase
from ...registry import EXTRACTORS

from typing import Dict

@EXTRACTORS.register
class ResSeries(ExtractorBase):
    """
    The extractors for ResNet.

    Hyper-Parameters
        extract_features (list): indicates which feature maps to output. See available_feas for available feature maps.
            If it is ["all"], then all available features will be output.
    """
    default_hyper_params = {
        "extract_features": list(),
    }

    available_feas = ["pool5", "pool4", "pool3"]

    def __init__(self, model, hps: Dict or None = None):
        """
        Args:
            model (nn.Module): the model for extracting features.
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        children = list(model.children())
        feature_modules = {
            "pool5": children[-3][-1].relu,
            "pool4": children[-4][-1].relu,
            "pool3": children[-5][-1].relu
        }
        super(ResSeries, self).__init__(model, feature_modules, hps)
