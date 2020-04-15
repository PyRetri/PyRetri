# -*- coding: utf-8 -*-

import torch

from ..aggregators_base import AggregatorBase
from ...registry import AGGREGATORS

from typing import Dict

@AGGREGATORS.register
class GAP(AggregatorBase):
    """
    Global average pooling.
    """
    default_hyper_params = dict()

    def __init__(self, hps: Dict or None = None):
        """
        Args:
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        self.first_show = True
        super(GAP, self).__init__(hps)

    def __call__(self, features: Dict[str, torch.tensor]) -> Dict[str, torch.tensor]:
        ret = dict()
        for key in features:
            fea = features[key]
            if fea.ndimension() == 4:
                fea = fea.mean(dim=3).mean(dim=2)
                ret[key + "_{}".format(self.__class__.__name__)] = fea
            else:
                # In case of fc feature.
                assert fea.ndimension() == 2
                if self.first_show:
                    print("[GAP Aggregator]: find 2-dimension feature map, skip aggregation")
                    self.first_show = False
                ret[key] = fea
        return ret
