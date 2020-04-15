# -*- coding: utf-8 -*-

import torch
import numpy as np

from ..aggregators_base import AggregatorBase
from ...registry import AGGREGATORS

from typing import Dict

@AGGREGATORS.register
class SPoC(AggregatorBase):
    """
    SPoC with center prior.
    c.f. https://arxiv.org/pdf/1510.07493.pdf
    """
    default_hyper_params = dict()

    def __init__(self, hps: Dict or None = None):
        """
        Args:
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        super(SPoC, self).__init__(hps)
        self.first_show = True
        self.spatial_weight_cache = dict()

    def __call__(self, features: Dict[str, torch.tensor]) -> Dict[str, torch.tensor]:
        ret = dict()
        for key in features:
            fea = features[key]
            if fea.ndimension() == 4:
                h, w = fea.shape[2:]
                if (h, w) in self.spatial_weight_cache:
                    spatial_weight = self.spatial_weight_cache[(h, w)]
                else:
                    sigma = min(h, w) / 2.0 / 3.0
                    x = torch.Tensor(range(w))
                    y = torch.Tensor(range(h))[:, None]
                    spatial_weight = torch.exp(-((x - (w - 1) / 2.0) ** 2 + (y - (h - 1) / 2.0) ** 2) / 2.0 / (sigma ** 2))
                    if torch.cuda.is_available():
                        spatial_weight = spatial_weight.cuda()
                    spatial_weight = spatial_weight[None, None, :, :]
                    self.spatial_weight_cache[(h, w)] = spatial_weight
                fea = (fea * spatial_weight).sum(dim=(2, 3))
                ret[key + "_{}".format(self.__class__.__name__)] = fea
            else:
                # In case of fc feature.
                assert fea.ndimension() == 2
                if self.first_show:
                    print("[SPoC Aggregator]: find 2-dimension feature map, skip aggregation")
                    self.first_show = False
                ret[key] = fea
        return ret
