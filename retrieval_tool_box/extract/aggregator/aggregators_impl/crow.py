# -*- coding: utf-8 -*-

import torch

from ..aggregators_base import AggregatorBase
from ...registry import AGGREGATORS

from typing import Dict

@AGGREGATORS.register
class Crow(AggregatorBase):
    """
    Cross-dimensional Weighting for Aggregated Deep Convolutional Features.
    c.f. https://arxiv.org/pdf/1512.04065.pdf

    Hyper-Params
        spatial_a (float): hyper-parameter for calculating spatial weight.
        spatial_b (float): hyper-parameter for calculating spatial weight.
    """
    default_hyper_params = {
        "spatial_a": 2.0,
        "spatial_b": 2.0,
    }

    def __init__(self, hps: Dict or None = None):
        """
        Args:
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        self.first_show = True
        super(Crow, self).__init__(hps)

    def __call__(self, features: Dict[str, torch.tensor]) -> Dict[str, torch.tensor]:
        spatial_a = self._hyper_params["spatial_a"]
        spatial_b = self._hyper_params["spatial_b"]

        ret = dict()
        for key in features:
            fea = features[key]
            if fea.ndimension() == 4:
                spatial_weight = fea.sum(dim=1, keepdims=True)
                z = (spatial_weight ** spatial_a).sum(dim=(2, 3), keepdims=True)
                z = z ** (1.0 / spatial_a)
                spatial_weight = (spatial_weight / z) ** (1.0 / spatial_b)

                c, w, h = fea.shape[1:]
                nonzeros = (fea!=0).float().sum(dim=(2, 3)) / 1.0 / (w * h) + 1e-6
                channel_weight = torch.log(nonzeros.sum(dim=1, keepdims=True) / nonzeros)

                fea = fea * spatial_weight
                fea = fea.sum(dim=(2, 3))
                fea = fea * channel_weight

                ret[key + "_{}".format(self.__class__.__name__)] = fea
            else:
                # In case of fc feature.
                assert fea.ndimension() == 2
                if self.first_show:
                    print("[Crow Aggregator]: find 2-dimension feature map, skip aggregation")
                    self.first_show = False
                ret[key] = fea
        return ret
