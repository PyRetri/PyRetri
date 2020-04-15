# -*- coding: utf-8 -*-

import torch

from ..aggregators_base import AggregatorBase
from ...registry import AGGREGATORS

from typing import Dict

@AGGREGATORS.register
class GeM(AggregatorBase):
    """
    Generalized-mean pooling.
    c.f. https://pdfs.semanticscholar.org/a2ca/e0ed91d8a3298b3209fc7ea0a4248b914386.pdf

    Hyper-Params
        p (float): hyper-parameter for calculating generalized mean. If p = 1, GeM is equal to global average pooling, and
            if p = +infinity, GeM is equal to global max pooling.
    """
    default_hyper_params = {
        "p": 3.0,
    }

    def __init__(self, hps: Dict or None = None):
        """
        Args:
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        self.first_show = True
        super(GeM, self).__init__(hps)

    def __call__(self, features: Dict[str, torch.tensor]) -> Dict[str, torch.tensor]:
        p = self._hyper_params["p"]

        ret = dict()
        for key in features:
            fea = features[key]
            if fea.ndimension() == 4:
                fea = fea ** p
                h, w = fea.shape[2:]
                fea = fea.sum(dim=(2, 3)) * 1.0 / w / h
                fea = fea ** (1.0 / p)
                ret[key + "_{}".format(self.__class__.__name__)] = fea
            else:
                # In case of fc feature.
                assert fea.ndimension() == 2
                if self.first_show:
                    print("[GeM Aggregator]: find 2-dimension feature map, skip aggregation")
                    self.first_show = False
                ret[key] = fea
        return ret
