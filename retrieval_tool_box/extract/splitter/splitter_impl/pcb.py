# -*- coding: utf-8 -*-

import torch
import numpy as np

from ..splitter_base import SplitterBase
from ...registry import SPLITTERS

from typing import Dict


@SPLITTERS.register
class PCB(SplitterBase):
    """
    PCB function to split feature maps.
    c.f. http://openaccess.thecvf.com/content_ECCV_2018/papers/Yifan_Sun_Beyond_Part_Models_ECCV_2018_paper.pdf

    Hyper-Params:
        stripe_num (int): the number of stripes divided.
    """
    default_hyper_params = {
        'stripe_num': 2,
    }

    def __init__(self, hps: Dict or None = None):
        """
        Args:
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        super(PCB, self).__init__(hps)

    def __call__(self, features: torch.tensor) -> Dict:
        ret = dict()
        for key in features:
            fea = features[key]
            assert fea.ndimension() == 4
            assert self.default_hyper_params["stripe_num"] <= fea.shape[2], \
                'stripe num must be less than or equal to the height of fea'

            stride = fea.shape[2] // self.default_hyper_params["stripe_num"]

            for i in range(int(self.default_hyper_params["stripe_num"])):
                ret[key + "_part_{}".format(i)] = fea[:, :, stride * i: stride * (i + 1), :]
            ret[key + "_global"] = fea
        return ret
