# -*- coding: utf-8 -*-

import torch
import numpy as np

from ..aggregators_base import AggregatorBase
from ...registry import AGGREGATORS
from ....index.utils import feature_loader

from typing import Dict

@AGGREGATORS.register
class PWA(AggregatorBase):
    """
    Part-based Weighting Aggregation.
    c.f. https://arxiv.org/abs/1705.01247

    Hyper-Params
        train_fea_dir (str): path of feature dir for selecting channels.
        n_proposal (int): number of proposals to be selected.
        alpha (float): alpha for calculate spatial weight.
        beta (float): beta for calculate spatial weight.
    """

    default_hyper_params = {
        "train_fea_dir": "",
        "n_proposal": 25,
        "alpha": 2.0,
        "beta": 2.0,
        "train_fea_names": ["pool5_GAP"],
    }

    def __init__(self, hps: Dict or None = None):
        """
        Args:
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        super(PWA, self).__init__(hps)
        self.first_show = True
        assert self._hyper_params["train_fea_dir"] != ""
        self.selected_proposals_idx = None
        self.train()

    def train(self) -> None:
        n_proposal = self._hyper_params["n_proposal"]
        stacked_fea, _, pos_info = feature_loader.load(
            self._hyper_params["train_fea_dir"],
            self._hyper_params["train_fea_names"]
        )
        self.selected_proposals_idx = dict()
        for fea_name in pos_info:
            st_idx, ed_idx = pos_info[fea_name]
            fea = stacked_fea[:, st_idx: ed_idx]
            assert fea.ndim == 2, "invalid train feature"
            channel_variance = np.std(fea, axis=0)
            selected_idx = channel_variance.argsort()[-n_proposal:]
            fea_name = "_".join(fea_name.split("_")[:-1])
            self.selected_proposals_idx[fea_name] = selected_idx.tolist()

    def __call__(self, features: Dict[str, torch.tensor]) -> Dict[str, torch.tensor]:
        alpha, beta = self._hyper_params["alpha"], self._hyper_params["beta"]
        ret = dict()
        for key in features:
            fea = features[key]
            if fea.ndimension() == 4:
                assert (key in self.selected_proposals_idx), '{} is not in the {}'.format(key, self.selected_proposals_idx.keys())
                proposals_idx = np.array(self.selected_proposals_idx[key])
                proposals = fea[:, proposals_idx, :, :]
                power_norm = (proposals ** alpha).sum(dim=(2, 3), keepdims=True) ** (1.0 / alpha)
                normed_proposals = (proposals / (power_norm + 1e-5)) ** (1.0 / beta)
                fea = (fea[:, None, :, :, :] * normed_proposals[:, :, None, :, :]).sum(dim=(3, 4))
                fea = fea.view(fea.shape[0], -1)
                ret[key + "_{}".format(self.__class__.__name__)] = fea
            else:
                # In case of fc feature.
                assert fea.ndimension() == 2
                if self.first_show:
                    print("[PWA Aggregator]: find 2-dimension feature map, skip aggregation")
                    self.first_show = False
                ret[key] = fea
        return ret
