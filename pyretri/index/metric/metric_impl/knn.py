# -*- coding: utf-8 -*-

import torch

from ..metric_base import MetricBase
from ...registry import METRICS

from typing import Dict

@METRICS.register
class KNN(MetricBase):
    """
    Similarity measure based on the euclidean distance.

    Hyper-Params:
        top_k (int): top_k nearest neighbors will be output in sorted order. If it is 0, all neighbors will be output.
    """
    default_hyper_params = {
        "top_k": 0,
    }

    def __init__(self, hps: Dict or None = None):
        """
        Args:
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        super(KNN, self).__init__(hps)

    def _cal_dis(self, query_fea: torch.tensor, gallery_fea: torch.tensor) -> torch.tensor:
        """
        Calculate the distance between query set features and gallery set features.

        Args:
            query_fea (torch.tensor): query set features.
            gallery_fea (torch.tensor): gallery set features.

        Returns:
            dis (torch.tensor): the distance between query set features and gallery set features.
        """
        query_fea = query_fea.transpose(1, 0)
        inner_dot = gallery_fea.mm(query_fea)
        dis = (gallery_fea ** 2).sum(dim=1, keepdim=True) + (query_fea ** 2).sum(dim=0, keepdim=True)
        dis = dis - 2 * inner_dot
        dis = dis.transpose(1, 0)
        return dis

    def __call__(self, query_fea: torch.tensor, gallery_fea: torch.tensor) -> (torch.tensor, torch.tensor):

        dis = self._cal_dis(query_fea, gallery_fea)
        sorted_index = torch.argsort(dis, dim=1)
        if self._hyper_params["top_k"] != 0:
            sorted_index = sorted_index[:, :self._hyper_params["top_k"]]
        return dis, sorted_index
