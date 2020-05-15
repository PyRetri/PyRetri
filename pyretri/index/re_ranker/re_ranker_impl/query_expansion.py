# -*- coding: utf-8 -*-

import torch

from ..re_ranker_base import ReRankerBase
from ...registry import RERANKERS

from typing import Dict

@RERANKERS.register
class QE(ReRankerBase):
    """
    Combining the retrieved topk nearest neighbors with the original query and doing another retrieval.
    c.f. https://www.robots.ox.ac.uk/~vgg/publications/papers/chum07b.pdf

    Hyper-Params:
        qe_times (int): number of query expansion times.
        qe_k (int): number of the neighbors to be combined.
    """
    default_hyper_params = {
        "qe_times": 1,
        "qe_k": 10,
    }

    def __init__(self, hps: Dict or None = None):
        """
        Args:
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        super(QE, self).__init__(hps)

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

    def __call__(self, query_fea: torch.tensor, gallery_fea: torch.tensor, dis: torch.tensor or None = None,
                 sorted_index: torch.tensor or None = None, kr=None) -> torch.tensor:
        if sorted_index is None:
            sorted_index = torch.argsort(dis, dim=1)
        for i in range(self._hyper_params['qe_times']):
            sorted_index = sorted_index[:, :self._hyper_params['qe_k']]
            sorted_index = sorted_index.reshape(-1)
            requery_fea = gallery_fea[sorted_index].view(query_fea.shape[0], -1, query_fea.shape[1]).sum(dim=1)
            requery_fea = requery_fea + query_fea
            query_fea = requery_fea
            dis = self._cal_dis(query_fea, gallery_fea)

            if kr is None:
                sorted_index = torch.argsort(dis, dim=1)
            else:
                sorted_index = kr(query_fea, gallery_fea, dis)

        return sorted_index
