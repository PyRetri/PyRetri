# -*- coding: utf-8 -*-

import torch

from ..re_ranker_base import ReRankerBase
from ...registry import RERANKERS

from typing import Dict

@RERANKERS.register
class Identity(ReRankerBase):
    """
    Directly return features without any re-rank operations.
    """
    default_hyper_params = dict()

    def __init__(self, hps: Dict or None = None):
        """
        Args:
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        super(Identity, self).__init__(hps)

    def __call__(self, query_fea: torch.tensor, gallery_fea: torch.tensor, dis: torch.tensor or None = None,
                 sorted_index: torch.tensor or None = None) -> torch.tensor:
        if sorted_index is None:
            sorted_index = torch.argsort(dis, dim=1)
        return sorted_index
