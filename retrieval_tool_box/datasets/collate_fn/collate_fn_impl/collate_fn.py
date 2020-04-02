# -*- coding: utf-8 -*-

import torch

from ..collate_fn_base import CollateFnBase
from ...registry import COLLATEFNS
from torch.utils.data.dataloader import default_collate

from typing import Dict, List

@COLLATEFNS.register
class CollateFn(CollateFnBase):
    """
    A wrapper for torch.utils.data.dataloader.default_collate.
    """
    default_hyper_params = dict()

    def __init__(self, hps: Dict or None = None):
        """
        Args:
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        super(CollateFn, self).__init__(hps)

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.tensor]:
        assert isinstance(batch, list)
        assert isinstance(batch[0], dict)
        return default_collate(batch)
