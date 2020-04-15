# -*- coding: utf-8 -*-

from abc import abstractmethod

from PIL import Image

import torch

from ...utils import ModuleBase
from ...utils import Registry

from typing import Dict


class TransformerBase(ModuleBase):
    """
    The base class of data augmentation operations.
    """
    default_hyper_params = dict()

    def __init__(self, hps: Dict or None = None):
        """
        Args:
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        super(TransformerBase, self).__init__(hps)

    @abstractmethod
    def __call__(self, img: Image) -> Image or torch.tensor:
        pass
