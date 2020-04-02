# -*- coding: utf-8 -*-

from yacs.config import CfgNode

import torch.nn as nn

from .extractors_impl.vgg_series import VggSeries
from .extractors_impl.res_series import ResSeries
from .extractors_impl.reid_series import ReIDSeries
from .extractors_base import ExtractorBase


__all__ = [
    'ExtractorBase',
    'VggSeries', 'ResSeries',
    'ReIDSeries',
]
