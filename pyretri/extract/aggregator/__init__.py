# -*- coding: utf-8 -*-

from yacs.config import CfgNode

from .aggregators_impl.crow import Crow
from .aggregators_impl.gap import GAP
from .aggregators_impl.gem import GeM
from .aggregators_impl.gmp import GMP
from .aggregators_impl.pwa import PWA
from .aggregators_impl.r_mac import RMAC
from .aggregators_impl.scda import SCDA
from .aggregators_impl.spoc import SPoC

from .aggregators_base import AggregatorBase


__all__ = [
    'AggregatorBase',
    'Crow', 'GAP', 'GeM', 'GMP', 'PWA', 'RMAC', 'SCDA', 'SPoC',
    'build_aggregators',
]
