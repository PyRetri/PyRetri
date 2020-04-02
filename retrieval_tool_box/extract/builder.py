# -*- coding: utf-8 -*-

from yacs.config import CfgNode

from .registry import AGGREGATORS, SPLITTERS, EXTRACTORS
from .extractor import ExtractorBase
from .splitter import SplitterBase
from .aggregator import AggregatorBase
from .helper import ExtractHelper

from ..utils import simple_build

import torch.nn as nn

from typing import List


def build_aggregators(cfg: CfgNode) -> List[AggregatorBase]:
    """
    Instantiate a list of aggregator classes.

    Args:
        cfg (CfgNode): the configuration tree.

    Returns:
        aggregators (list): a list of instances of aggregator class.
    """
    names = cfg["names"]
    aggregators = list()
    for name in names:
        aggregators.append(simple_build(name, cfg, AGGREGATORS))
    return aggregators


def build_extractor(model: nn.Module, cfg: CfgNode) -> ExtractorBase:
    """
    Instantiate a extractor class.

    Args:
        model (nn.Module): the model for extracting features.
        cfg (CfgNode): the configuration tree.

    Returns:
        extractor (ExtractorBase): an instance of extractor class.
    """
    name = cfg["name"]
    extractor = simple_build(name, cfg, EXTRACTORS, model=model)
    return extractor


def build_splitter(cfg: CfgNode) -> SplitterBase:
    """
    Instantiate a splitter class.

    Args:
        cfg (CfgNode): the configuration tree.

    Returns:
        splitter (SplitterBase): an instance of splitter class.
    """
    name = cfg["name"]
    splitter = simple_build(name, cfg, SPLITTERS)
    return splitter


def build_extract_helper(model: nn.Module, cfg: CfgNode) -> ExtractHelper:
    """
    Instantiate a extract helper class.

    Args:
        model (nn.Module): the model for extracting features.
        cfg (CfgNode): the configuration tree.

    Returns:
        helper (ExtractHelper): an instance of extract helper class.
    """
    assemble = cfg.assemble
    extractor = build_extractor(model, cfg.extractor)
    splitter = build_splitter(cfg.splitter)
    aggregators = build_aggregators(cfg.aggregators)
    helper = ExtractHelper(assemble, extractor, splitter, aggregators)
    return helper

