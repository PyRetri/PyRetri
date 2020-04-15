# -*- coding: utf-8 -*-

from yacs.config import CfgNode

from .registry import EXTRACTORS, SPLITTERS, AGGREGATORS

from ..utils import get_config_from_registry


def get_aggregators_cfg() -> CfgNode:
    cfg = get_config_from_registry(AGGREGATORS)
    cfg["names"] = list()
    return cfg


def get_splitter_cfg() -> CfgNode:
    cfg = get_config_from_registry(SPLITTERS)
    cfg["name"] = "unknown"
    return cfg


def get_extractor_cfg() -> CfgNode:
    cfg = get_config_from_registry(EXTRACTORS)
    cfg["name"] = "unknown"
    return cfg


def get_extract_cfg() -> CfgNode:
    cfg = CfgNode()
    cfg["assemble"] = 0
    cfg["extractor"] = get_extractor_cfg()
    cfg["splitter"] = get_splitter_cfg()
    cfg["aggregators"] = get_aggregators_cfg()
    return cfg

