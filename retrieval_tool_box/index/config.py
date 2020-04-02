# -*- coding: utf-8 -*-

from yacs.config import CfgNode

from .registry import ENHANCERS, METRICS, DIMPROCESSORS, RERANKERS

from ..utils import get_config_from_registry


def get_enhancer_cfg() -> CfgNode:
    cfg = get_config_from_registry(ENHANCERS)
    cfg["name"] = "unknown"
    return cfg


def get_metric_cfg() -> CfgNode:
    cfg = get_config_from_registry(METRICS)
    cfg["name"] = "unknown"
    return cfg


def get_processors_cfg() -> CfgNode:
    cfg = get_config_from_registry(DIMPROCESSORS)
    cfg["names"] = ["unknown"]
    return cfg


def get_ranker_cfg() -> CfgNode:
    cfg = get_config_from_registry(RERANKERS)
    cfg["name"] = "unknown"
    return cfg


def get_index_cfg() -> CfgNode:
    cfg = CfgNode()
    cfg["query_fea_dir"] = "unknown"
    cfg["gallery_fea_dir"] = "unknown"
    cfg["feature_names"] = ["all"]
    cfg["dim_processors"] = get_processors_cfg()
    cfg["feature_enhancer"] = get_enhancer_cfg()
    cfg["metric"] = get_metric_cfg()
    cfg["re_ranker"] = get_ranker_cfg()
    return cfg
