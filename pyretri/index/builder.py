# -*- coding: utf-8 -*-

from yacs.config import CfgNode

from .registry import ENHANCERS, METRICS, DIMPROCESSORS, RERANKERS
from .feature_enhancer import EnhanceBase
from .helper import IndexHelper
from .metric import MetricBase
from .dim_processor import DimProcessorBase
from .re_ranker import ReRankerBase

from ..utils import simple_build

from typing import List


def build_enhance(cfg: CfgNode) -> EnhanceBase:
    """
    Instantiate a feature enhancer class.

    Args:
        cfg (CfgNode): the configuration tree.

    Returns:
        enhance (EnhanceBase): an instance of feature enhancer class.
    """
    name = cfg["name"]
    enhance = simple_build(name, cfg, ENHANCERS)
    return enhance


def build_metric(cfg: CfgNode) -> MetricBase:
    """
    Instantiate a metric class.

    Args:
        cfg (CfgNode): the configuration tree.

    Returns:
        metric (MetricBase): an instance of metric class.
    """
    name = cfg["name"]
    metric = simple_build(name, cfg, METRICS)
    return metric


def build_processors(feature_names: List[str], cfg: CfgNode) -> DimProcessorBase:
    """
    Instantiate a list of dimension processor classes.

    Args:
        cfg (CfgNode): the configuration tree.

    Returns:
        processors (list): a list of instances of dimension process class.
    """
    names = cfg["names"]
    processors = list()
    for name in names:
        processors.append(simple_build(name, cfg, DIMPROCESSORS, feature_names=feature_names))
    return processors


def build_ranker(cfg: CfgNode) -> ReRankerBase:
    """
    Instantiate a re-ranker class.

    Args:
        cfg (CfgNode): the configuration tree.

    Returns:
        re_rank (list): an instance of re-ranker class.
    """
    name = cfg["name"]
    re_rank = simple_build(name, cfg, RERANKERS)
    return re_rank


def build_index_helper(cfg: CfgNode) -> IndexHelper:
    """
    Instantiate a index helper class.

    Args:
        cfg (CfgNode): the configuration tree.

    Returns:
        helper (IndexHelper): an instance of index helper class.
    """
    dim_processors = build_processors(cfg["feature_names"], cfg.dim_processors)
    metric = build_metric(cfg.metric)
    feature_enhancer = build_enhance(cfg.feature_enhancer)
    re_ranker = build_ranker(cfg.re_ranker)
    helper = IndexHelper(dim_processors, feature_enhancer, metric, re_ranker)
    return helper
