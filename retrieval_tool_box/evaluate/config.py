# -*- coding: utf-8 -*-

from yacs.config import CfgNode

from .registry import EVALUATORS

from ..utils import get_config_from_registry


def get_evaluator_cfg() -> CfgNode:
    cfg = get_config_from_registry(EVALUATORS)
    cfg["name"] = "unknown"
    return cfg

def get_evaluate_cfg() -> CfgNode:
    cfg = CfgNode()
    cfg["evaluator"] = get_evaluator_cfg()
    return cfg
