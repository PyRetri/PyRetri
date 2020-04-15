# -*- coding: utf-8 -*-

from yacs.config import CfgNode

from .registry import EVALUATORS
from .evaluator import EvaluatorBase
from .helper import EvaluateHelper

from ..utils import simple_build


def build_evaluator(cfg: CfgNode) -> EvaluatorBase:
    """
    Instantiate a evaluator class.

    Args:
        cfg (CfgNode): the configuration tree.

    Returns:
        evaluator (EvaluatorBase): a evaluator class.
    """
    name = cfg["name"]
    evaluator = simple_build(name, cfg, EVALUATORS)
    return evaluator


def build_evaluate_helper(cfg: CfgNode) -> EvaluateHelper:
    """
    Instantiate a evaluate helper class.

    Args:
        cfg (CfgNode): the configuration tree.

    Returns:
        helper (EvaluateHelper): a evaluate helper class.
    """
    evaluator = build_evaluator(cfg.evaluator)
    helper = EvaluateHelper(evaluator)
    return helper
