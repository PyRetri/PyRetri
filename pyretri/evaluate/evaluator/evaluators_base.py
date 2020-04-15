# -*- coding: utf-8 -*-

from abc import abstractmethod

from ...utils import ModuleBase

from typing import Dict


class EvaluatorBase(ModuleBase):
    """
    The base class of evaluators which compute mAP and recall.
    """
    default_hyper_params = {}

    def __init__(self, hps: Dict or None = None):
        super(EvaluatorBase, self).__init__(hps)

    @abstractmethod
    def __call__(self, query_result: Dict, gallery_info: Dict) -> (float, Dict):
        pass
