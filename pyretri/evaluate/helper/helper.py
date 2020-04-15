# -*- coding: utf-8 -*-

import torch
from ..evaluator import EvaluatorBase

from typing import Dict, List

class EvaluateHelper:
    """
    A helper class to evaluate query results.
    """
    def __init__(self, evaluator: EvaluatorBase):
        """
        Args:
            evaluator: a evaluator class.
        """
        self.evaluator = evaluator
        self.recall_k = evaluator.default_hyper_params["recall_k"]

    def show_results(self, mAP: float, recall_at_k: Dict) -> None:
        """
        Show the evaluate results.

        Args:
            mAP (float): mean average precision.
            recall_at_k (Dict): recall at the k position.
        """
        repr_str = "mAP: {:.1f}\n".format(mAP)

        for k in self.recall_k:
            repr_str += "R@{}: {:.1f}\t".format(k, recall_at_k[k])

        print('--------------- Retrieval Evaluation ------------')
        print(repr_str)

    def do_eval(self, query_result_info: List, gallery_info: List) -> (float, Dict):
        """
        Get the evaluate results.

        Args:
            query_result_info (list): a list of indexing results.
            gallery_info (list): a list of gallery set information.

        Returns:
            tuple (float, Dict): mean average precision and recall for each position.
        """
        mAP, recall_at_k = self.evaluator(query_result_info, gallery_info)

        return mAP, recall_at_k
