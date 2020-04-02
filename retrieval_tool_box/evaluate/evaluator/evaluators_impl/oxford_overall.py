# -*- coding: utf-8 -*-

import os

import numpy as np

from ..evaluators_base import EvaluatorBase
from ...registry import EVALUATORS

from typing import Dict, List


@EVALUATORS.register
class OxfordOverAll(EvaluatorBase):
    """
    A evaluator for Oxford mAP and recall computation.

    Hyper-Params
        gt_dir (str): the path of the oxford ground truth.
        recall_k (sequence): positions of recalls to be calculated.
    """
    default_hyper_params = {
        "gt_dir": "/data/cbir/oxford/gt",
        "recall_k": [1, 2, 4, 8],
    }

    def __init__(self, hps: Dict or None = None):
        """
        Args:
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        super(OxfordOverAll, self).__init__(hps)
        assert os.path.exists(self._hyper_params["gt_dir"]), 'the ground truth files must be existed!'

    @staticmethod
    def _load_tag_set(file: str) -> set:
        """
        Read information from the txt file.

        Args:
            file (str): the path of the txt file.

        Returns:
            ret (set): the information.
        """
        ret = set()
        with open(file, "r") as f:
            for line in f.readlines():
                ret.add(line.strip())
        return ret

    def compute_ap(self, query_tag: str, ranked_tags: List[str], recall_at_k: Dict) -> float:
        """
        Calculate the ap for one query.

        Args:
            query_tag (str): name of the query image.
            ranked_tags (list): a list of label of the indexing results.
            recall_at_k (dict): positions of recalls to be calculated.

        Returns:
            ap (float): ap for one query.
        """
        gt_prefix = os.path.join(self._hyper_params["gt_dir"], query_tag)
        good_set = self._load_tag_set(gt_prefix + "_good.txt")
        ok_set = self._load_tag_set(gt_prefix + "_ok.txt")
        junk_set = self._load_tag_set(gt_prefix + "_junk.txt")
        pos_set = set.union(good_set, ok_set)

        old_recall = 0.0
        old_precision = 1.0
        ap = 0.0
        intersect_size = 0.0
        i = 0
        first_tp = -1

        for tag in ranked_tags:
            if tag in junk_set:
                continue
            if tag in pos_set:
                intersect_size += 1

                # Remember that in oxford query mode, the first element in rank_list is the query itself.
                if first_tp == -1:
                    first_tp = i

            recall = intersect_size * 1.0 / len(pos_set)
            precision = intersect_size / (i + 1.0)

            ap += (recall - old_recall) * ((old_precision + precision) / 2.0)
            old_recall = recall
            old_precision = precision
            i += 1

        if first_tp != -1:
            ks = self._hyper_params["recall_k"]
            for k in ks:
                if k >= first_tp + 1:
                    recall_at_k[k] = recall_at_k[k] + 1

        return ap

    def __call__(self, query_result: List, gallery_info: List) -> (float, Dict):
        """
        Calculate the mAP and recall for the indexing results.

        Args:
            query_result (list): a list of indexing results.
            gallery_info (list): a list of gallery set information.

        Returns:
            tuple (float, dict): mean average precision and recall for each position.
        """
        aps = list()

        recall_at_k = dict()
        for k in self._hyper_params["recall_k"]:
            recall_at_k[k] = 0

        for i in range(len(query_result)):
            ranked_idx = query_result[i]["ranked_neighbors_idx"]
            ranked_tags = list()
            for idx in ranked_idx:
                ranked_tags.append(gallery_info[idx]["label"])
            aps.append(self.compute_ap(query_result[i]["query_name"], ranked_tags, recall_at_k))

        mAP = np.mean(aps) * 100

        for k in recall_at_k:
            recall_at_k[k] = recall_at_k[k] * 100 / len(query_result)

        return mAP, recall_at_k
