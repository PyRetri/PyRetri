# -*- coding: utf-8 -*-

import numpy as np

import torch

from ..evaluators_base import EvaluatorBase
from ...registry import EVALUATORS
from sklearn.metrics import average_precision_score

from typing import Dict, List


@EVALUATORS.register
class ReIDOverAll(EvaluatorBase):
    """
    A evaluator for Re-ID task mAP and recall computation.

    Hyper-Params
        recall_k (sequence): positions of recalls to be calculated.
    """
    default_hyper_params = {
        "recall_k": [1, 2, 4, 8],
    }

    def __init__(self, hps: Dict or None = None):
        """
        Args:
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        super(ReIDOverAll, self).__init__(hps)
        self._hyper_params["recall_k"] = np.sort(self._hyper_params["recall_k"])

    def compute_ap_cmc(self, index: np.ndarray, good_index: np.ndarray, junk_index: np.ndarray) -> (float, torch.tensor):
        """
        Calculate the ap and cmc for one query.

        Args:
            index (np.ndarray): the sorted retrieval index for one query.
            good_index (np.ndarray): the index for good matching.
            junk_index (np.ndarray): the index for junk matching.

        Returns:
            tupele (float, torch.tensor): (ap, cmc), ap and cmc for one query.
        """
        ap = 0
        cmc = torch.IntTensor(len(index)).zero_()
        if good_index.size == 0:
            cmc[0] = -1
            return ap, cmc

        # remove junk_index
        mask = np.in1d(index, junk_index, invert=True)
        index = index[mask]

        # find good_index index
        ngood = len(good_index)
        mask = np.in1d(index, good_index)
        rows_good = np.argwhere(mask == True)
        rows_good = rows_good.flatten()

        cmc[rows_good[0]:] = 1
        for i in range(ngood):
            d_recall = 1.0 / ngood
            precision = (i + 1) * 1.0 / (rows_good[i] + 1)
            if rows_good[i] != 0:
                old_precision = i * 1.0 / rows_good[i]
            else:
                old_precision = 1.0
            ap = ap + d_recall * (old_precision + precision) / 2

        return ap, cmc

    def evaluate_once(self, index: np.ndarray, ql: int, qc: int, gl: np.ndarray, gc: np.ndarray) -> (float, torch.tensor):
        """
        Generate the indexes and calculate the ap and cmc for one query.

        Args:
            index (np.ndarray): the sorted retrieval index for one query.
            ql (int): the person id of the query.
            qc (int): the camera id of the query.
            gl (np.ndarray): the person ids of the gallery set.
            gc (np.ndarray): the camera ids of the gallery set.

        Returns:
            tuple (float, torch.tensor): ap and cmc for one query.
        """
        query_index = (ql == gl)
        query_index = np.argwhere(query_index)

        camera_index = (qc == gc)
        camera_index = np.argwhere(camera_index)

        # good index
        good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)

        # junk index
        junk_index1 = np.argwhere(gl == -1)
        junk_index2 = np.intersect1d(query_index, camera_index)
        junk_index = np.append(junk_index2, junk_index1)

        AP_tmp, CMC_tmp = self.compute_ap_cmc(index, good_index, junk_index)

        return AP_tmp, CMC_tmp

    def __call__(self, query_result: List, gallery_info: List) -> (float, Dict):
        """
        Calculate the mAP and recall for the indexing results.

        Args:
            query_result (list): a list of indexing results.
            gallery_info (list): a list of gallery set information.

        Returns:
            tuple (float, dict): mean average precision and recall for each position.
        """
        AP = 0.0
        CMC = torch.IntTensor(range(len(gallery_info))).zero_()
        gallery_label = np.array([int(gallery_info[idx]["label"]) for idx in range(len(gallery_info))])
        gallery_cam = np.array([int(gallery_info[idx]["cam"]) for idx in range(len(gallery_info))])

        recall_at_k = dict()
        for k in self._hyper_params["recall_k"]:
            recall_at_k[k] = 0

        for i in range(len(query_result)):
            AP_tmp, CMC_tmp = self.evaluate_once(np.array(query_result[i]["ranked_neighbors_idx"]),
                                                 int(query_result[i]["label"]), int(query_result[i]["cam"]),
                                                 gallery_label, gallery_cam)
            if CMC_tmp[0] == -1:
                continue
            CMC = CMC + CMC_tmp
            AP += AP_tmp

        CMC = CMC.float()
        CMC = CMC / len(query_result)  # average CMC

        for k in recall_at_k:
            recall_at_k[k] = (CMC[k-1] * 100).item()

        mAP = AP / len(query_result) * 100

        return mAP, recall_at_k

