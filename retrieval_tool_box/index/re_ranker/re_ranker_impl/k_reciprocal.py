# -*- coding: utf-8 -*-

import torch
import numpy as np

from ..re_ranker_base import ReRankerBase
from ...registry import RERANKERS

from typing import Dict

@RERANKERS.register
class KReciprocal(ReRankerBase):
    """
    Encoding k-reciprocal nearest neighbors to enhance the performance of retrieval.
    c.f. https://arxiv.org/pdf/1701.08398.pdf

    Hyper-Params:
        k1 (int): hyper-parameter for calculating jaccard distance.
        k2 (int): hyper-parameter for calculating local query expansion.
        lambda_value (float): hyper-parameter for calculating the final distance.
    """
    default_hyper_params = {
        "k1": 20,
        "k2": 6,
        "lambda_value": 0.3,
    }

    def __init__(self, hps: Dict or None = None):
        """
        Args:
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        super(KReciprocal, self).__init__(hps)

    def _cal_dis(self, query_fea: torch.tensor, gallery_fea: torch.tensor) -> torch.tensor:
        """
        Calculate the distance between query set features and gallery set features.

        Args:
            query_fea (torch.tensor): query set features.
            gallery_fea (torch.tensor): gallery set features.

        Returns:
            dis (torch.tensor): the distance between query set features and gallery set features.
        """
        query_fea = query_fea.transpose(1, 0)
        inner_dot = gallery_fea.mm(query_fea)
        dis = (gallery_fea ** 2).sum(dim=1, keepdim=True) + (query_fea ** 2).sum(dim=0, keepdim=True)
        dis = dis - 2 * inner_dot
        dis = dis.transpose(1, 0)
        return dis

    def __call__(self, query_fea: torch.tensor, gallery_fea: torch.tensor,  dis: torch.tensor or None = None,
                 sorted_index: torch.tensor or None = None) -> torch.tensor or np.ndarray:
        # The following naming, e.g. gallery_num, is different from outer scope.
        # Don't care about it.
        q_g_dist = dis.cpu().numpy()
        g_g_dist = self._cal_dis(gallery_fea, gallery_fea).cpu().numpy()
        q_q_dist = self._cal_dis(query_fea, query_fea).cpu().numpy()

        original_dist = np.concatenate(
            [np.concatenate([q_q_dist, q_g_dist], axis=1),
             np.concatenate([q_g_dist.T, g_g_dist], axis=1)],
            axis=0)
        original_dist = np.power(original_dist, 2).astype(np.float32)
        original_dist = np.transpose(1. * original_dist / np.max(original_dist, axis=0))
        V = np.zeros_like(original_dist).astype(np.float32)
        initial_rank = np.argsort(original_dist).astype(np.int32)

        query_num = q_g_dist.shape[0]
        gallery_num = q_g_dist.shape[0] + q_g_dist.shape[1]
        all_num = gallery_num

        for i in range(all_num):
            # k-reciprocal neighbors
            forward_k_neigh_index = initial_rank[i, :self._hyper_params["k1"] + 1]
            backward_k_neigh_index = initial_rank[forward_k_neigh_index, :self._hyper_params["k1"] + 1]
            fi = np.where(backward_k_neigh_index == i)[0]
            k_reciprocal_index = forward_k_neigh_index[fi]
            k_reciprocal_expansion_index = k_reciprocal_index
            for j in range(len(k_reciprocal_index)):
                candidate = k_reciprocal_index[j]
                candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(self._hyper_params["k1"] / 2.)) + 1]
                candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                                   :int(np.around(self._hyper_params["k1"] / 2.)) + 1]
                fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
                candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
                if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2. / 3 * len(
                        candidate_k_reciprocal_index):
                    k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

            k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
            weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
            V[i, k_reciprocal_expansion_index] = 1. * weight / np.sum(weight)
        original_dist = original_dist[:query_num, ]
        if self._hyper_params["k2"] != 1:
            V_qe = np.zeros_like(V, dtype=np.float32)
            for i in range(all_num):
                V_qe[i, :] = np.mean(V[initial_rank[i, :self._hyper_params["k2"]], :], axis=0)
            V = V_qe
            del V_qe
        del initial_rank
        invIndex = []
        for i in range(gallery_num):
            invIndex.append(np.where(V[:, i] != 0)[0])

        jaccard_dist = np.zeros_like(original_dist, dtype=np.float32)

        for i in range(query_num):
            temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float32)
            indNonZero = np.where(V[i, :] != 0)[0]
            indImages = [invIndex[ind] for ind in indNonZero]
            for j in range(len(indNonZero)):
                temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                                   V[indImages[j], indNonZero[j]])
            jaccard_dist[i] = 1 - temp_min / (2. - temp_min)

        final_dist = jaccard_dist * (1 - self._hyper_params["lambda_value"]) + original_dist * self._hyper_params[
            "lambda_value"]
        del original_dist, V, jaccard_dist
        final_dist = final_dist[:query_num, query_num:]

        if torch.cuda.is_available():
            final_dist = torch.Tensor(final_dist).cuda()
            sorted_idx = torch.argsort(final_dist, dim=1)
        else:
            sorted_idx = np.argsort(final_dist, axis=1)
        return sorted_idx
