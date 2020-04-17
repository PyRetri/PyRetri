# -*- coding: utf-8 -*-

import numpy as np

from ..dim_processors_base import DimProcessorBase
from ...registry import DIMPROCESSORS
from ...utils import feature_loader
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD as SKSVD

from typing import Dict, List

@DIMPROCESSORS.register
class PartSVD(DimProcessorBase):
    """
    Part SVD will divided whole feature into several parts. Then apply SVD transformation to each part.
    It is usually used for features that extracted by several feature maps and concatenated together.

    Hyper-Params:
        proj_dim (int):  the dimension after reduction. If it is 0, then no reduction will be done
            (in SVD, we will minus origin dimension by 1).
        whiten (bool): whether do whiten for each part.
        train_fea_dir (str): the path of features for training SVD.
        l2 (bool): whether do l2-normalization for the training features.
    """
    default_hyper_params = {
        "proj_dim": 0,
        "whiten": True,
        "train_fea_dir": "unknown",
        "l2": True,
    }

    def __init__(self, feature_names: List[str], hps: Dict or None = None):
        """
        Args:
            feature_names (list): a list of features names to be loaded.
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        super(PartSVD, self).__init__(feature_names, hps)

        self.svds = dict()
        self._train(self._hyper_params["train_fea_dir"])

    def _train(self, fea_dir: str) -> None:
        """
        Train the part SVD.

        Args:
            fea_dir (str): the path of features for training part SVD.
        """
        fea, _, pos_info = feature_loader.load(fea_dir, self.feature_names)
        fea_names = list(pos_info.keys())
        ori_dim = fea.shape[1]

        already_proj_dim = 0
        for fea_name in fea_names:
            st_idx, ed_idx = pos_info[fea_name][0], pos_info[fea_name][1]
            ori_part_dim = ed_idx - st_idx
            if self._hyper_params["proj_dim"] == 0:
                proj_part_dim = ori_part_dim - 1
            else:
                ratio = self._hyper_params["proj_dim"] * 1.0 / ori_dim
                if fea_name != fea_names[-1]:
                    proj_part_dim = int(ori_part_dim * ratio)
                else:
                    proj_part_dim = self._hyper_params["proj_dim"] - already_proj_dim
                    assert proj_part_dim < ori_part_dim, "reduction dimension can not be distributed to each part!"
                already_proj_dim += proj_part_dim

            svd = SKSVD(n_components=proj_part_dim)
            train_fea = fea[:, st_idx: ed_idx]
            if self._hyper_params["l2"]:
                train_fea = normalize(train_fea, norm="l2")
            train_fea = svd.fit_transform(train_fea)
            std = train_fea.std(axis=0, keepdims=True)
            self.svds[fea_name] = {
                "pos": (st_idx, ed_idx),
                "svd": svd,
                "std": std
            }

    def __call__(self, fea: np.ndarray) -> np.ndarray:
        fea_names = np.sort(list(self.svds.keys()))
        ret = list()

        for fea_name in fea_names:
            st_idx, ed_idx = self.svds[fea_name]["pos"][0], self.svds[fea_name]["pos"][1]
            svd = self.svds[fea_name]["svd"]

            proj_fea = fea[:, st_idx: ed_idx]
            proj_fea = normalize(proj_fea, norm='l2')
            proj_fea = svd.transform(proj_fea)
            if self._hyper_params["whiten"]:
                proj_fea = proj_fea / (self.svds[fea_name]["std"] + 1e-6)
            proj_fea = normalize(proj_fea, norm='l2')

            ret.append(proj_fea)

        ret = np.concatenate(ret, axis=1)
        return ret


