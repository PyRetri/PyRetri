# -*- coding: utf-8 -*-

import numpy as np

from ..dim_processors_base import DimProcessorBase
from ...registry import DIMPROCESSORS
from ...utils import feature_loader
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

from typing import Dict, List

@DIMPROCESSORS.register
class RMACPCA(DimProcessorBase):
    """
    Do the PCA transformation for R-MAC only.
    When call this transformation, each part feature is processed by l2-normalize, PCA and l2-normalize.
    Then the global feature is processed by l2-normalize.

    Hyper-Params:
        proj_dim: int. The dimension after reduction. If it is 0, then no reduction will be done.
        whiten: bool, whether do whiten for each part.
        train_fea: str, feature directory for training PCA.
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
        super(RMACPCA, self).__init__(feature_names, hps)

        self.pca = dict()
        self._train(self._hyper_params["train_fea_dir"])

    def _train(self, fea_dir: str) -> None:
        """
        Train the PCA for R-MAC.

        Args:
            fea_dir (str): the path of features for training PCA.
        """
        fea, _, pos_info = feature_loader.load(fea_dir, self.feature_names)

        fea_names = np.sort(list(pos_info.keys()))
        region_feas = list()
        for fea_name in fea_names:
            st_idx, ed_idx = pos_info[fea_name][0], pos_info[fea_name][1]
            region_fea = fea[:, st_idx: ed_idx]
            region_feas.append(region_fea)

        train_fea = np.concatenate(region_feas, axis=0)
        if self._hyper_params["l2"]:
            train_fea = normalize(train_fea, norm="l2")
        pca = PCA(n_components=self._hyper_params["proj_dim"], whiten=self._hyper_params["whiten"])
        pca.fit(train_fea)

        self.pca = {
            "pca": pca,
            "pos_info": pos_info
        }

    def __call__(self, fea: np.ndarray) -> np.ndarray:
        pca = self.pca["pca"]
        pos_info = self.pca["pos_info"]

        fea_names = np.sort(list(pos_info.keys()))

        final_fea = None
        for fea_name in fea_names:
            st_idx, ed_idx = pos_info[fea_name][0], pos_info[fea_name][1]
            region_fea = fea[:, st_idx: ed_idx]
            region_fea = normalize(region_fea)
            region_fea = pca.transform(region_fea)
            region_fea = normalize(region_fea)
            if final_fea is None:
                final_fea = region_fea
            else:
                final_fea += region_fea
        final_fea = normalize(final_fea)

        return final_fea