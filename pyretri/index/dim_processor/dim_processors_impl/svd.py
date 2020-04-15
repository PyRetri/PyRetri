# -*- coding: utf-8 -*-

import numpy as np

from ..dim_processors_base import DimProcessorBase
from ...registry import DIMPROCESSORS
from ...utils import feature_loader
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD as SKSVD

from typing import Dict, List

@DIMPROCESSORS.register
class SVD(DimProcessorBase):
    """
    Do the SVD transformation for dimension reduction.

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
        super(SVD, self).__init__(feature_names, hps)

        self.svd = SKSVD(n_components=self._hyper_params["proj_dim"])
        self.std = 0.0
        self._train(self._hyper_params["train_fea_dir"])

    def _train(self, fea_dir: str) -> None:
        """
        Train the SVD.

        Args:
            fea_dir: the path of features for training SVD.
        """
        train_fea, _, _ = feature_loader.load(fea_dir, self.feature_names)
        if self._hyper_params["l2"]:
            train_fea = normalize(train_fea, norm="l2")
        train_fea = self.svd.fit_transform(train_fea)
        self.std = train_fea.std(axis=0, keepdims=True)

    def __call__(self, fea: np.ndarray) -> np.ndarray:
        ori_fea = fea
        proj_fea = self.svd.transform(ori_fea)
        if self._hyper_params["whiten"]:
            proj_fea = proj_fea / (self.std + 1e-6)
        return proj_fea
