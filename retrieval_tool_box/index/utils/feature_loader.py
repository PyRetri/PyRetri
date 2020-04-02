# -*- coding: utf-8 -*-

import os
import pickle

import numpy as np

from typing import Dict, List

class FeatureLoader:
    """
    A class for load features and information.
    """
    def __init__(self):
        self.feature_cache = dict()

    def _load_from_cache(self, fea_dir: str, feature_names: List[str]) -> (np.ndarray, Dict, Dict):
        """
        Load feature and its information from cache.

        Args:
            fea_dir (str): the path of features to be loaded.
            feature_names (list): a list of str indicating which feature will be output.

        Returns:
            tuple (np.ndarray, Dict, Dict): a stacked feature, a list of dicts which describes the image information of each feature,
                and a dict map from feature name to its position.
        """
        assert fea_dir in self.feature_cache, "feature in {} not cached!".format(fea_dir)

        feature_dict = self.feature_cache[fea_dir]["feature_dict"]
        info_dicts = self.feature_cache[fea_dir]["info_dicts"]
        stacked_feature = list()
        pos_info = dict()

        if len(feature_names) == 1 and feature_names[0] == "all":
            feature_names = list(feature_dict.keys())
        feature_names = np.sort(feature_names)

        st_idx = 0
        for name in feature_names:
            assert name in feature_dict, "invalid feature name: {} not in {}!".format(name, feature_dict.keys())
            stacked_feature.append(feature_dict[name])
            pos_info[name] = (st_idx, st_idx + stacked_feature[-1].shape[1])
            st_idx = st_idx + stacked_feature[-1].shape[1]
        stacked_feature = np.concatenate(stacked_feature, axis=1)

        print("[LoadFeature] Success, total {} images, \n feature names: {}".format(
            len(info_dicts),
            pos_info.keys())
        )
        return stacked_feature, info_dicts, pos_info

    def load(self, fea_dir: str, feature_names: List[str]) -> (np.ndarray, Dict, Dict):
        """
        Load and concat feature from feature directory.

        Args:
            fea_dir (str): the path of features to be loaded.
            feature_names (list): a list of str indicating which feature will be output.

        Returns:
            tuple (np.ndarray, Dict, Dict): a stacked feature, a list of dicts which describes the image information of each feature,
                and a dict map from feature name to its position.

        """
        assert os.path.exists(fea_dir), "non-exist feature path: {}".format(fea_dir)

        if fea_dir in self.feature_cache:
            return self._load_from_cache(fea_dir, feature_names)

        feature_dict = dict()
        info_dicts = list()

        for root, dirs, files in os.walk(fea_dir):
            for file in files:
                if file.endswith(".json"):
                    print("[LoadFeature]: loading feature from {}...".format(os.path.join(root, file)))
                    with open(os.path.join(root, file), "rb") as f:
                        part_info = pickle.load(f)
                        for info in part_info["info_dicts"]:
                            for key in info["feature"].keys():
                                if key not in feature_dict:
                                    feature_dict[key] = list()
                                feature_dict[key].append(info["feature"][key])
                            del info["feature"]
                            info_dicts.append(info)
        for key, fea in feature_dict.items():
            fea = np.array(fea)
            feature_dict[key] = fea

        self.feature_cache[fea_dir] = {
            "feature_dict": feature_dict,
            "info_dicts": info_dicts
        }

        return self._load_from_cache(fea_dir, feature_names)


feature_loader = FeatureLoader()
