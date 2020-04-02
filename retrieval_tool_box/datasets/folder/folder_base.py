# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image
import pickle
import os
from abc import abstractmethod

from ...utils import ModuleBase

from typing import Dict, List


class FolderBase(ModuleBase):
    """
    The base class of folder function.
    """
    default_hyper_params = dict()

    def __init__(self, data_json_path: str, transformer: callable or None = None, hps: Dict or None = None):
        """
        Args:
            data_json_path (str): the path for data json file.
            transformer (callable): a list of data augmentation operations.
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        super(FolderBase, self).__init__(hps)
        with open(data_json_path, "rb") as f:
            self.data_info = pickle.load(f)
        self.data_json_path = data_json_path
        self.transformer = transformer

    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict:
        pass

    def find_classes(self, info_dicts: Dict) -> (List, Dict):
        pass

    def read_img(self, path: str) -> Image:
        """
        Load image.

        Args:
            path (str): the path of the image.

        Returns:
            image (Image): shape (H, W, C).
        """
        try:
            img = Image.open(path)
            img = img.convert("RGB")
            return img
        except Exception as e:
            print('[DataSet]: WARNING image can not be loaded: {}'.format(str(e)))
            return None
