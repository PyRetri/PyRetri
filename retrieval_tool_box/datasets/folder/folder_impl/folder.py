# -*- coding: utf-8 -*-

import pickle

from ..folder_base import FolderBase
from ...registry import FOLDERS

from typing import Dict, List


@FOLDERS.register
class Folder(FolderBase):
    """
    A folder function for loading images.

    Hyper-Params:
        use_bbox: bool, whether use bbox to crop image. When set to true,
            make sure that bbox attribute is provided in your data json and bbox format is [x1, y1, x2, y2].
    """
    default_hyper_params = {
        "use_bbox": False,
    }

    def __init__(self, data_json_path: str, transformer: callable or None = None, hps: Dict or None = None):
        """
        Args:
            data_json_path (str): the path for data json file.
            transformer (callable): a list of data augmentation operations.
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        super(Folder, self).__init__(data_json_path, transformer, hps)
        self.classes, self.class_to_idx = self.find_classes(self.data_info["info_dicts"])

    def find_classes(self, info_dicts: Dict) -> (List, Dict):
        """
        Get the class names and the mapping relations.

        Args:
            info_dicts (dict): the dataset information contained the data json file.

        Returns:
            tuple (list, dict): a list of class names and a dict for projecting class name into int label.
        """
        classes = list()
        for i in range(len(info_dicts)):
            if info_dicts[i]["label"] not in classes:
                classes.append(info_dicts[i]["label"])
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __len__(self) -> int:
        """
        Get the number of total training samples.

        Returns:
            length (int): the number of total training samples.
        """
        return len(self.data_info["info_dicts"])

    def __getitem__(self, idx: int) -> Dict:
        """
        Load the image and convert it to tensor for training.

        Args:
            idx (int): the serial number of the image.

        Returns:
            item (dict): the dict containing the image after augmentations, serial number and label.
        """
        info = self.data_info["info_dicts"][idx]
        img = self.read_img(info["path"])
        if self._hyper_params["use_bbox"]:
            assert info["bbox"] is not None, 'image {} does not have a bbox'.format(info["path"])
            x1, y1, x2, y2 = info["bbox"]
            box = map(int, (x1, y1, x2, y2))
            img = img.crop(box)
        img = self.transformer(img)
        return {"img": img, "idx": idx, "label": self.class_to_idx[info["label"]]}
