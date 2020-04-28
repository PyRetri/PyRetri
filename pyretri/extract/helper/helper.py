# -*- coding: utf-8 -*-

import os
import pickle
from tqdm import tqdm

import torch

from ..extractor import ExtractorBase
from ..aggregator import AggregatorBase
from ..splitter import SplitterBase
from ...utils import ensure_dir
from torch.utils.data import DataLoader
from typing import Dict, List

import time

class ExtractHelper:
    """
    A helper class to extract feature maps from model, and then aggregate them.
    """
    def __init__(self, assemble: int, extractor: ExtractorBase, splitter: SplitterBase, aggregators: List[AggregatorBase]):
        """
        Args:
            assemble (int): way to assemble features if transformers produce multiple images (e.g. TwoFlip, TenCrop).
            extractor (ExtractorBase): a extractor class for extracting features.
            splitter (SplitterBase): a splitter class for splitting features.
            aggregators (list): a list of extractor classes for aggregating features.
        """
        self.assemble = assemble
        self.extractor = extractor
        self.splitter = splitter
        self.aggregators = aggregators

    def _save_part_fea(self, datainfo: Dict, save_fea: List, save_path: str) -> None:
        """
        Save features in a json file.

        Args:
            datainfo (dict): the dataset information contained the data json file.
            save_fea (list): a list of features to be saved.
            save_path (str): the save path for the extracted features.
        """
        save_json = dict()
        for key in datainfo:
            if key != "info_dicts":
                save_json[key] = datainfo[key]
        save_json["info_dicts"] = save_fea

        with open(save_path, "wb") as f:
            pickle.dump(save_json, f)

    def extract_one_batch(self, batch: Dict) -> Dict:
        """
        Extract features for a batch of images.

        Args:
            batch (dict): a dict containing several image tensors.

        Returns:
            all_fea_dict (dict): a dict containing extracted features.
        """
        img = batch["img"]
        if torch.cuda.is_available():
            img = img.cuda()
        # img is in the shape (N, IMG_AUG, C, H, W)
        batch_size, aug_size = img.shape[0], img.shape[1]
        img = img.view(-1, img.shape[2], img.shape[3], img.shape[4])

        features = self.extractor(img)

        features = self.splitter(features)

        all_fea_dict = dict()
        for aggregator in self.aggregators:
            fea_dict = aggregator(features)
            all_fea_dict.update(fea_dict)

        # PyTorch will duplicate inputs if batch_size < n_gpu
        for key in all_fea_dict.keys():
            if self.assemble == 0:
                features = all_fea_dict[key][:img.shape[0], :]
                features = features.view(batch_size, aug_size, -1)
                features = features.view(batch_size, -1)
                all_fea_dict[key] = features
            elif self.assemble == 1:
                features = all_fea_dict[key].view(batch_size, aug_size, -1)
                features = features.sum(dim=1)
                all_fea_dict[key] = features

        return all_fea_dict

    def do_extract(self, dataloader: DataLoader, save_path: str, save_interval: int = 5000) -> None:
        """
        Extract features for a whole dataset and save features in json files.

        Args:
            dataloader (DataLoader): a DataLoader class for loading images for training.
            save_path (str): the save path for the extracted features.
            save_interval (int, optional): number of features saved in one part file.
        """
        datainfo = dataloader.dataset.data_info
        pbar = tqdm(range(len(dataloader)))
        save_fea = list()
        part_cnt = 0
        ensure_dir(save_path)

        start = time.time()
        for _, batch in zip(pbar, dataloader):
            feature_dict = self.extract_one_batch(batch)
            for i in range(len(batch["img"])):
                idx = batch["idx"][i]
                save_fea.append(datainfo["info_dicts"][idx])
                single_fea_dict = dict()
                for key in feature_dict:
                    single_fea_dict[key] = feature_dict[key][i].tolist()
                save_fea[-1]["feature"] = single_fea_dict
                save_fea[-1]["idx"] = int(idx)

            if len(save_fea) >= save_interval:
                self._save_part_fea(datainfo, save_fea, os.path.join(save_path, "part_{}.json".format(part_cnt)))
                part_cnt += 1
                del save_fea
                save_fea = list()
        end = time.time()
        print('time: ', end - start)

        if len(save_fea) >= 1:
            self._save_part_fea(datainfo, save_fea, os.path.join(save_path, "part_{}.json".format(part_cnt)))

    def do_single_extract(self, img: torch.Tensor) -> [Dict]:
        """
        Extract features for a single image.

        Args:
            img (torch.Tensor): a single image tensor.

        Returns:
            [fea_dict] (sequence): the extract features of the image.
        """
        batch = dict()
        batch["img"] = img.view(1, img.shape[0], img.shape[1], img.shape[2], img.shape[3])
        fea_dict = self.extract_one_batch(batch)

        return [fea_dict]
