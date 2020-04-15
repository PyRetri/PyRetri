# -*- coding: utf-8 -*-

import argparse
import os

import torch

from pyretri.config import get_defaults_cfg, setup_cfg
from pyretri.datasets import build_folder, build_loader
from pyretri.models import build_model
from pyretri.extract import build_extract_helper

from torchvision import models


def parse_args():
    parser = argparse.ArgumentParser(description='A tool box for deep learning-based image retrieval')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--data_json', '-dj', default=None, type=str, help='json file for dataset to be extracted')
    parser.add_argument('--save_path', '-sp', default=None, type=str, help='save path for features')
    parser.add_argument('--config_file', '-cfg', default=None, metavar='FILE', type=str, help='path to config file')
    parser.add_argument('--save_interval', '-si', default=5000, type=int, help='number of features saved in one part file')
    args = parser.parse_args()
    return args


def main():

    # init args
    args = parse_args()
    assert args.data_json is not None, 'the dataset json must be provided!'
    assert args.save_path is not None, 'the save path must be provided!'
    assert args.config_file is not None, 'a config file must be provided!'

    # init and load retrieval pipeline settings
    cfg = get_defaults_cfg()
    cfg = setup_cfg(cfg, args.config_file, args.opts)

    # build dataset and dataloader
    dataset = build_folder(args.data_json, cfg.datasets)
    dataloader = build_loader(dataset, cfg.datasets)

    # build model
    model = build_model(cfg.model)

    # build helper and extract features
    extract_helper = build_extract_helper(model, cfg.extract)
    extract_helper.do_extract(dataloader, args.save_path, args.save_interval)


if __name__ == '__main__':
    main()
