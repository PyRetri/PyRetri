# -*- coding: utf-8 -*-

import argparse
import os
from PIL import Image
import numpy as np

from pyretri.config import get_defaults_cfg, setup_cfg
from pyretri.datasets import build_transformers
from pyretri.models import build_model
from pyretri.extract import build_extract_helper
from pyretri.index import build_index_helper, feature_loader


def parse_args():
    parser = argparse.ArgumentParser(description='A tool box for deep learning-based image retrieval')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--config_file', '-cfg', default=None, metavar='FILE', type=str, help='path to config file')
    args = parser.parse_args()
    return args


def main():

    # init args
    args = parse_args()
    assert args.config_file is not "", 'a config file must be provided!'
    assert os.path.exists(args.config_file), 'the config file must be existed!'

    # init and load retrieval pipeline settings
    cfg = get_defaults_cfg()
    cfg = setup_cfg(cfg, args.config_file, args.opts)

    # set path for single image
    path = '/data/caltech101/query/airplanes/image_0004.jpg'

    # build transformers
    transformers = build_transformers(cfg.datasets.transformers)

    # build model
    model = build_model(cfg.model)

    # read image and convert it to tensor
    img = Image.open(path).convert("RGB")
    img_tensor = transformers(img)

    # build helper and extract feature for single image
    extract_helper = build_extract_helper(model, cfg.extract)
    img_fea_info = extract_helper.do_single_extract(img_tensor)
    stacked_feature = list()
    for name in cfg.index.feature_names:
        assert name in img_fea_info[0], "invalid feature name: {} not in {}!".format(name, img_fea_info[0].keys())
        stacked_feature.append(img_fea_info[0][name].cpu())
    img_fea = np.concatenate(stacked_feature, axis=1)

    # load gallery features
    gallery_fea, gallery_info, _ = feature_loader.load(cfg.index.gallery_fea_dir, cfg.index.feature_names)

    # build helper and single index feature
    index_helper = build_index_helper(cfg.index)
    index_result_info, query_fea, gallery_fea = index_helper.do_index(img_fea, img_fea_info, gallery_fea)

    index_helper.save_topk_retrieved_images('retrieved_images/', index_result_info[0], 5, gallery_info)

    print('single index have done!')


if __name__ == '__main__':
    main()
