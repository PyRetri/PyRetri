# -*- coding: utf-8 -*-
import os
from shutil import copyfile


def split_dataset(dataset_path: str, split_file: str) -> None:
    """
    Split the dataset according to the given splitting rules.

    Args:
        dataset_path (str): the path of the dataset.
        split_file (str): the path of the file containing the splitting rules.
    """

    with open(split_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            path = line.strip('\n').split(' ')[0]
            is_gallery = line.strip('\n').split(' ')[1]
            if is_gallery == '0':
                src = os.path.join(dataset_path, path)
                dst = src.replace(path.split('/')[0], 'query')
                dst_index = len(dst.split('/')[-1])
                dst_dir = dst[:len(dst) - dst_index]
                if not os.path.isdir(dst_dir):
                    os.makedirs(dst_dir)
                if not os.path.exists(dst):
                    os.symlink(src, dst)
            elif is_gallery == '1':
                src = os.path.join(dataset_path, path)
                dst = src.replace(path.split('/')[0], 'gallery')
                dst_index = len(dst.split('/')[-1])
                dst_dir = dst[:len(dst) - dst_index]
                if not os.path.isdir(dst_dir):
                    os.makedirs(dst_dir)
                if not os.path.exists(dst):
                    os.symlink(src, dst)
