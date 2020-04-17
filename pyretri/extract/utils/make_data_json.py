# -*- coding: utf-8 -*-

import pickle
import os


def make_ds_for_general(dataset_path: str, save_path: str) -> None:
    """
    Generate data json file for dataset collecting images with the same label one directory. e.g. CUB-200-2011.

    Args:
        dataset_path (str): the path of the dataset.
        save_ds_path (str): the path for saving the data json files.
    """
    info_dicts = list()
    img_dirs = os.listdir(dataset_path)
    label_list = list()
    label_to_idx = dict()
    for dir in img_dirs:
        for root, _, files in os.walk(os.path.join(dataset_path, dir)):
            for file in files:
                info_dict = dict()
                info_dict['path'] = os.path.join(root, file)
                if dir not in label_list:
                    label_to_idx[dir] = len(label_list)
                    label_list.append(dir)
                info_dict['label'] = dir
                info_dict['label_idx'] = label_to_idx[dir]
                info_dicts += [info_dict]
    with open(save_path, 'wb') as f:
        pickle.dump({'nr_class': len(img_dirs), 'path_type': 'absolute_path', 'info_dicts': info_dicts}, f)


def make_ds_for_oxford(dataset_path, save_path: str or None=None, gt_path: str or None=None) -> None:
    """
    Generate data json file for oxford dataset.

    Args:
        dataset_path (str): the path of the dataset.
        save_ds_path (str): the path for saving the data json files.
        gt_path (str, optional): the path of the ground truth, necessary for Oxford.
    """
    label_list = list()
    info_dicts = list()
    query_info = dict()
    if 'query' in dataset_path:
        for root, _, files in os.walk(gt_path):
            for file in files:
                if 'query' in file:
                    with open(os.path.join(root, file), 'r') as f:
                        line = f.readlines()[0].strip('\n').split(' ')
                        query_name = file[:-10]
                        label = line[0][5:]
                        bbox = [float(line[1]), float(line[2]), float(line[3]), float(line[4])]
                        query_info[label] = {'query_name': query_name, 'bbox': bbox,}

    for root, _, files in os.walk(dataset_path):
        for file in files:
            info_dict = dict()
            info_dict['path'] = os.path.join(root, file)
            label = file.split('.')[0]
            if label not in label_list:
                label_list.append(label)
            info_dict['label'] = label
            if 'query' in dataset_path:
                info_dict['bbox'] = query_info[label]['bbox']
                info_dict['query_name'] = query_info[label]['query_name']
            info_dicts += [info_dict]

    with open(save_path, 'wb') as f:
        pickle.dump({'nr_class': len(label_list), 'path_type': 'absolute_path', 'info_dicts': info_dicts}, f)


def make_ds_for_reid(dataset_path: str, save_path: str) -> None:
    """
    Generating data json file for Re-ID dataset.

    Args:
        dataset_path (str): the path of the dataset.
        save_ds_path (str): the path for saving the data json files.
    """
    label_list = list()
    info_dicts = list()
    for root, _, files in os.walk(dataset_path):
        for file in files:
            info_dict = dict()
            info_dict['path'] = os.path.join(root, file)
            label = file.split('_')[0]
            cam = file.split('_')[1][1]
            if label not in label_list:
                label_list.append(label)
            info_dict['label'] = label
            info_dict['cam'] = cam
            info_dicts += [info_dict]
    with open(save_path, 'wb') as f:
        pickle.dump({'nr_class': len(label_list), 'path_type': 'absolute_path', 'info_dicts': info_dicts}, f)


def make_data_json(dataset_path: str, save_path: str, type: str, gt_path: str or None=None) -> None:
    """
    Generate data json file for dataset.

    Args:
        dataset_path (str): the path of the dataset.
        save_ds_path (str): the path for saving the data json files.
        type (str): the structure type of the dataset.
        gt_path (str, optional): the path of the ground truth, necessary for Oxford.
    """
    assert type in ['general', 'oxford', 'reid']
    if type == 'general':
        make_ds_for_general(dataset_path, save_path)
    elif type == 'oxford':
        make_ds_for_oxford(dataset_path, save_path, gt_path)
    elif type == 'reid':
        make_ds_for_reid(dataset_path, save_path)
