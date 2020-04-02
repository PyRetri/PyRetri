# -*- coding: utf-8 -*-

from typing import Dict, List


def check_exist(now_res: Dict, exist_results: List) -> bool:
    """
    Check if the config exists.

    Args:
        now_res (Dict): configuration to be checked.
        exist_results (List): a list of existing configurations.

    Returns:
        bool: if the config exists.
    """
    for e_r in exist_results:
        totoal_equal = True
        for key in now_res:
            if now_res[key] != e_r[key]:
                totoal_equal = False
                break
        if totoal_equal:
            return True
    return False


def get_dir(root_path: str, dir: str, dataset: Dict) -> (str, str, str):
    """
    Get the feature directory path of gallery set, query set and feature set for training PCA/SVD.

    Args:
        root_path (str): the root path of all extracted features.
        dir (str): the path of one single extracted feature directory.
        dataset (Dict): a dict containing the information of gallery set, query set and training set.

    Returns:
        tuple(str, str, str): path of gallery set, query set and feature set for training PCA/SVD.
    """
    template_dir = os.path.join(root_path, dir)
    target = dir.split('_')[0] + '_' + dir.split('_')[1]
    gallery_fea_dir = template_dir.replace(target, dataset["gallery"])
    query_fea_dir = template_dir.replace(target, dataset["query"])
    train_fea_dir = template_dir.replace(target, dataset["train"])
    return gallery_fea_dir, query_fea_dir, train_fea_dir


def get_default_result_dict(dir, data_name, query_name, fea_name) -> Dict:
    """
    Get the default result dict based on the experimental factors.

    Args:
        dir (str): the path of one single extracted feature directory.
        data_name (str): the name of the dataset.
        query_name (str): the name of query process.
        fea_name (str): the name of the features to be loaded.

    Returns:
        result_dict (Dict): a default configuration dict.
    """
    result_dict = {
        "data_name": data_name.split("_")[0],
        "dataprocess": dir.split("_")[0],
        "model_name": "_".join(dir.split("_")[-2:]),
        "feature_map_name": fea_name.split("_")[0],
        "fea_process_name": query_name
    }

    if fea_name == "fc":
        result_dict["aggregator_name"] = "none"
    else:
        result_dict["aggregator_name"] = fea_name.split("_")[1]

    return result_dict

