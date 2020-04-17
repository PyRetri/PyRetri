# -*- coding: utf-8 -*-

import os
from typing import Dict, List
import csv

def check_result_exist(now_res: Dict, exist_results: List) -> bool:
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


def get_default_result_dict(dir: str, data_name: str, index_name: str, fea_name: str) -> Dict:
    """
    Get the default result dict based on the experimental factors.

    Args:
        dir (str): the path of one single extracted feature directory.
        data_name (str): the name of the dataset.
        index_name (str): the name of query process.
        fea_name (str): the name of the features to be loaded.

    Returns:
        result_dict (Dict): a default configuration dict.
    """
    result_dict = {
        "data_name": data_name.split("_")[0],
        "pre_process_name": dir.split("_")[2],
        "model_name": "_".join(dir.split("_")[-2:]),
        "feature_map_name": fea_name.split("_")[0],
        "post_process_name": index_name
    }

    if len(fea_name.split("_")) == 1:
        result_dict["aggregator_name"] = "none"
    else:
        result_dict["aggregator_name"] = fea_name.split("_")[1]

    return result_dict


def save_to_csv(results: List[Dict], csv_path: str) -> None:
    """
    Save the search results in a csv format file.

    Args:
        results (List): a list of retrieval results.
        csv_path (str): the path for saving the csv file.
    """
    start = ["data", "pre_process", "model", "feature_map", "aggregator", "post_process"]
    for i in range(len(start)):
        results = sorted(results, key=lambda result: result[start[len(start) - i - 1] + "_name"])
    start.append('mAP')
    start.append('Recall@1')

    with open(csv_path, 'w') as f:
        csv_write = csv.writer(f)
        if len(start) > 0:
            csv_write.writerow(start)
        for i in range(len(results)):
            data_row = [0 for x in range(len(start))]
            data_row[0] = results[i]["data_name"]
            data_row[1] = results[i]["pre_process_name"]
            data_row[2] = results[i]["model_name"]
            data_row[3] = results[i]["feature_map_name"]
            data_row[4] = results[i]["aggregator_name"]
            data_row[5] = results[i]["post_process_name"]
            data_row[6] = results[i]["mAP"]
            data_row[7] = results[i]["recall_at_k"]['1']
            csv_write.writerow(data_row)


def filter_by_keywords(results: List[Dict], keywords: Dict) -> List[Dict]:
    """
    Filter the search results according to the given keywords

    Args:
        results (List): a list of retrieval results.
        keywords (Dict): a dict containing keywords to be selected.

    Returns:

    """
    for key in keywords:
        no_match = []
        if len(keywords[key]) == 0:
            continue
        else:
            for i in range(len(results)):
                if not results[i][key] in keywords[key]:
                    no_match.append(i)
        for num in no_match[::-1]:
            results.pop(num)
    return results
