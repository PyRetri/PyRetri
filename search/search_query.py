# -*- coding: utf-8 -*-

import json
import importlib
import os
import argparse

from pyretri.config import get_defaults_cfg
from pyretri.query import build_query_helper
from pyretri.evaluate import build_evaluate_helper
from pyretri.index import feature_loader


vgg_fea = ["pool5_PWA"]
res_fea = ["pool5_PWA"]

task_mapping = {
    "oxford_gallery": {
        "gallery": "oxford_gallery",
        "query": "oxford_query",
        "train_fea_dir": "paris"
    },
    "cub_gallery": {
        "gallery": "cub_gallery",
        "query": "cub_query",
        "train_fea_dir": "cub_gallery"
    },
    "indoor_gallery": {
        "gallery": "indoor_gallery",
        "query": "indoor_query",
        "train_fea_dir": "indoor_gallery"
    },
    "caltech101_gallery": {
        "gallery": "caltech101_gallery",
        "query": "caltech101_query",
        "train_fea_dir": "caltech101_gallery"
    }
}


def check_exist(now_res, exist_results):
    for e_r in exist_results:
        totoal_equal = True
        for key in now_res:
            if now_res[key] != e_r[key]:
                totoal_equal = False
                break
        if totoal_equal:
            return True
    return False


def get_default_result_dict(dir, task_name, query_name, fea_name):
    result_dict = {
        "task_name": task_name.split("_")[0],
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


def parse_args():
    parser = argparse.ArgumentParser(description='A tool box for deep learning-based image retrieval')
    parser.add_argument('--fea_dir', '-fd', default=None, type=str, help="path of feature dirs", required=True)
    parser.add_argument("--search_modules", "-sm", default=None, type=str, help="name of search module's directory")
    parser.add_argument("--save_path", "-sp", default=None, type=str, help="path for saving results")
    args = parser.parse_args()

    return args


def main():

    # init args
    args = parse_args()
    assert args.fea_dir is not None, 'the feature directory must be provided!'
    assert args.search_modules is not None, 'the search modules must be provided!'
    assert args.save_path is not None, 'the save path must be provided!'

    # init retrieval pipeline settings
    cfg = get_defaults_cfg()

    # load search space
    queries = importlib.import_module("{}.query_dict".format(args.search_modules)).queries
    evaluates = importlib.import_module("{}.query_dict".format(args.search_modules)).evaluates

    if os.path.exists(args.save_path):
        with open(args.save_path, "r") as f:
            results = json.load(f)
    else:
        results = list()

    q_cnt = 0
    for dir in os.listdir(args.fea_dir):
        q_cnt += 1
        print("Processing {} / {} queries...".format(q_cnt, len(queries)))
        for query_name, query_args in queries.items():
            for task_name in task_mapping:
                if task_name in dir:

                    if "vgg" in gallery_fea_dir:
                        fea_names = vgg_fea
                    else:
                        fea_names = res_fea

                    for fea_name in fea_names:
                        gallery_fea_dir = os.path.join(args.fea_dir, dir)
                        query_fea_dir = gallery_fea_dir.replace(task_name, task_mapping[task_name]["query"])
                        train_fea_dir = gallery_fea_dir.replace(task_name, task_mapping[task_name]["train_fea_dir"])

                        for post_proc in ["PartPCA", "PartSVD"]:
                            if post_proc in query_args.post_processors.names:
                                query_args.post_processors[post_proc].train_fea_dir = train_fea_dir

                        query.gallery_fea_dir, query.query_fea_dir = gallery_fea_dir, query_fea_dir

                        query.feature_names = [fea_name]
                        if task_name == "oxford_base":
                            evaluate = evaluates["oxford_overall"]
                        else:
                            evaluate = evaluates["overall"]

                        result_dict = get_default_result_dict(dir, task_name, query_name, fea_name)

                        if check_exist(result_dict, results):
                            print("[Search Query]: config exists...")
                            continue

                        # load retrieval pipeline settings
                        cfg.query.merge_from_other_cfg(query)
                        cfg.evaluate.merge_from_other_cfg(evaluate)

                        # load features
                        query_fea, query_info, _ = feature_loader.load(cfg.query.query_fea_dir, cfg.query.feature_names)
                        gallery_fea, gallery_info, _ = feature_loader.load(cfg.query.gallery_fea_dir,
                                                                                 cfg.query.feature_names)

                        # build helper and index features
                        query_helper = build_query_helper(cfg.query)
                        query_result_info, _, _ = query_helper.do_query(query_fea, query_info, gallery_fea)

                        # build helper and evaluate results
                        evaluate_helper = build_evaluate_helper(cfg.evaluate)
                        mAP, recall_at_k = evaluate_helper.do_eval(query_result_info, gallery_info)

                        # save results
                        to_save_dict = dict()
                        for k in recall_at_k:
                            to_save_dict[str(k)] = recall_at_k[k]
                        result_dict["mAP"] = float(mAP)
                        result_dict["recall_at_k"] = to_save_dict

                        results.append(result_dict)
                        with open(args.save_path, "w") as f:
                            json.dump(results, f)


if __name__ == '__main__':
    main()
