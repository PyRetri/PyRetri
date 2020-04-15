# -*- coding: utf-8 -*-

import json
import importlib
import os
import argparse

from pyretri.config import get_defaults_cfg
from pyretri.query import build_query_helper
from pyretri.evaluate import build_evaluate_helper
from pyretri.query.utils import feature_loader

fea_names = ["output"]

task_mapping = {
    "market_gallery": {
        "gallery": "market_gallery",
        "query": "market_query",
        "train_fea_dir": "market_gallery"
    },
    "duke_gallery": {
        "gallery": "duke_gallery",
        "query": "duke_query",
        "train_fea_dir": "duke_gallery"
    },
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


def parse_args():
    parser = argparse.ArgumentParser(description='A tool box for deep learning-based image retrieval')
    parser.add_argument('--fea_dir', '-f', default=None, type=str, help="path of feature directory", required=True)
    parser.add_argument(
        "--search_modules",
        "-m",
        default="",
        help="name of search module's directory",
        type=str,
        required=True
    )
    parser.add_argument("--save_path", "-s", default=None, type=str, required=True)
    args = parser.parse_args()

    return args


def get_default_result_dict(dir, task_name, query_name, fea_name):

    result_dict = {
        "task_name": task_name.split("_")[0],
        "dataprocess": dir.split("_")[0],
        "model_name": "_".join(dir.split("_")[-2:]),
        "feature_map_name": fea_name.split("_")[0],
        "fea_process_name": query_name
    }
    if fea_name == "output":
        result_dict["aggregator_name"] = "none"
    else:
        result_dict["aggregator_name"] = fea_name.split("_")[1]

    return result_dict


def main():

    args = parse_args()

    cfg = get_defaults_cfg()

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
        for task_name in task_mapping:
            for query_name, query_args in queries.items():
                if task_name in dir:
                    for fea_name in fea_names:

                        gallery_fea_dir = os.path.join(args.fea_dir, dir)
                        query_fea_dir = gallery_fea_dir.replace(task_name, task_mapping[task_name]["query"])
                        train_fea_dir = gallery_fea_dir.replace(task_name, task_mapping[task_name]["train_fea_dir"])

                        assert os.path.exists(gallery_fea_dir), gallery_fea_dir
                        assert os.path.exists(query_fea_dir), query_fea_dir
                        assert os.path.exists(train_fea_dir), train_fea_dir

                        for post_proc in ["PartPCA", "PartSVD"]:
                            if post_proc in query_args.post_processors.names:
                                query_args.post_processors[post_proc].train_fea_dir = train_fea_dir
                        query_args.gallery_fea_dir, query_args.query_fea_dir = gallery_fea_dir, query_fea_dir
                        query_args.feature_names = [fea_name]
                        eval_args = evaluates["reid_overall"]

                        result_dict = get_default_result_dict(dir, task_name, query_name, fea_name)

                        if check_exist(result_dict, results):
                            print("[Search Query]: config exists...")
                            continue

                        cfg.query.merge_from_other_cfg(query_args)
                        cfg.evaluate.merge_from_other_cfg(eval_args)

                        query_helper = build_query_helper(cfg.query)
                        evaluate_helper = build_evaluate_helper(cfg.evaluate)

                        query_fea, query_info_dicts, _ = feature_loader.load(cfg.query.query_fea_dir,
                                                                             cfg.query.feature_names)
                        gallery_fea, gallery_info_dicts, _ = feature_loader.load(cfg.query.gallery_fea_dir,
                                                                                 cfg.query.feature_names)

                        query_result_info_dicts, _, _ = query_helper.do_query(query_fea, query_info_dicts, gallery_fea)
                        mAP, recall_at_k = evaluate_helper.do_eval(query_result_info_dicts, gallery_info_dicts)

                        to_save_dict = dict()
                        for k in recall_at_k:
                            to_save_dict[str(k)] = recall_at_k[k]

                        result_dict["mAP"] = float(mAP)
                        result_dict["recall_at_k"] = to_save_dict
                        print(result_dict)
                        assert False

                        results.append(result_dict)
                        with open(args.save_path, "w") as f:
                            json.dump(results, f)


if __name__ == '__main__':
    main()
