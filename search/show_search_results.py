# -*- coding: utf-8 -*-

import os
import argparse
import json

import codecs

from utils.misc import save_to_csv, filter_by_keywords


def parse_args():
    parser = argparse.ArgumentParser(description='A tool box for deep learning-based image retrieval')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--results_json_path', '-r', default=None, type=str, help="path of the result json")
    args = parser.parse_args()

    return args


def show_results(results):
    for i in range(len(results)):
        print(results[i])


def main():
    # init args
    args = parse_args()
    assert os.path.exists(args.results_json_path), 'the config file must be existed!'

    with open(args.results_json_path, "r") as f:
        results = json.load(f)

    # save the search results in a csv format file.
    csv_path = '/home/songrenjie/projects/RetrievalToolBox/test.csv'
    save_to_csv(results, csv_path)

    # define the keywords to be selected
    keywords = {
        'data_name': ['market'],
        'pre_process_name': list(),
        'model_name': list(),
        'feature_map_name': list(),
        'aggregator_name': list(),
        'post_process_name': ['no_fea_process', 'l2_normalize', 'pca_whiten', 'pca_wo_whiten'],
    }

    # show search results according to the given keywords
    results = filter_by_keywords(results, keywords)
    show_results(results)


if __name__ == '__main__':
    main()
