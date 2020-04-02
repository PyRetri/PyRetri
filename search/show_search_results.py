# -*- coding: utf-8 -*-

import os
import argparse
import json

import csv
import codecs


def parse_args():
    parser = argparse.ArgumentParser(description='A tool box for deep learning-based image retrieval')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--results_json_path', '-r', default=None, type=str, help="the path of the result json")
    args = parser.parse_args()

    return args


def show_results(results):
    for i in range(len(results)):
        print(results[i])


def save_to_csv(results, csv_path):
    start = []
    col_num = 12
    if not os.path.exists(csv_path):
        start = ["data_process", "model", "feature", "fea_process", "market_mAP", "market_mAP_re", "market_R1",
                 "market_R1_re", "duke_mAP", "duke_mAP_re", "duke_R1", "duke_R1_re"]
    with open(csv_path, 'a+') as f:
        csv_write = csv.writer(f)
        if len(start) > 0:
            csv_write.writerow(start)
        for i in range(len(results)):
            data_row = [0 for x in range(col_num)]
            data_row[0] = results[i]["dataprocess"]
            data_row[1] = results[i]["model_name"]
            data_row[2] = results[i]["feature_map_name"]
            data_row[3] = results[i]["fea_process_name"]
            if results[i]["task_name"] == 'market':
                data_row[4] = results[i]["mAP"]
                data_row[6] = results[i]["recall_at_k"]['1']
            elif results[i]["task_name"] == 'duke':
                data_row[8] = results[i]["mAP"]
                data_row[10] = results[i]["recall_at_k"]['1']
            csv_write.writerow(data_row)


def main():

    args = parse_args()
    assert os.path.exists(args.results_json_path), 'the config file must be existed!'

    with open(args.results_json_path, "r") as f:
        results = json.load(f)

    key_words = {
        'task_name': ['market'],
        'dataprocess': list(),
        'model_name': list(),
        'feature_map_name': list(),
        'aggregator_name': list(),
        'fea_process_name': ['no_fea_process', 'l2_normalize', 'pca_whiten', 'pca_wo_whiten'],
    }

    csv_path = '/home/songrenjie/projects/RetrievalToolBox/test.csv'
    save_to_csv(results, csv_path)

    for key in key_words:
        no_match = []
        if len(key_words[key]) == 0:
            continue
        else:
            for i in range(len(results)):
                if not results[i][key] in key_words[key]:
                    no_match.append(i)
        for num in no_match[::-1]:
            results.pop(num)

    show_results(results)


if __name__ == '__main__':
    main()
