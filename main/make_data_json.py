# -*- coding: utf-8 -*-

import argparse

from pyretri.extract import make_data_json


def parse_args():
    parser = argparse.ArgumentParser(description='A tool box for deep learning-based image retrieval')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--dataset', '-d', default=None, type=str, help="path for the dataset that make the json file")
    parser.add_argument('--save_path', '-sp', default=None, type=str, help="save path for the json file")
    parser.add_argument('--type', '-t', default=None, type=str, help="mode of the dataset")
    parser.add_argument('--ground_truth', '-gt', default=None, type=str, help="ground truth of the dataset")

    args = parser.parse_args()

    return args


def main():

    # init args
    args = parse_args()
    assert args.dataset is not None, 'the data must be provided!'
    assert args.save_path is not None, 'the save path must be provided!'
    assert args.type is not None, 'the type must be provided!'

    # make data json
    make_data_json(args.dataset, args.save_path, args.type, args.ground_truth)

    print('make data json have done!')


if __name__ == '__main__':
    main()
