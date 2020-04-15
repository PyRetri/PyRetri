# -*- coding: utf-8 -*-

from yacs.config import CfgNode

from .registry import COLLATEFNS, FOLDERS, TRANSFORMERS

from ..utils import get_config_from_registry

def get_collate_cfg() -> CfgNode:
    cfg = get_config_from_registry(COLLATEFNS)
    cfg["name"] = "unknown"
    return cfg


def get_folder_cfg() -> CfgNode:
    cfg = get_config_from_registry(FOLDERS)
    cfg["name"] = "unknown"
    return cfg


def get_tranformers_cfg() -> CfgNode:
    cfg = get_config_from_registry(TRANSFORMERS)
    cfg["names"] = ["unknown"]
    return cfg


def get_datasets_cfg() -> CfgNode:
    cfg = CfgNode()
    cfg["collate_fn"] = get_collate_cfg()
    cfg["folder"] = get_folder_cfg()
    cfg["transformers"] = get_tranformers_cfg()
    cfg["batch_size"] = 1
    return cfg
