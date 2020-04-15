# -*- coding: utf-8 -*-

from yacs.config import CfgNode

from ..datasets import get_datasets_cfg
from ..models import get_model_cfg
from ..extract import get_extract_cfg
from ..index import get_index_cfg
from ..evaluate import get_evaluate_cfg


def get_defaults_cfg() -> CfgNode:
    """
    Construct the default configuration tree.

    Returns:
        cfg (CfgNode): the default configuration tree.
    """
    cfg = CfgNode()

    cfg["datasets"] = get_datasets_cfg()
    cfg["model"] = get_model_cfg()
    cfg["extract"] = get_extract_cfg()
    cfg["index"] = get_index_cfg()
    cfg["evaluate"] = get_evaluate_cfg()

    return cfg


def setup_cfg(cfg: CfgNode, cfg_file: str, cfg_opts: list or None = None) -> CfgNode:
    """
    Load a yaml config file and merge it this CfgNode.

    Args:
        cfg (CfgNode): the configuration tree with default structure.
        cfg_file (str): the path for yaml config file which is matched with the CfgNode.
        cfg_opts (list, optional): config (keys, values) in a list (e.g., from command line) into this CfgNode.

    Returns:
        cfg (CfgNode): the configuration tree with settings in the config file.
    """
    cfg.merge_from_file(cfg_file)
    cfg.merge_from_list(cfg_opts)
    cfg.freeze()

    return cfg
