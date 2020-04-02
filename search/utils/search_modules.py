# -*- coding: utf-8 -*-

from yacs.config import CfgNode
from copy import deepcopy


def _convert_dict_to_cfg(d: dict) -> CfgNode:
    ret = CfgNode()
    for key in d:
        if isinstance(d[key], dict):
            ret[key] = _convert_dict_to_cfg(d[key])
        else:
            ret[key] = d[key]
    return ret


class SearchModules(dict):
    r"""
    This class defines the search args for one module, e.g., data process, feature extraction, feature
    aggregation, feature process and query.
    """
    def __init__(self):
        super(SearchModules, self).__init__()

    def add(self, name: str, value: dict):
        self[name] = _convert_dict_to_cfg(value)

    def check_valid(self, cfg: CfgNode):
        cfg = deepcopy(cfg)
        for module in self.values():
            cfg.merge_from_other_cfg(module)
