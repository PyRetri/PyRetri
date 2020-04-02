# -*- coding: utf-8 -*-

from yacs.config import CfgNode

# from .registry import BACKBONES
from .backbone.backbone_base import BACKBONES


def get_model_cfg() -> CfgNode:
    cfg = CfgNode()
    for name in BACKBONES:
        cfg[name] = CfgNode()
        cfg[name]["load_checkpoint"] = ""
    cfg["name"] = "unknown"
    return cfg
