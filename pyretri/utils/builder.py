# -*- coding: utf-8 -*-

from yacs.config import CfgNode

from .module_base import ModuleBase
from .registry import Registry


def get_config_from_registry(registry: Registry) -> CfgNode:
    """
    Collect all hyper-parameters from modules in registry.

    Args:
        registry (Registry): module registry.

    Returns:
        cfg (CfgNode): configurations for this registry.
    """
    cfg = CfgNode()
    for name in registry:
        cfg[name] = CfgNode()
        loss = registry[name]
        hps = loss.default_hyper_params
        for hp_name in hps:
            cfg[name][hp_name] = hps[hp_name]
    return cfg


def simple_build(name: str, cfg: CfgNode, registry: Registry, **kwargs):
    """
    Simply build a module according to name and hyper-parameters.

    Args:
        name (str): name for instance to be built.
        cfg (CfgNode): configurations for this sub-module.
        registry (Registry): registry for this sub-module.
        **kwargs: keyword arguments.

    Returns:
        module: a initialized instance
    """
    assert name in registry
    module = registry[name]
    hps = module.default_hyper_params

    for hp_name in hps:
        new_value = cfg[name][hp_name]
        hps[hp_name] = new_value

    return module(hps=hps, **kwargs)
