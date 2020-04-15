# -*- coding: utf-8 -*-

import os

import torch.nn as nn
from torch.nn import Parameter

from torchvision.models.utils import load_state_dict_from_url

from typing import Dict

def ensure_dir(path: str) -> None:
    """
    Check if a directory exists, if not, create a new one.

    Args:
        path (str): the path of the directory.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def load_state_dict(model: nn.Module, state_dict: Dict) -> None:
    """
    Load parameters regardless the shape of parameters with the same name need to match,
    which is a slight modification to load_state_dict of pytorch.

    Args:
        model (nn.Module): the model for extracting features.
        state_dict (Dict): a dict of model parameters.
    """
    own_state = model.state_dict()
    success_keys = list()
    for name, param in state_dict.items():
        if name in own_state:
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
                success_keys.append(name)
            except Exception:
                print("[LoadStateDict]: shape mismatch in parameter {}, {} vs {}".format(
                    name, own_state[name].size(), param.size()
                ))
        else:
            print("[LoadStateDict]: " + 'unexpected key "{}" in state_dict'.format(name))
    missing = set(own_state.keys()) - set(success_keys)
    if len(missing) > 0:
        print("[LoadStateDict]: " + "missing keys or mismatch param in state_dict: {}".format(missing))
