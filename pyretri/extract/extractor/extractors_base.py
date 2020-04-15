# -*- coding: utf-8 -*-

from functools import partial

import torch
import torch.nn as nn
import numpy as np

from ...utils import ModuleBase

from typing import Dict


class ExtractorBase(ModuleBase):
    """
    The base class feature map extractors.

    Hyper-Parameters
        extract_features (list): indicates which feature maps to output. See available_feas for available feature maps.
            If it is ["all"], then all available features will be output.
    """
    available_feas = list()
    default_hyper_params = {
        "extract_features": list(),
    }

    def __init__(self, model: nn.Module, feature_modules: Dict[str, nn.Module], hps: Dict or None = None):
        """
        Args:
            model (nn.Module): the model for extracting features.
            feature_modules (dict): the output layer of the model.
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        super(ExtractorBase, self).__init__(hps)
        assert len(self._hyper_params["extract_features"]) > 0

        self.model = model.eval()
        if torch.cuda.is_available():
            self.model.cuda()
            if torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model)
        self.feature_modules = feature_modules
        self.feature_buffer = dict()

        if self._hyper_params["extract_features"][0] == "all":
            self._hyper_params["extract_features"] = self.available_feas
        for fea in self._hyper_params["extract_features"]:
            self.feature_buffer[fea] = dict()

        self._register_hook()

    def _register_hook(self) -> None:
        """
        Register hooks to output inner feature map.
        """
        def hook(feature_buffer, fea_name, module, input, output):
            feature_buffer[fea_name][str(output.device)] = output.data

        for fea in self._hyper_params["extract_features"]:
            assert fea in self.feature_modules, 'unknown feature {}!'.format(fea)
            self.feature_modules[fea].register_forward_hook(partial(hook, self.feature_buffer, fea))

    def __call__(self, x: torch.tensor) -> Dict:
        with torch.no_grad():
            self.model(x)
            ret = dict()
            for fea in self._hyper_params["extract_features"]:
                ret[fea] = list()
                devices = list(self.feature_buffer[fea].keys())
                devices = np.sort(devices)
                for d in devices:
                    ret[fea].append(self.feature_buffer[fea][d])
                ret[fea] = torch.cat(ret[fea], dim=0)
        return ret
