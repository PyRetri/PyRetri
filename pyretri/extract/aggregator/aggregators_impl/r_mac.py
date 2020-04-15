# -*- coding: utf-8 -*-

import torch

from ..aggregators_base import AggregatorBase
from ...registry import AGGREGATORS

from typing import Dict, List

@AGGREGATORS.register
class RMAC(AggregatorBase):
    """
    Regional Maximum activation of convolutions (R-MAC).
    c.f. https://arxiv.org/pdf/1511.05879.pdf

    Hyper-Params
        level_n (int): number of levels for selecting regions.
    """

    default_hyper_params = {
        "level_n": 3,
    }

    def __init__(self, hps: Dict or None = None):
        """
        Args:
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        super(RMAC, self).__init__(hps)
        self.first_show = True
        self.cached_regions = dict()

    def _get_regions(self, h: int, w: int) -> List:
        """
        Divide the image into several regions.

        Args:
            h (int): height for dividing regions.
            w (int): width for dividing regions.

        Returns:
            regions (List): a list of region positions.
        """
        if (h, w) in self.cached_regions:
            return self.cached_regions[(h, w)]

        m = 1
        n_h, n_w = 1, 1
        regions = list()
        if h != w:
            min_edge = min(h, w)
            left_space = max(h, w) - min(h, w)
            iou_target = 0.4
            iou_best = 1.0
            while True:
                iou_tmp = (min_edge ** 2 - min_edge * (left_space // m)) / (min_edge ** 2)

                # small m maybe result in non-overlap
                if iou_tmp <= 0:
                    m += 1
                    continue

                if abs(iou_tmp - iou_target) <= iou_best:
                    iou_best = abs(iou_tmp - iou_target)
                    m += 1
                else:
                    break
            if h < w:
                n_w = m
            else:
                n_h = m

        for i in range(self._hyper_params["level_n"]):
            region_width = int(2 * 1.0 / (i + 2) * min(h, w))
            step_size_h = (h - region_width) // n_h
            step_size_w = (w - region_width) // n_w

            for x in range(n_h):
                for y in range(n_w):
                    st_x = step_size_h * x
                    ed_x = st_x + region_width - 1
                    assert ed_x < h
                    st_y = step_size_w * y
                    ed_y = st_y + region_width - 1
                    assert ed_y < w
                    regions.append((st_x, st_y, ed_x, ed_y))

            n_h += 1
            n_w += 1

        self.cached_regions[(h, w)] = regions
        return regions

    def __call__(self, features: Dict[str, torch.tensor]) -> Dict[str, torch.tensor]:
        ret = dict()
        for key in features:
            fea = features[key]
            if fea.ndimension() == 4:
                h, w = fea.shape[2:]
                final_fea = None
                regions = self._get_regions(h, w)
                for _, r in enumerate(regions):
                    st_x, st_y, ed_x, ed_y = r
                    region_fea = (fea[:, :, st_x: ed_x, st_y: ed_y].max(dim=3)[0]).max(dim=2)[0]
                    region_fea = region_fea / torch.norm(region_fea, dim=1, keepdim=True)
                    if final_fea is None:
                        final_fea = region_fea
                    else:
                        final_fea = final_fea + region_fea
                ret[key + "_{}".format(self.__class__.__name__)] = final_fea
            else:
                # In case of fc feature.
                assert fea.ndimension() == 2
                if self.first_show:
                    print("[RMAC Aggregator]: find 2-dimension feature map, skip aggregation")
                    self.first_show = False
                ret[key] = fea
        return ret
