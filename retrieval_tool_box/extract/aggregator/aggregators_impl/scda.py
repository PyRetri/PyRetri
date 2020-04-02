# -*- coding: utf-8 -*-

import queue

import torch

from ..aggregators_base import AggregatorBase
from ...registry import AGGREGATORS

from typing import Dict


@AGGREGATORS.register
class SCDA(AggregatorBase):
    """
    Selective Convolutional Descriptor Aggregation for Fine-Grained Image Retrieval.
    c.f. http://www.weixiushen.com/publication/tip17SCDA.pdf
    """
    default_hyper_params = dict()

    def __init__(self, hps: Dict or None = None):
        """
        Args:
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        super(SCDA, self).__init__(hps)
        self.first_show = True

    def bfs(self, x: int, y: int, mask: torch.tensor, cc_map: torch.tensor, cc_id: int) -> int:
        dirs = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        q = queue.LifoQueue()
        q.put((x, y))

        ret = 1
        cc_map[x][y] = cc_id

        while not q.empty():
            x, y = q.get()

            for (dx, dy) in dirs:
                new_x = x + dx
                new_y = y + dy
                if 0 <= new_x < mask.shape[0] and 0 <= new_y < mask.shape[1]:
                    if mask[new_x][new_y] == 1 and cc_map[new_x][new_y] == 0:
                        q.put((new_x, new_y))
                        ret += 1
                        cc_map[new_x][new_y] = cc_id
        return ret

    def find_max_cc(self, mask: torch.tensor) -> torch.tensor:
        """
        Find the largest connected component of the mask。

        Args:
            mask (torch.tensor): the original mask.

        Returns:
            mask (torch.tensor): the mask only containing the maximum connected component.
        """
        assert mask.ndim == 4
        assert mask.shape[1] == 1
        mask = mask[:, 0, :, :]
        for i in range(mask.shape[0]):
            m = mask[i]
            cc_map = torch.zeros(m.shape)
            cc_num = list()

            for x in range(m.shape[0]):
                for y in range(m.shape[1]):
                    if m[x][y] == 1 and cc_map[x][y] == 0:
                        cc_id = len(cc_num) + 1
                        cc_num.append(self.bfs(x, y, m, cc_map, cc_id))

            max_cc_id = cc_num.index(max(cc_num)) + 1
            m[cc_map != max_cc_id] = 0
        mask = mask[:, None, :, :]
        return mask

    def __call__(self, features: Dict[str, torch.tensor]) -> Dict[str, torch.tensor]:
        ret = dict()
        for key in features:
            fea = features[key]
            if fea.ndimension() == 4:
                mask = fea.sum(dim=1, keepdims=True)
                thres = mask.mean(dim=(2, 3), keepdims=True)
                mask[mask <= thres] = 0
                mask[mask > thres] = 1
                mask = self.find_max_cc(mask)
                fea = fea * mask

                gap = fea.mean(dim=(2, 3))
                gmp, _ = fea.max(dim=3)
                gmp, _ = gmp.max(dim=2)

                ret[key + "_{}".format(self.__class__.__name__)] = torch.cat([gap, gmp], dim=1)
            else:
                # In case of fc feature.
                assert fea.ndimension() == 2
                if self.first_show:
                    print("[SCDA Aggregator]: find 2-dimension feature map, skip aggregation")
                    self.first_show = False
                ret[key] = fea
        return ret
