# -*- coding: utf-8 -*-

import torch
import numpy as np
from PIL import Image
from ..transformers_base import TransformerBase
from ...registry import TRANSFORMERS
from torchvision.transforms import Resize as TResize
from torchvision.transforms import TenCrop as TTenCrop
from torchvision.transforms import CenterCrop as TCenterCrop
from torchvision.transforms import ToTensor as TToTensor
from torchvision.transforms.functional import hflip

from typing import Dict

@TRANSFORMERS.register
class DirectResize(TransformerBase):
    """
    Directly resize image to target size, regardless of h: w ratio.

    Hyper-Params
        size (sequence): desired output size.
        interpolation (int): desired interpolation.
    """
    default_hyper_params = {
        "size": (224, 224),
        "interpolation": Image.BILINEAR,
    }

    def __init__(self, hps: Dict or None = None):
        """
        Args:
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        super(DirectResize, self).__init__(hps)
        self.t_transformer = TResize(self._hyper_params["size"], self._hyper_params["interpolation"])

    def __call__(self, img: Image) -> Image:
        return self.t_transformer(img)


@TRANSFORMERS.register
class PadResize(TransformerBase):
    """
    Resize image's longer edge to target size, and then pad the shorter edge to target size.

    Hyper-Params
        size (int): desired output size of the longer edge.
        padding_v (sequence): padding pixel value.
        interpolation (int): desired interpolation.
    """
    default_hyper_params = {
        "size": 224,
        "padding_v": [124, 116, 104],
        "interpolation": Image.BILINEAR,
    }

    def __init__(self, hps: Dict or None = None):
        """
        Args:
            hps: default hyper parameters in a dict (keys, values).
        """
        super(PadResize, self).__init__(hps)

    def __call__(self, img: Image) -> Image:
        target_size = self._hyper_params["size"]
        padding_v = tuple(self._hyper_params["padding_v"])
        interpolation = self._hyper_params["interpolation"]

        w, h = img.size
        if w > h:
            img = img.resize((int(target_size), int(h * target_size * 1.0 / w)), interpolation)
        else:
            img = img.resize((int(w * target_size * 1.0 / h), int(target_size)), interpolation)

        ret_img = Image.new("RGB", (target_size, target_size), padding_v)
        w, h = img.size
        st_w = int((ret_img.size[0] - w) / 2.0)
        st_h = int((ret_img.size[1] - h) / 2.0)
        ret_img.paste(img, (st_w, st_h))
        return ret_img


@TRANSFORMERS.register
class ShorterResize(TransformerBase):
    """
    Resize image's shorter edge to target size, while keep h: w ratio.

    Hyper-Params
        size (int): desired output size.
        interpolation (int): desired interpolation.
    """
    default_hyper_params = {
        "size": 224,
        "interpolation": Image.BILINEAR,
    }

    def __init__(self, hps: Dict or None = None):
        """
        Args:
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        super(ShorterResize, self).__init__(hps)
        self.t_transformer = TResize(self._hyper_params["size"], self._hyper_params["interpolation"])

    def __call__(self, img: Image) -> Image:
        return self.t_transformer(img)


@TRANSFORMERS.register
class CenterCrop(TransformerBase):
    """
    A wrapper from CenterCrop in pytorch, see torchvision.transformers.CenterCrop for explanation.

    Hyper-Params
        size(sequence or int): desired output size.
    """
    default_hyper_params = {
        "size": 224,
    }

    def __init__(self, hps: Dict or None = None):
        """
        Args:
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        super(CenterCrop, self).__init__(hps)
        self.t_transformer = TCenterCrop(self._hyper_params["size"])

    def __call__(self, img: Image) -> Image:
        return self.t_transformer(img)


@TRANSFORMERS.register
class ToTensor(TransformerBase):
    """
    A wrapper from ToTensor in pytorch, see torchvision.transformers.ToTensor for explanation.
    """
    default_hyper_params = dict()

    def __init__(self, hps: Dict or None = None):
        """
        Args:
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        super(ToTensor, self).__init__(hps)
        self.t_transformer = TToTensor()

    def __call__(self, imgs: Image or tuple) -> torch.Tensor:
        if not isinstance(imgs, tuple):
            imgs = [imgs]
        ret_tensor = list()
        for img in imgs:
            ret_tensor.append(self.t_transformer(img))
        ret_tensor = torch.stack(ret_tensor, dim=0)
        return ret_tensor


@TRANSFORMERS.register
class ToCaffeTensor(TransformerBase):
    """
    Create tensors for models trained in caffe.
    """
    default_hyper_params = dict()

    def __init__(self, hps: Dict or None = None):
        """
        Args:
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        super(ToCaffeTensor, self).__init__(hps)

    def __call__(self, imgs: Image or tuple) -> torch.tensor:
        if not isinstance(imgs, tuple):
            imgs = [imgs]

        ret_tensor = list()
        for img in imgs:
            img = np.array(img, np.int32, copy=False)
            r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            img = np.stack([b, g, r], axis=2)
            img = torch.from_numpy(img)
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
            img = img.float()
            ret_tensor.append(img)
        ret_tensor = torch.stack(ret_tensor, dim=0)
        return ret_tensor


@TRANSFORMERS.register
class Normalize(TransformerBase):
    """
    Normalize a tensor image with mean and standard deviation.

    Hyper-Params
        mean (sequence): sequence of means for each channel.
        std (sequence): sequence of standard deviations for each channel.
    """
    default_hyper_params = {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }

    def __init__(self, hps: Dict or None = None):
        """
        Args:
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        super(Normalize, self).__init__(hps)
        for v in ["mean", "std"]:
            self.__dict__[v] = np.array(self._hyper_params[v])[None, :, None, None]
            self.__dict__[v] = torch.from_numpy(self.__dict__[v]).float()

    def __call__(self, tensor: torch.tensor) -> torch.tensor:
        assert tensor.ndimension() == 4
        tensor.sub_(self.mean).div_(self.std)
        return tensor


@TRANSFORMERS.register
class TenCrop(TransformerBase):
    """
    A wrapper from TenCrop in pytorch，see torchvision.transformers.TenCrop for explanation.

    Hyper-Params
        size (sequence or int): desired output size.
    """
    default_hyper_params = {
        "size": 224,
    }

    def __init__(self, hps: Dict or None = None):
        """
        Args:
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        super(TenCrop, self).__init__(hps)
        self.t_transformer = TTenCrop(self._hyper_params["size"])

    def __call__(self, img: Image) -> Image:
        return self.t_transformer(img)


@TRANSFORMERS.register
class TwoFlip(TransformerBase):
    """
    Return the image itself and its horizontal flipped one.
    """
    default_hyper_params = dict()

    def __init__(self, hps: Dict or None = None):
        """
        Args:
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        super(TwoFlip, self).__init__(hps)

    def __call__(self, img: Image) -> (Image, Image):
        return img, hflip(img)
