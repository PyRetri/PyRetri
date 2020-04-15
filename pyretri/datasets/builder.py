# -*- coding: utf-8 -*-

from yacs.config import CfgNode

from .registry import COLLATEFNS, FOLDERS, TRANSFORMERS
from .collate_fn import CollateFnBase
from .folder import FolderBase
from .transformer import TransformerBase

from ..utils import simple_build

from torch.utils.data import DataLoader

from torchvision.transforms import Compose


def build_collate(cfg: CfgNode) -> CollateFnBase:
    """
    Instantiate a collate class with the given configuration tree.

    Args:
        cfg (CfgNode): the configuration tree.

    Returns:
        collate (CollateFnBase): a collate class.
    """
    name = cfg["name"]
    collate = simple_build(name, cfg, COLLATEFNS)
    return collate


def build_transformers(cfg: CfgNode) -> Compose:
    """
    Instantiate a compose class containing several transforms with the given configuration tree.

    Args:
        cfg (CfgNode): the configuration tree.

    Returns:
        transformers (Compose): a compose class.
    """
    names = cfg["names"]
    transformers = list()
    for name in names:
        transformers.append(simple_build(name, cfg, TRANSFORMERS))
    transformers = Compose(transformers)
    return transformers


def build_folder(data_json_path: str, cfg: CfgNode) -> FolderBase:
    """
    Instantiate a folder class with the given configuration tree.

    Args:
        data_json_path (str): the path of the data json file.
        cfg (CfgNode): the configuration tree.

    Returns:
        folder (FolderBase): a folder class.
    """
    trans = build_transformers(cfg.transformers)
    folder = simple_build(cfg.folder["name"], cfg.folder, FOLDERS, data_json_path=data_json_path, transformer=trans)
    return folder


def build_loader(folder: FolderBase, cfg: CfgNode) -> DataLoader:
    """
    Instantiate a data loader class with the given configuration tree.

    Args:
        folder (FolderBase): the folder function.
        cfg (CfgNode): the configuration tree.

    Returns:
        data_loader (DataLoader): a data loader class.
    """
    co_fn = build_collate(cfg.collate_fn)

    data_loader = DataLoader(folder, cfg["batch_size"], collate_fn=co_fn, num_workers=8, pin_memory=True)

    return data_loader
