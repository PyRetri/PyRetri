# -*- coding: utf-8 -*-

from .collate_fn_impl.collate_fn import CollateFn
from .collate_fn_base import CollateFnBase

__all__ = [
    'CollateFnBase',
    'CollateFn',
]
