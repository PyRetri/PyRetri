# -*- coding: utf-8 -*-

from utils.search_modules import SearchModules
from pyretri.config import get_defaults_cfg

pre_processes = SearchModules()

pre_processes.add(
    "Direct256128",
    {
        "batch_size": 32,
        "folder": {
            "name": "Folder"
        },
        "collate_fn": {
            "name": "CollateFn"
        },
        "transformers": {
            "names": ["DirectResize", "TwoFlip", "ToTensor", "Normalize"],
            "DirectResize": {
                "size": (256, 128),
                "interpolation": 3
            },
            "Normalize": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225]
            }
        }
    }
)

cfg = get_defaults_cfg()

pre_processes.check_valid(cfg["datasets"])
