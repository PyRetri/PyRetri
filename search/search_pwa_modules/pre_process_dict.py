# -*- coding: utf-8 -*-

from utils.search_modules import SearchModules
from pyretri.config import get_defaults_cfg

pre_processes = SearchModules()

pre_processes.add(
    "Shorter256Center224",
    {
        "batch_size": 32,
        "folder": {
            "name": "Folder"
        },
        "collate_fn": {
            "name": "CollateFn"
        },
        "transformers": {
            "names": ["ShorterResize", "CenterCrop", "ToTensor", "Normalize"],
            "ShorterResize": {
                "size": 256
            },
            "CenterCrop": {
                "size": 224
            },
            "Normalize": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225]
            }
        }
    }
)

pre_processes.add(
    "Direct224",
    {
        "batch_size": 32,
        "folder": {
            "name": "Folder"
        },
        "collate_fn": {
            "name": "CollateFn"
        },
        "transformers": {
            "names": ["DirectResize", "ToTensor", "Normalize"],
            "DirectResize": {
                "size": (224, 224)
            },
            "Normalize": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225]
            }
        }
    }
)

pre_processes.add(
    "PadResize224",
    {
        "batch_size": 32,
        "folder": {
            "name": "Folder"
        },
        "collate_fn": {
            "name": "CollateFn"
        },
        "transformers": {
            "names": ["PadResize", "ToTensor", "Normalize"],
            "PadResize": {
                "size": 224,
                "padding_v": [124, 116, 104]
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
