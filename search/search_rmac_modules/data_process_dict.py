# -*- coding: utf-8 -*-

from utils.search_modules import SearchModules
from retrieval_tool_box.config import get_defaults_cfg

data_processes = SearchModules()

data_processes.add(
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
            "names": ["ResizeShorter", "CenterCrop", "ToTensor", "Normalize"],
            "ResizeShorter": {
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

data_processes.add(
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

data_processes.add(
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

data_processes.check_valid(cfg["datasets"])
