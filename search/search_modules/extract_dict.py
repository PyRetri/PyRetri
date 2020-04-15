# -*- coding: utf-8 -*-

from utils.search_modules import SearchModules
from pyretri.config import get_defaults_cfg

models = SearchModules()
extracts = SearchModules()


models.add(
    "imagenet_vgg16",
    {
        "name": "vgg16",
        "vgg16": {
            "load_checkpoint": "torchvision://vgg16"
        }
    }
)
extracts.add(
    "imagenet_vgg16",
    {
        "extractor": {
            "name": "VggSeries",
            "VggSeries": {
                "extract_features": ["all"],
            }
        },
        "splitter": {
            "name": "Identity",
        },
        "aggregators": {
            "names": ["Crow", "GAP", "GMP", "GeM", "SPoC"]
        },
    }
)


models.add(
    "imagenet_res50",
    {
        "name": "resnet50",
        "resnet50": {
            "load_checkpoint": "torchvision://resnet50"
        }
    }
)
extracts.add(
    "imagenet_res50",
    {
        "extractor": {
            "name": "ResSeries",
            "ResSeries": {
                "extract_features": ["all"],
            }
        },
        "splitter": {
            "name": "Identity",
        },
        "aggregators": {
            "names": ["Crow", "GAP", "GMP", "GeM", "SPoC"]
        },
    }
)


models.add(
    "places365_res50",
    {
        "name": "resnet50",
        "resnet50": {
            "load_checkpoint": "/data/places365_model/res50_places365.pt"
        }
    }
)
extracts.add(
    "places365_res50",
    {
        "extractor": {
            "name": "ResSeries",
            "ResSeries": {
                "extract_features": ["all"],
            }
        },
        "splitter": {
            "name": "Identity",
        },
        "aggregators": {
            "names": ["Crow", "GAP", "GMP", "GeM", "SPoC"]
        },
    }
)


models.add(
    "hybrid1365_res50",
    {
        "name": "resnet50",
        "resnet50": {
            "load_checkpoint": "/data/places365_model/res50_hybrid1365.pt"
        }
    }
)
extracts.add(
    "hybrid1365_res50",
    {
        "extractor": {
            "name": "ResSeries",
            "ResSeries": {
                "extract_features": ["all"],
            }
        },
        "splitter": {
            "name": "Identity",
        },
        "aggregators": {
            "names": ["Crow", "GAP", "GMP", "GeM", "SPoC"]
        },
    }
)

cfg = get_defaults_cfg()

models.check_valid(cfg["model"])
extracts.check_valid(cfg["extract"])
