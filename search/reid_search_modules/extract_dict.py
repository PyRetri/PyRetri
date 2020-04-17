# -*- coding: utf-8 -*-

from utils.search_modules import SearchModules
from pyretri.config import get_defaults_cfg

models = SearchModules()
extracts = SearchModules()


models.add(
    "market_res50",
    {
        "name": "ft_net",
        "ft_net": {
            "load_checkpoint": "/data/my_model_zoo/res50_market1501.pth"
        }
    }
)
extracts.add(
    "market_res50",
    {
        "assemble": 1,
        "extractor": {
            "name": "ReIDSeries",
            "ReIDSeries": {
                "extract_features": ["output"],
            }
        },
        "splitter": {
            "name": "Identity",
        },
        "aggregators": {
            "names": ["GAP"]
        },
    }
)

models.add(
    "duke_res50",
    {
        "name": "ft_net",
        "ft_net": {
            "load_checkpoint": "/home/songrenjie/projects/reID_baseline/model/ft_ResNet50/res50_duke.pth"
        }
    }
)
extracts.add(
    "duke_res50",
    {
        "assemble": 1,
        "extractor": {
            "name": "ReIDSeries",
            "ReIDSeries": {
                "extract_features": ["output"],
            }
        },
        "splitter": {
            "name": "Identity",
        },
        "aggregators": {
            "names": ["GAP"]
        },
    }
)


cfg = get_defaults_cfg()

models.check_valid(cfg["model"])
extracts.check_valid(cfg["extract"])
