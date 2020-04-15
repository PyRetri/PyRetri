# -*- coding: utf-8 -*-

from utils.search_modules import SearchModules
from pyretri.config import get_defaults_cfg

queries = SearchModules()
evaluates = SearchModules()

queries.add(
    "no_fea_process",
    {
        "gallery_fea_dir": "",
        "query_fea_dir": "",

        "feature_names": [],

        "dim_processors": {
            "names": ["Identity"],
        },

        "feature_enhancer": {
            "name": "Identity"
        },

        "metric": {
            "name": "KNN"
        },

        "re_ranker": {
            "name": "Identity"
        }
    }
)

queries.add(
    "l2_normalize",
    {
        "gallery_fea_dir": "",
        "query_fea_dir": "",

        "feature_names": [],

        "dim_processors": {
            "names": ["L2Normalize"],
        },

        "feature_enhancer": {
            "name": "Identity"
        },

        "metric": {
            "name": "KNN"
        },

        "re_ranker": {
            "name": "Identity"
        }
    }
)

queries.add(
    "pca_wo_whiten",
    {
        "gallery_fea_dir": "",
        "query_fea_dir": "",

        "feature_names": [],

        "dim_processors": {
            "names": ["PartPCA"],
            "PartPCA": {
                "whiten": False,
                "train_fea_dir": "",
                "proj_dim": 512
            }
        },

        "feature_enhancer": {
            "name": "Identity"
        },

        "metric": {
            "name": "KNN"
        },

        "re_ranker": {
            "name": "Identity"
        }
    }
)

queries.add(
    "pca_whiten",
    {
        "gallery_fea_dir": "",
        "query_fea_dir": "",

        "feature_names": [],

        "dim_processors": {
            "names": ["PartPCA"],
            "PartPCA": {
                "whiten": True,
                "train_fea_dir": "",
                "proj_dim": 512
            }
        },

        "feature_enhancer": {
            "name": "Identity"
        },

        "metric": {
            "name": "KNN"
        },

        "re_ranker": {
            "name": "Identity"
        }
    }
)

queries.add(
    "svd_wo_whiten",
    {
        "gallery_fea_dir": "",
        "query_fea_dir": "",

        "feature_names": [],

        "dim_processors": {
            "names": ["PartSVD"],
            "PartSVD": {
                "whiten": False,
                "train_fea_dir": "",
                "proj_dim": 511
            }
        },

        "feature_enhancer": {
            "name": "Identity"
        },

        "metric": {
            "name": "KNN"
        },

        "re_ranker": {
            "name": "Identity"
        }
    }
)

queries.add(
    "svd_whiten",
    {
        "gallery_fea_dir": "",
        "query_fea_dir": "",

        "feature_names": [],

        "dim_processors": {
            "names": ["PartSVD"],
            "PartSVD": {
                "whiten": True,
                "train_fea_dir": "",
                "proj_dim": 511
            }
        },

        "feature_enhancer": {
            "name": "Identity"
        },

        "metric": {
            "name": "KNN"
        },

        "re_ranker": {
            "name": "Identity"
        }
    }
)

evaluates.add(
    "overall",
    {
        "evaluator": {
            "name": "OverAll"
        }
    }
)

evaluates.add(
    "oxford_overall",
    {
        "evaluator": {
            "name": "OxfordOverAll"
        }
    }
)

evaluates.add(
    "reid_overall",
    {
        "evaluator": {
            "name": "ReIDOverAll"
        }
    }
)

cfg = get_defaults_cfg()

queries.check_valid(cfg["index"])
evaluates.check_valid(cfg["evaluate"])
