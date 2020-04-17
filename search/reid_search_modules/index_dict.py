# -*- coding: utf-8 -*-

from utils.search_modules import SearchModules
from pyretri.config import get_defaults_cfg

indexes = SearchModules()
evaluates = SearchModules()

indexes.add(
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

indexes.add(
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

indexes.add(
    "pca_wo_whiten",
    {
        "gallery_fea_dir": "",
        "query_fea_dir": "",

        "feature_names": [],

        "dim_processors": {
            "names": ["L2Normalize", "PCA", "L2Normalize"],
            "PCA": {
                "whiten": False,
                "train_fea_dir": "",
                "proj_dim": 512,
                "l2": True,
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

indexes.add(
    "pca_whiten",
    {
        "gallery_fea_dir": "",
        "query_fea_dir": "",

        "feature_names": [],

        "dim_processors": {
            "names": ["L2Normalize", "PCA", "L2Normalize"],
            "PCA": {
                "whiten": True,
                "train_fea_dir": "",
                "proj_dim": 512,
                "l2": True,
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

indexes.add(
    "svd_wo_whiten",
    {
        "gallery_fea_dir": "",
        "query_fea_dir": "",

        "feature_names": [],

        "dim_processors": {
            "names": ["L2Normalize", "SVD", "L2Normalize"],
            "SVD": {
                "whiten": False,
                "train_fea_dir": "",
                "proj_dim": 511,
                "l2": True,
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

indexes.add(
    "svd_whiten",
    {
        "gallery_fea_dir": "",
        "query_fea_dir": "",

        "feature_names": [],

        "dim_processors": {
            "names": ["L2Normalize", "SVD", "L2Normalize"],
            "SVD": {
                "whiten": True,
                "train_fea_dir": "",
                "proj_dim": 511,
                "l2": True,
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

indexes.check_valid(cfg["index"])
evaluates.check_valid(cfg["evaluate"])
