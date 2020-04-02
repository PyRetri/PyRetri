# -*- coding: utf-8 -*-

from utils.search_modules import SearchModules
from retrieval_tool_box.config import get_defaults_cfg

queries = SearchModules()
evaluates = SearchModules()

queries.add(
    "pca_wo_whiten",
    {
        "gallery_fea_dir": "",
        "query_fea_dir": "",

        "feature_names": [],

        "post_processor": {
            "name": "PartPCA",
            "PartPCA": {
                "whiten": False,
                "train_fea_dir": "",
                "proj_dim": 512
            }
        },

        "database_enhance": {
            "name": "Identity"
        },

        "metric": {
            "name": "KNN"
        },

        "re_rank": {
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

        "post_processor": {
            "name": "PartPCA",
            "PartPCA": {
                "whiten": True,
                "train_fea_dir": "",
                "proj_dim": 512
            }
        },

        "database_enhance": {
            "name": "Identity"
        },

        "metric": {
            "name": "KNN"
        },

        "re_rank": {
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

        "post_processor": {
            "name": "PartSVD",
            "PartSVD": {
                "whiten": False,
                "train_fea_dir": "",
                "proj_dim": 511
            }
        },

        "database_enhance": {
            "name": "Identity"
        },

        "metric": {
            "name": "KNN"
        },

        "re_rank": {
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

        "post_processor": {
            "name": "PartSVD",
            "PartSVD": {
                "whiten": True,
                "train_fea_dir": "",
                "proj_dim": 511
            }
        },

        "database_enhance": {
            "name": "Identity"
        },

        "metric": {
            "name": "KNN"
        },

        "re_rank": {
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

cfg = get_defaults_cfg()

queries.check_valid(cfg["query"])
evaluates.check_valid(cfg["evaluate"])
