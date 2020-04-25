# Getting started

This page provides basic tutorials about the usage of PyRetri. For installation instructions and dataset preparation, please see [INSTALL.md](../docs/INSTALL.md).

## Make Data Json

After the gallery set and query set are separated, we package the information of each sub-dataset in pickle format for further process. We use different types to package different structured folders: `general`, `oxford` and `reid`. 

The general object recognition dataset collects images with the same label in one directory and the folder structure should be like this:

```shell
# type: general
general_recognition
├── class A
│   ├── XXX.jpg
│   └── ···
├── class B
│   ├── XXX.jpg
│   └── ···
└── ···
```

Oxford5k is a typical dataset in image retrieval field and the folder structure is as follows:

```shell
# type: oxford
oxford
├── gt
│   ├── XXX.txt
│   └── ···
└── images
    ├── XXX.jpg
    └── ···

```

The person re-identification dataset have already split the query set and gallery set, its folder structure should be like this:

```shell
# type: reid
person_re_identification
├── bounding_box_test
│   ├── XXX.jpg
│   └── ···
├── query
│   ├── XXX.jpg
│   └── ···
└── ···
```

Choosing the mode carefully, you can generate data jsons by: 

```shell
python3 main/make_data_json.py [-d ${dataset}] [-sp ${save_path}] [-t ${type}] [-gt ${ground_truth}]
```

Auguments:

- `data`: Path of the dataset for generating data json file.
- `save_path`: Path for saving the output file.
- `type`: Type of the dataset collecting images. For dataset collecting images with the same label in one directory, we use `general`. For oxford dataset, we use `oxford`. For re-id dataset, we use `reid`.
- `ground_truth`: Optional. Path of the gt information, which is necessary for generating data json file of oxford dataset.

Examples:

```shell
# for dataset collecting images with the same label in one directory
python3 main/make_data_json.py -d /data/caltech101/gallery/ -sp data_jsons/caltech_gallery.json -t general

python3 main/make_data_json.py -d /data/caltech101/query/ -sp data_jsons/caltech_query.json -t general

# for oxford dataset
python3 main/make_data_json.py -d /data/cbir/oxford/gallery/ -sp data_jsons/oxford_gallery.json -t oxford -gt /data/cbir/oxford/gt/

python3 main/make_data_json.py -d /data/cbir/oxford/query/ -sp data_jsons/oxford_query.json -t oxford -gt /data/cbir/oxford/gt/

# for re-id dataset
python3 main/make_data_json.py -d /data/market1501/bounding_box_test/ -sp data_jsons/market_gallery.json -t reid

python3 main/make_data_json.py -d /data/market1501/query/ -sp data_jsons/market_query.json -t reid
```

Note: Oxford dataset contains the ground truth of each query image in a txt file, so remember to give the path of gt file when generating data json file of Oxford.

## Extract

All outputs (features and labels) will be saved to the target directory in pickle format.

Extract feature for each data json file by:

```shell
python3 main/extract_feature.py [-dj ${data_json}] [-sp ${save_path}] [-cfg ${config_file}] [-si ${save_interval}]
```

Arguments:

- `data_json`: Path of the data json file to be extrated.
- `save_path`: Path for saving the output features in pickle format.

- `config_file`: Path of the configuration file in yaml format.
- `save_interval`: Optional. It is the number of features saved in one part file, which is set to 5000 by default.

```shell
# extract features of gallert set and query set
python3 main/extract_feature.py -dj data_jsons/caltech_gallery.json -sp /data/features/caltech/gallery/ -cfg configs/caltech.yaml

python3 main/extract_feature.py -dj data_jsons/caltech_query.json -sp /data/features/caltech/query/ -cfg configs/caltech.yaml
```

## Index

The path of query set features and gallery set features is specified in the config file.

After extracting gallery set features and query set features, you can index the query set features by:

```shell
python3 main/index.py [-cfg ${config_file}]
```

Arguments:

- `config_file`: Path of the configuration file in yaml format.

Examples:

```shell
python3 main/index.py -cfg configs/caltech.yaml
```

## Single Image Index

For visulization results and wrong case analysis, we provide the script for single query image and you can visualize or save the retrieval results easily. 

Use this command to single image index:

```shell
python3 main/single_index.py [-cfg ${config_file}]
```

Arguments:

- `config_file`: Path of the configuration file in yaml format.

Examples:

```shell
python3 main/single_index.py -cfg configs/caltech.yaml
```

Please see [single_index.py](../main/single_index.py) for more details.

## Add Your Own Module

We basically categorize retrieval process into 4 components.

- model: the pre-trained model for feature extraction.
- extract: assign which layer to output, including splitter functions and aggregation methods.
- index: index features, including dimension process, feature enhance, distance metric and re-rank.
- evaluate: evaluate retrieval results, outputting recall and mAP results.

Here we show how to add your own model to extract features.

1. Create your model file `pyretri/models/backbone/backbone_impl/reid_baseline.py`.

```shell
import torch.nn as nn

from ..backbone_base import BackboneBase
from ...registry import BACKBONES

@BACKBONES.register
class ft_net(BackboneBase):
    def __init__(self):
        pass

    def forward(self, x):
        pass
```

​       or

```shell
import torch.nn as nn

from ..backbone_base import BackboneBase
from ...registry import BACKBONES

class FT_NET(BackboneBase):
    def __init__(self):
        pass

    def forward(self, x):
        pass

@BACKBONES.register
def ft_net():
    model = FT_NET()
    return model
```

2. Import the module in `pyretri/models/backbone/__init__.py`.

```shell
from .backbone_impl.reid_baseline import ft_net

__all__ = [
    'ft_net',
]
```

3. Use it in your config file.

```shell
model:
  name: "ft_net"
  ft_net:
    load_checkpoint: "/data/my_model_zoo/res50_market1501.pth"
```

## Pipeline Combinations Search

Since tricks used in each stage have a signicant impact on retrieval performance, we present the pipeline combinations search scripts to help users to find possible combinations of approaches with various hyper-parameters. 

### Get into the combinations search scripts

```shell
cd search/
```

### Define Search Space

We decompose the search space into three sub search spaces: pre_process, extract and index, each of which corresponds to a specified file. Search space is defined by  adding methods with hyper-parameters to a specified dict.  You can add a search operator as follows:

```shell
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
```

By doing this, a pre_process operator named "PadResize224" is added to the data_process sub search space and will be searched in the following process.

### Search

Similar to the image retrieval pipeline, combinations search includes two stages: search for feature extraction and search for indexing.

#### search for feature extraction

Search for the feature extraction combinations by:

```shell
python3 search_extract.py [-sp ${save_path}] [-sp ${search_modules}] 
```

Arguments:

- `save_path`: path for saving the output features in pickle format.
- `search_modules`: name of the folder containing search space files.

Examples:

```shell
python3 search_extract.py -sp /data/features/gap_gmp_gem_crow_spoc/ -sm search_modules
```

#### search for indexing

Search for the indexing combinations by:

```shell
python3 search_index.py [-fd ${fea_dir}] [-sm ${search_modules}] [-sp ${save_path}]
```

Arguments:

- `fea_dir`: path of the output features extracted by the feature extraction combinations search.
- `search_modules`: name of the folder containing search space files.
- `save_path`: path for saving the retrieval results of each combination.

Examples:

```shell
python3 search_index.py -fd /data/features/gap_gmp_gem_crow_spoc/ -sm search_modules -sp /data/features/gap_gmp_gem_crow_spoc_result.json
```

#### show search results

We provide two ways to show the search results. One is to save all the search results in a csv format file, which can be used for further analyses. Another is to show the search results according to the given keywords. You can define the keywords as follows:

```sh
keywords = {
        'data_name': ['market'],
        'pre_process_name': list(),
        'model_name': list(),
        'feature_map_name': list(),
        'aggregator_name': list(),
        'post_process_name': ['no_fea_process', 'l2_normalize', 'pca_whiten', 'pca_wo_whiten'],
    }
```

Show the search results by:

```shell
show_search_results.py [-r ${result_json_path}] 
```

Arguments:

- `result_json_path`: path of the result json file.

Examples:

```shell
show_search_results.py -r /data/features/gap_gmp_gem_crow_spoc_result.json
```

See [show_search_results.py](../search/show_search_results.py) for more details.

