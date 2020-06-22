Installation

## Requirements

- Linux (Windows is not officially supported)
- Python 3
- PyTorch 1.2.0 or higher
- torchvison 0.4.0 or higher
- numpy
- sklearn
- yacs
- tqdm

Our experiments are conducted on the following environment:

- Python 3.5
- PyTorch 1.2.0
- torchvision 0.4.0
- numpy 1.17.2
- sklearn 0.21.3
- tqdm 4.36.1

## Install PyRetri

1. Install PyTorch and torchvision following the official instructions. 

2. Clone the PyRetri repository.

```she
git clone https://github.com/PyRetri/PyRetri.git
cd PyRetri
```

3. Install PyRetri.

```shell
python3 setup.py install
```

## Prepare Datasets

### Datasets

In our experiments, we use four general image retrieval datasets and two person re-identification datasets.

- [Oxford5k](https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/): collecting crawling images from Flickr using the names of 11 different landmarks in Oxford, which stands for landmark recognition task.
- [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html): containing photos of 200 bird species, which represents fine-grained visual categorization task.
- [Indoor](http://web.mit.edu/torralba/www/indoor.html): containing indoor scene images with 67 categories, representing scene recognition task.
- [Caltech101](http://www.vision.caltech.edu/Image_Datasets/Caltech101/): consisting pictures of objects belonging to 101 categories, standing for general object recognition task.
- [Market-1501](http://www.liangzheng.com.cn/Project/project_reid.html): containing images taken on the Tsinghua campus under 6 camera viewpoints, representing person re-identification task.
- [DukeMTMC-reID](https://drive.google.com/file/d/1jjE85dRCMOgRtvJ5RQV9-Afs-2_5dY3O/view): containing images captured by 8 cameras, which is more challenging.

To reproduce our experimental results, you need to first download these datasets, then follow the following step to re-organize the dataset.

### Split Dataset

For image retrieval task, the dataset should be divided into two subset: query set and gallery set. If your dataset has been divided already, you can skip this step.

In order to help you to reproduce our results conventionally, we provide four txt files, each of which is the division protocol used in our experiments. Before using our script to split the dataset,  you should construct the four datasets like this:

```shell
data
├── caltech101
│   └── 101_ObjectCategories
│       ├── accordion
│       │   ├── image_0001.jpg
│       │   └── ··· 
│       └── ···  
├── cbir
│   ├── oxford
│   │   ├── gt
│   │   │   ├── all_souls_1_good.txt
│   │   │   └── ··· 
│   │   └── images
│   │       ├── all_souls_000000.jpg
│   │       └── ··· 
│   └── pairs
│       ├── gt
│       │   ├── defense_1_good.txt
│       │   └── ··· 
│       └── images
│           ├── defense
│           │   ├── paris_defense_000000.jpg
│           │   └── ··· 
│           └── ··· 
├── cub_200_2011
│   ├── images
│   │   ├── 001.Black_footed_Albatross
│   │   │   ├── Black_Footed_Albatross_0001_796111.jpg
│   │   │   └── ··· 
│   │   └── ···
│   └── ···
└── indoor
    ├── Images
    │ 	├── airport_inside
    │ 	│   ├── airport_inside_0001.jpg
    │ 	│   └── ··· 
    │   └── ···
    └── ···

```

Splitting txt files can be found in [split_file](../main/split_file) and you can use the following command to split the dataset mentioned above:

```shell
python3 main/split_dataset.py [-d ${dataset}] [-sf ${split_file}]
```

Arguments:

- `dataset`: Path of the dataset to be splitted.
- `split_file`: **Absolute Path** of the division protocol txt file, with each line corresponding to one image:<image_path> <is_gallery_image>. <image_path> corresponds to the relative path of the image, and a value of 1 or 0 for <is_gallery_image> denotes that the file is in the gallery or query set, respectively.

Examples:

```shell
python3 main/split_dataset.py -d /data/caltech101/ -sf main/split_file/caltech_split.txt
```

Then query folder and gallery folder will be created under the dataset folder.

Note: For Re-ID dataset, the images are well divided in advance, so we do not need to split it.
