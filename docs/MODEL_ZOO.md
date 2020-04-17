# Model Zoo

Here we provide the pre-trained models to help you reproduce our experimental results easily.

## General image retrieval

### pre-trained models

|     Training Set     | Backbone  | for Short |                           Download                           |
| :------------------: | :-------: | :-------: | :----------------------------------------------------------: |
|       ImageNet       |  VGG-16   |  I-VGG16  | [model](https://download.pytorch.org/models/vgg16-397923af.pth) |
|      Places365       |  VGG-16   |  P-VGG16  | [model](https://drive.google.com/open?id=1U_VWbn_0L9mSDCBGiAIFbXxMvBeOiTG9) |
| ImageNet + Places365 |  VGG-16   |  H-VGG16  | [model](https://drive.google.com/open?id=11zE5kGNeeAXMhlHNv31Ye4kDcECrlJ1t) |
|       ImageNet       | ResNet-50 |  I-Res50  | [model](https://download.pytorch.org/models/resnet50-19c8e357.pth) |
|      Places365       | ResNet-50 |  P-Res50  | [model](https://drive.google.com/open?id=1lp_nNw7hh1MQO_kBW86GG8y3_CyugdS2)  |
| ImageNet + Places365 | ResNet-50 |  H-Res50  | [model](https://drive.google.com/open?id=1_USt_gOxgV4NJ9Zjw_U8Fq-1HEC_H_ki)  |

### performance

|  Dataset   |     Data Augmentation      | Backbone | Pooling |  Dimension Process   | mAP  |
| :--------: | :------------------------: | :------: | :-----: | :------------------: | :--: |
|  Oxford5k  | ShorterResize + CenterCrop | H-VGG16  |   GAP   | l2 +SVD(whiten) + l2 | 62.9 |
|  CUB-200   | ShorterResize + CenterCrop | I-Res50  |  SCDA   |    l2 + PCA + l2     | 27.8 |
|   Indoor   |        DirectResize        | P-Res50  |  CroW   |    l2 + PCA + l2     | 51.8 |
| Caltech101 |         PadResize          | I-Res50  |   GeM   |    l2 + PCA + l2     | 77.9 |

Choosing the implementations mentioned above as baselines and adding some tricks, we have:

|  Dataset   |          Implementations           | mAP  |
| :--------: | :--------------------------------: | :--: |
|  Oxford5k  |      baseline + K-reciprocal       | 72.9 |
|  CUB-200   |      baseline + K-reciprocal       | 38.9 |
|   Indoor   |        baseline + DBA + QE         | 63.7 |
| Caltech101 | baseline + DBA + QE + K-reciprocal | 86.1 |

## Person re-identification

For person re-identification, we use the model provided by [Person_reID_baseline](https://github.com/layumi/Person_reID_baseline_pytorch) and reproduce its resutls. In addition, we train a model on DukeMTMC-reID through the open source code for further experiments.

### pre-trained models

| Training Set  | Backbone  | for Short | Download |
| :-----------: | :-------: | :-------: | :------: |
|  Market-1501  | ResNet-50 |  M-Res50  | [model](https://drive.google.com/open?id=1-6LT_NCgp_0ps3EO-uqERrtlGnbynWD5)         |
| DukeMTMC-reID | ResNet-50 |  D-Res50  | [model](https://drive.google.com/open?id=1X2Tiv-SQH3FxwClvBUalWkLqflgZHb9m)      |

### performance

|    Dataset    |   Data Augmentation    | Backbone | Pooling | Dimension Process | mAP  | Recall@1 |
| :-----------: | :--------------------: | :------: | :-----: | :---------------: | ---- | :------: |
|  Market-1501  | DirectResize + TwoFlip | M-Res50  |   GAP   |        l2         | 71.6 |   88.8   |
| DukeMTMC-reID | DirectResize + TwoFlip | D-Res50  |   GAP   |        l2         | 62.5 |   80.4   |

Choosing the implementations mentioned above as baselines and adding some tricks, we have:

|    Dataset    |             Implementations             | mAP  | Recall@1 |
| :-----------: | :-------------------------------------: | :--: | :------: |
|  Market-1501  | Baseline + l2 + PCA + l2 + K-reciprocal | 84.8 |   90.4   |
| DukeMTMC-reID | Baseline + l2 + PCA + l2 + K-reciprocal | 78.3 |   84.2   |

