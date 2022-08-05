# HIAST: Hard-aware Instance Adaptive Self-training for Unsupervised Cross-domain Semantic Segmentation

## Overview

### Introduction

The divergence between labeled training data and unlabeled testing data is a significant challenge for recent deep learning models. Unsupervised domain adaptation (UDA) attempts to solve such problem. Recent works show that self-training is a powerful approach to UDA. However, existing methods have difficulty in balancing the scalability and performance. In this paper, we propose a hard-aware instance adaptive self-training framework for UDA on the task of semantic segmentation. To effectively improve the quality and diversity of pseudo-labels, we develop a novel pseudo-label generation strategy with an instance adaptive selector. We further enrich the hard class pseudo-labels with inter-image information through a skillfully designed hard-aware pseudo-label augmentation. Besides, we propose the region-adaptive regularization to smooth the pseudo-label region and sharpen the non-pseudo-label region. For the non-pseudo-label region, consistency constraint is also constructed to introduce stronger supervision signals during model optimization. Our method is so concise and efficient that it is easy to be generalized to other UDA methods. Experiments on GTA5-to-Cityscapes, SYNTHIA-to-Cityscapes, and Cityscapes-to-Oxford RobotCar demonstrate the superior performance of our approach compared with the state-of-the-art methods.

<img src="imgs/framework.png" alt="framework" style="zoom: 80%;" />

### Result

|        UDA  Scenarios         | mIoU-19 | mIoU-16 | mIoU-13 | mIoU-9 |
| :---------------------------: | :-----: | :-----: | :-----: | :----: |
|      GTA5-to-Cityscapes       |  56.2   |    -    |    -    |   -    |
|     SYNTHIA-to-Cityscapes     |    -    |  54.6   |  61.7   |   -    |
| Cityscapes-to-Oxford RobotCar |    -    |    -    |    -    |  75.4  |

## Setup

### Environment

1. Create virtual environment with Python 3.7.3

```bash
conda create -n HIAST python=3.7.3
conda activate HIAST
```

2. Install requirements.

```bash
pip install -r requirements.txt
```

3. Install [apex](https://github.com/NVIDIA/apex#linux) for easy mixed precision and distributed training in Pytorch. If you encounter problems, please see this [solution](https://github.com/NVIDIA/apex/issues/802#issuecomment-618699214).

### Dataset

1. Download dataset.
   - GTA5: Download all image and label packages from [here](https://download.visinf.tu-darmstadt.de/data/from_games/).
   - Cityscapes: Download `leftImg8bit_trainvaltest.zip` and `gt_trainvaltest.zip` from [here](https://www.cityscapes-dataset.com/downloads/).
   - SYNTHIA: Download `SYNTHIA-RAND-CITYSCAPES` from [here](http://synthia-dataset.net/downloads/).
   - Oxford RobotCar: Download all image and label packages from [here](https://www.nec-labs.com/~mas/adapt-seg/adapt-seg.html).
2. Extract downloaded files and place them in `data/`, if you have already downloaded before, you can create symlinks for them. The dataset directory should be as follows:

```
HIAST
├── ...
├── data
│   ├── cityscapes
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val
│   ├── GTA5
│   │   ├── images
│   │   ├── labels
│   ├── SYNTHIA_RAND_CITYSCAPES
│   │   ├── RAND_CITYSCAPES
│   │   │   ├── RGB
│   │   │   ├── GT
├── ...
```

### Pretrained

We have provided the final [model file](https://drive.google.com/drive/folders/1-qdT1JqV0XKsk8h_b8zo7tPfF1o0b5Er?usp=sharing) of GTA5-to-Cityscapes for evaluation. Other pretrained files are being prepared.

## Training

Coming soon.

## Evaluation

```bash
cd code
python validate.py --config_file configs/gtav-to-cityscapes/validate.yaml --resume_from pretrained/gtav-to-cityscapes/HIAST_final.pth --color_mask_dir_path ../outputs
```

## Contact

If you encounter any problems please contact us without hesitation.

- Email: tangwenqi@bupt.edu.cn, czhu@bupt.edu.cn