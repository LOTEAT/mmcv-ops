# MMCV-OPs
 
### Introduction
MMCV-OPs is a simplfied operator libraries from [MMCV](https://github.com/open-mmlab/mmcv). In MMCV, there are a lot of operators. However, they are deeply encapsulated. Reading its source code or debugging is relatively difficult. Therefore, I create this repository for learning. For each operator, I organized a folder. The format of this folder is as follows
```shell
operator
│    ├── include
│       ├── xxx.hpp
│       ├── xxx.cuh
│    ├── kernel
│       ├── xxx.cpp
│       ├── xxx.cu
│    ├── xx.py
│    ├── xx.ipynb
```
In ipynb file, I write some comments of operator. Hope it can help you to understand these operators.

### Installation
#### Install Pytorch
Install PyTorch following (official instructions)[https://pytorch.org/get-started/locally/].
#### Install MMCV-OPs
```shell
pip install -e .
```

### Main Content
- [x] <a href='mmcv_ops/roi_align/roi_align.ipynb'>ROIAlign</a>