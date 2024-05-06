# LDDMM-Face: Large deformation diffeomorphic metric learning for cross-annotation face alignment

By Huilin Yang†, Junyan Lyu†, Pujin Cheng, Roger Tam, Xiaoying Tang.

This repository contains an official implementation of LDDMM-Face for the **Pattern Recognition** paper ["LDDMM-Face: Large deformation diffeomorphic metric learning for cross-annotation face alignment"](https://doi.org/10.1016/j.patcog.2024.110569).

## Quick start
### Environment
This code is developed using on **Python 3.6** and Pytorch 1.7.1 on CentOS 7 with NVIDIA GPUs. Training and testing are performed using 1 RTX 3090 GPU with CUDA 11.0. Other platforms or GPUs are not fully tested.

### Install
1. Install Pytorch
2. Install dependencies
```shell
pip install -r requirements.txt
```

### Train
Please specify the configuration file in ```experiments```.
```shell
python tools/train.py --cfg <CONFIG-FILE>
```

### Test
```shell
python tools/test.py --cfg <CONFIG-FILE> --model-file <MODEL-FILE>
```

## Citation
If you find this repository useful, please consider citing LDDMM-Face paper:

```
@article{YANG2024110569,
title = {LDDMM-Face: Large deformation diffeomorphic metric learning for cross-annotation face alignment},
journal = {Pattern Recognition},
pages = {110569},
year = {2024},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2024.110569},
url = {https://www.sciencedirect.com/science/article/pii/S0031320324003200},
author = {Huilin Yang and Junyan Lyu and Pujin Cheng and Roger Tam and Xiaoying Tang},
keywords = {Face alignment, Facial landmarks, Diffeomorphic mapping, Deep learning}
}
```
