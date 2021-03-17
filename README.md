# LDDMM-Face: 

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
