# MusiteDeep Installation Guide

## Environment Setup

### 1. Create conda environment
```bash
mamba create -n musitedeep python=3.5.2 -y
```

### 2. Install dependencies
```bash
# Install grpcio first
mamba install -c conda-forge grpcio

# Install Python packages using Tsinghua mirror
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy scipy scikit-learn pillow h5py pandas keras==2.2.4 tensorflow==1.12.0
```

### 3. Activate environment
```bash
conda activate musitedeep
```

## Usage

After environment setup, follow the instructions in README.md to use MusiteDeep for PTM prediction.

### GPU Support
For GPU support, additionally install:
- [CUDA](https://developer.nvidia.com/cuda-toolkit)
- [cuDNN](https://developer.nvidia.com/cudnn)

Then replace tensorflow with:
```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorflow-gpu==1.12.0
```
