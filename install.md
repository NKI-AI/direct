# Installation

## Requirements
- CUDA 10.2 supported GPU.
- Linux with Python ≥ 3.8
- PyTorch ≥ 1.6

## Install using Docker

We provide a [Dockerfile](docker) which install DIRECT with a few commands. While recommended due to the use of specific
pytorch features, DIRECT should also work in a virtual environment.

## Install using `conda`

1. Clone the repository using `git clone`.

2. First, install conda. Here is a guide on how to install conda on Linux if you don't already have it [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html). If you downloaded conda for the first time it is possible that you will need to restart your machine.  Once you have conda, create a python 3.9 conda environment:
```
conda create -n myenv python=3.9
```
Then, activate the virtual environment `myenv` you created where you will install the software:
```
conda activate myenv
```

3. If you are using GPUs, cuda is required for the project to run. To install [PyTorch](https://pytorch.org/get-started/locally/) with cuda run:
```
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```  
**otherwise**, install the CPU PyTorch installation (not recommended):
```
pip3 install torch==1.10.0+cpu torchvision==0.11.1+cpu torchaudio==0.10.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
```

4. To download `direct` as a python module navigate to `direct/direct/` and run
```
python3 setup.py install
```

## Common Installation Issues
If you met issues using DIRECT, please first update the repository to the latest version, and rebuild the docker. When
this does not work, create a GitHub issue so we can see whether this is a bug, or an installation problem.
