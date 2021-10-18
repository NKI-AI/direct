[![tox](https://github.com/directgroup/direct/actions/workflows/tox.yml/badge.svg)](https://github.com/directgroup/direct/actions/workflows/tox.yml)
[![pylint](https://github.com/directgroup/direct/actions/workflows/pylint.yml/badge.svg)](https://github.com/directgroup/direct/actions/workflows/pylint.yml)
[![black](https://github.com/directgroup/direct/actions/workflows/black.yml/badge.svg)](https://github.com/directgroup/direct/actions/workflows/black.yml)

# DIRECT: Deep Image REConstruction Toolkit
`DIRECT` is a Python, end-to-end pipeline for solving Inverse Problems emerging in Imaging Processing. It is built with PyTorch and stores state-of-the-art Deep Learning imaging inverse problem solvers such as denoising, dealiasing and reconstruction. By defining a base forward linear or non-linear operator, `DIRECT` can be used for training models for recovering images such as MRIs from partially observed or noisy input data. 

`DIRECT` stores inverse problem solvers such as the Learned Primal Dual algorithm and Recurrent Inference Machine, which were part of the winning solution in Facebook & NYUs FastMRI challenge in 2019 and the Calgary-Campinas MRI reconstruction challenge at MIDL 2020. For a full list of the baselines currently implemented in DIRECT see [here](#baselines-and-trained-models). 

<div align="center">
  <img src=".github/direct.png"/>
</div>

## Installation
See [install.md](install.md).

## Quick Start
See [getting_started.md](getting_started.md), check out the [documentation](https://docs.aiforoncology.nl/direct).
In the [projects](projects) folder examples are given on how to train models on public datasets.

## Baselines and trained models

- Recurrent Inference Machine (RIM) 
 - End-to-end Variational Network (VarNet) 
 - Learned Primal Dual Network (LDPNet) 
 - X-Primal Dual Network (XPDNet)
 - KIKI-Net
 - U-Net
 - Joint-ICNet 
 - AIRS Medical fastmri model (MultiDomainNet)

We provide a set of baseline results and trained models in the [DIRECT Model Zoo](model_zoo.md).

## License
DIRECT is released under the [Apache 2.0 License](LICENSE).

## Citing DIRECT
If you use DIRECT in your own research, or want to refer to baseline results published in the
 [DIRECT Model Zoo](model_zoo.md), please use the following BiBTeX entry:

```BibTeX
@misc{DIRECTTOOLKIT,
  author =       {Yiasemis, George and  and Moriakov, Nikita and Karkalousos, Dimitrios  and Caan, Matthan and Teuwen, Jonas},
  title =        {DIRECT: Deep Image REConstruction Toolkit},
  howpublished = {\url{https://github.com/directgroup/direct}},
  year =         {2021}
}
```
