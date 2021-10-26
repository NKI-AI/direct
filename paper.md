---
title: 'DIRECT: Deep Image REConstruction Toolkit'

tags:
    - Python
    - Pytorch
    - Deep Learning
    - Inverse Problem Solver
    - Image Processing
    - Deep MRI reconstruction
    - Accelerated MRI
authors:
    - name: George Yiasemis
      orchid: 0000-0002-1348-8987
      affiliation: "1, 2"
    - name: Nikita Moriakov
      affiliation: "1, 4"
    - name: Dimitrios Karkalousos
      affiliation: 3
    - name: Matthan Caan
      affiliation: 3
    - name: Jonas Teuwen
      affiliation: "1, 2, 4"
affiliations:
    - name: Netherlands Cancer Institute
      index: 1
    - name: University of Amsterdam
      index: 2
    - name: Amsterdam UMC, Biomedical Engineering and Physics
      index: 3
    - name: Radboud University Medical Center
      index: 4
date: 30 October 2021
bibliography: paper.bib
---

# Summary

`DIRECT` is a Python, end-to-end pipeline for solving Inverse Problems emerging in Imaging Processing. It is built with PyTorch and stores state-of-the-art Deep Learning imaging inverse problem solvers for solving inverse problems such as denoising, dealiasing and reconstruction. By defining a base forward linear or non-linear operator, `DIRECT` can be used for training models for recovering images such as MRIs from partially observed or noisy input data. Additionally, it provides the user with the functionality to load saved weights of pre-trained models to be used for inference. Furthermore, it offers functions for peparing and pre-processing data such as `.h5` files into PyTorch Datasets compatible with the software's training pipeline but also allows for flexibility to work with any kind of PyTorch Dataset. In order for the user to view the proccess of their experiments, it allows for continuous visualisation of training and validation metrics as well as image predictions utilising Tensorboard (examples are illustrated in Figures 1 and 2). `DIRECT` is integrated with cuda and cuDNN enabling users to train and validate models not only on CPU memory but on multiple GPUs as well, if they are available.  

| ![image](https://user-images.githubusercontent.com/71031687/138093195-67004ec7-6bfd-448b-ba53-4cdd291a471b.png) |
|:--:|
| <b> Figure 1:  Visualised reconstructions in Tensorboard <b> |

| ![image](https://user-images.githubusercontent.com/71031687/138097866-221aebb5-9aa3-4b8b-8a95-c0541ae52bb1.png) |
|:--:|
| <b> Figure 2:  Visualised metrics in Tensorboard <b> |   
 
# Statement of need

A plethora of image processing problems arising in biology, chemistry and medicine can be defined as Inverse Problems. Inverse Problems aim in recovering a signal $\vec{x} \, \in \, \mathcal{X}$ (e.g. an image) that can not  be directly observed from a set of measurements $\vec{y} \, \in \, \mathcal{Y}$ and is subject to a given corruption process known as the forward model $$\tag{1} \vec{y} \, = \, \mathcal{A}(\vec{x}) \,+\,\vec{n},$$ where $\mathcal{A}$ is the forward operator and $\vec{n}$ is some measurement noise, oftenly assumed to be additive and normally distributed. Equation (1) is usually ill-posed and therefore an explicit solution is hard to find. Instead, Inverse Problems in Imaging are tipically solved by minimizing an objective function $\mathcal{J}$ which is consisted of a data-fidelity term $\mathcal{L}$ and a regularization term $\mathcal{R}$ (also known as Variational Problems):
$$\tag{2}  \vec{\hat{x}} \, = \, \min_{\vec{z} \, \in \, \mathcal{X}} \mathcal{J}(z) \, = \, \min_{\vec{z} \, \in \,  \mathcal{X}} \mathcal{L}\big( \, \vec{y}, \, \mathcal{A}(\vec{z})\big) \,+\, \lambda \mathcal{R}(\vec{z}),\quad \lambda \, \ge \, 0.$$ 

Accelerated Magnetic Ressonance Image (MRI) reconstruction, that is, reconstructing an MR image from a set of partially observed (or undersampled) $k$-space measurements, is par excellence an example of Inverse Problems with a base forward operator the Fourier Transform $\mathcal{F}$.  Conventional approaches of solving this class of Inverse Problems include Parallel Imaging (PI) and Compressed Sensing (CS). Combining these methods with Deep Learning Inverse Problem solvers can aid in providing reconstructed images with high fidelity from highly undersampled measurements. More specifically, given multicoil ($n_c$) undersampled $k$-space measurements $\vec{y} \, = \, \{ \vec{y}_{i=1}^{n_{c}} \} \, = \, \{ U \mathcal{F} ( S_{i} \vec{x} ) \}_{i=1}^{n_{c}}$ as input,  these models aim to predict the reconstructed picture $\vec{x}$. This Inverse problem takes the form:
$$\tag{3}   \vec{\hat{x}} \, = \, \min_{\vec{z} \, \in \,  \mathcal{X}} \sum_{i=1}^{n_{c}} \mathcal{L} \big( \, \vec{y_{i}}, \, U \mathcal{F} ( S_{i} \vec{z} ) \big) \, + \, \lambda \mathcal{R}(\vec{z}),$$
where the $S_{i}$ is a (usually unknown) coil sensitivity map, property of each individual coil and $U$ is a retrospective undersampling mask which simulates the undersampling process in clinical settings. 
As `DIRECT` stores several state-of-the-art [baselines](#baselines-stored), it is an essential tool for any research team working with partially observed $k$-space data.

# Functionality

`DIRECT` allows for easy experimentation. The user can define a configuration file with the `.yaml` extension in which all the training, validation, inference, model, and dataset parameters are specified.


# Baselines Stored

|   Model Name   |                                      Algorithm - Architecture                                      |
|:--------------:|:--------------------------------------------------------------------------------------------------:|
|       RIM      | Recurrent Inference Machine <br>[@beauferris2020multichannel; @LONNING201964]                      |
|     LPDNet     | Learned Primal Dual Network [@lpd2018]                                                             |
| EndToEndVarnet | End-to-end Variational Network [@varnetfastmri]                                                    |
|     XPDNet     | X - Primal Dual Network [@ramzi2021xpdnet]                                                         |
|     KIKINet    | Kspace-Image-Kspace-Image Network [@kiki2018]                                                      |
|   JointICNet   | Joint Deep Model-based MR Image and Coil <br>Sensitivity Reconstruction Network [@Jun_2021_CVPR]   |
| MultiDomainNet | Feature-level multi-domain learning with <br>standardization for multi-channel data [@fastmri2021] |
|     UNet2d     | U-Net for MRI Reconstruction [@zbontar2019fastmri]                                                 |    
 
# Research projects using `DIRECT`

`DIRECT` is the main software used for research by the MRI Reconstruction team of the Innovation Centre for Artificial Intelligence (ICAI) - AI for Oncology group of the Netherlands Cancer Institute (NKI).

Papers fully or partially making use of results output by `DIRECT` include @LONNING201964, @putzky2019irim, @beauferris2020multichannel, @fastmri2021 and @yiasemis2021deep.


# Acknowledgements


# References
