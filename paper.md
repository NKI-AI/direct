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

DIRECT is a Python, end-to-end pipeline for solving Inverse Problems emerging in Imaging Processing. It is built with PyTorch [@NEURIPS2019_9015] and stores state-of-the-art Deep Learning imaging inverse problem solvers for solving inverse problems such as denoising, dealiasing, and reconstruction. By defining a base forward linear or non-linear operator, DIRECT can be used for training models for recovering images such as MRIs from partially observed or noisy input data. Additionally, it provides the user with the functionality to load saved weights of pre-trained models to be used for inference. Furthermore, it offers functions for preparing and pre-processing data such as `.h5` files into PyTorch Datasets compatible with the software's training pipeline, but also allows for flexibility to work with any kind of PyTorch Dataset. Additionally, in order for the user to view the process of their experiments, it allows for continuous visualisation of training and validation metrics as well as image predictions utilising Tensorboard (examples are illustrated in Figures 1 and 2). 

| ![image](https://user-images.githubusercontent.com/71031687/138093195-67004ec7-6bfd-448b-ba53-4cdd291a471b.png) |
|:--:|
| <b> Figure 1:  Visualised reconstructions in Tensorboard <b> |

| ![image](https://user-images.githubusercontent.com/71031687/138097866-221aebb5-9aa3-4b8b-8a95-c0541ae52bb1.png) |
|:--:|
| <b> Figure 2:  Visualised metrics in Tensorboard <b> |   
 
# Statement of need

A plethora of image processing problems arising in biology, chemistry and medicine can be defined as inverse problems. Inverse problems aim in recovering a signal $\vec{x} \, \in \, \mathcal{X}$ (e.g. an image) that cannot be directly observed from a set of measurements $\vec{y} \, \in \, \mathcal{Y}$ and is subject to a given corruption process known as the forward model:
    
\begin{equation}
    \vec{y} \, = \, \mathcal{A}(\vec{x}) \,+\,\vec{n},
    \label{eq:eq1}
\end{equation}
    
where $\mathcal{A}$ denotes the forward operator and $\vec{n}$ is some measurement noise, often assumed to be additive and normally distributed. Equation \ref{eq:eq1} is usually ill-posed and therefore an explicit solution is hard to find. Instead, inverse problems in imaging are typically solved by minimizing an objective function $\mathcal{J}$ which is consisted of a data-fidelity term and a regularization term $\mathcal{R}$ (also known as Variational Problems):
    
\begin{equation}
    \vec{\hat{x}} \, = \, \min_{\vec{z} \, \in \, \mathcal{X}} \mathcal{J}(z) \, = \, \min_{\vec{z} \, \in \,  \mathcal{X}} \frac{1}{2}\big|\big| \, \vec{y}\,- \, \mathcal{A}(\vec{z})\big|\big|_2^2 \,+\, \lambda \mathcal{R}(\vec{z}),\quad \lambda \, \ge \, 0.
    \label{eq:eq2}
\end{equation}

## Accelerated Parallel MRI Reconstruction

Accelerated Parallel Magnetic Resonance Image (MRI) Reconstruction, that is, reconstructing an MR image from a set of partially observed (or sub-sampled) $k$-space measurements from multiple receiver coils (Parallel Imaging [@Larkman_2007]), is par excellence an example of inverse problems. The base forward operator of Accelerated MRI Reconstruction is usually the two or three-dimensional Fast Fourier Transform (FFT) denoted as $\mathcal{F}$.  
     
More specifically, given sub-sampled $k$-space measurements acquired from $n_c$ receiver coils, where

\begin{equation}
    \vec{y} \, = \, \big\{ \vec{y}_1, \, ...,\, \vec{y}_{n_c} \big\}, \quad \vec{y}_i  \, = \, U \circ \mathcal{F} \big( S_{i} \vec{x} \big), \quad i=1,...,n_{c},
\end{equation}

where $S_{i}$ denotes a (usually unknown or estimated) coil sensitivity map, property of each individual coil, and $U$ a retrospective binary sub-sampling mask operator which simulates the sub-sampling process in clinical settings,
 
the corresponding inverse problem of Accelerated Parallel MRI Reconstruction replaces \eqref{eq:eq2} with the following:
    
\begin{equation}
    \vec{\hat{x}} \, = \, \min_{\vec{z} \, \in \,  \mathcal{X}} \sum_{i=1}^{n_{c}} \frac{1}{2}\big|\big| \, \vec{y_{i}}\,- \, U \circ \mathcal{F} ( S_{i} \vec{z} ) \big|\big|_2^2 \, + \, \lambda \mathcal{R}(\vec{z}).
    \label{eq:eq3}
\end{equation}  

    
Conventional approaches for solving this class of inverse problems include Compressed Sensing (CS) [@1614066; @1580791; @Lustig2007], SENSE [@Pruessmann1999], and GRAPPA [@Griswold2002]. Deep Learning-based imaging inverse problem solvers have shown to outperform these conventional techniques by outputting reconstructed images with higher fidelity from highly sub-sampled measurements [@Knoll2020; @arxiv.2109.08618; @LONNING201964]. 
    
As DIRECT stores several state-of-the-art [baselines](#baselines-stored), it is an essential tool for any research team working with partially observed $k$-space data.

# Functionality

DIRECT allows for easy and flexible experimentation. The user can define a configuration file with the `.yaml` extension to perform any experiments. See [Configuration File](#configuration-file) below for an example of a configuration file. DIRECT can be employed for training and/or validating models on multiple machines and GPUs as it is integrated with PyTorch's `torch.distributed` module and NVIDIA's cuDNN [@chetlur2014cudnn]. Besides the already-stored baselines, the user can easily incorporate into DIRECT their own inverse problem solvers.

## Configuration File
    
In a configuration file it should be specified all the experiment parameters including model parameters, physics parameters, training and validation parameters, dataset parameters, etc. The following is a template example of a configuration file:

```yaml
model:
  model_name: <nn_model_path>
  model_parameter_1: <nn_model_paramter_1>
  model_parameter_2: <nn_model_paramter_2>
  ...
additional_models:
  sensitivity_model:
    model_name: <nn_sensitivity_model_path>
    ...
physics:
  forward_operator: fft2(centered=<true_or_false>)
  backward_operator: ifft2(centered=<true_or_false>)
  ...
training:
  datasets:
  - name: Dataset1
    lists:
    - <path_to_list_1_for_Dataset1>
    - <path_to_list_2_for_Dataset1>
    transforms:
      estimate_sensitivity_maps: <true_or_false>
      scaling_key: <scaling_key>
      image_center_crop: <true_or_false>
      masking:
        name: MaskingFunctionName
        accelerations: [acceleration_1, accelaration_2, ...]
        ...
    ...
  - name: Dataset2
    ...
  optimizer: <optimizer>
  lr: <learning_rate>
  batch_size: <batch_size>
  lr_step_size: <lr_step_size>
  lr_gamma: <lr_gamma>
  lr_warmup_iter: <num_warmup_iterations>
  num_iterations: <num_iterations>
  validation_steps: <num_val_steps>
  loss:
    losses:
    - function: <fun1_as_in_model_engine>
      multiplier: <multiplier_1>
    - function: <fun2_as_in_model_engine>
      multiplier: <multiplier_2>
  checkpointer:
    checkpoint_steps: <num_checkpointer_steps>
  metrics: [<metric_1>, <metric_2>, ...]
  ...
validation:
  datasets:
  - name: ValDataset1
    transforms:
      ...
      masking:
        ...
    text_description: <val_description_1>
    ...
  - name: ValDataset2
    ...
  batch_size: <val_batch_size>
  metrics:
  - val_metric_1
  - val_metric_2
  - ...
  ...
inference:
  dataset:
    name: InferenceDataset
    lists: ...
    transforms:
      masking:
        ...
      ...
    text_description: <inference_description>
    ...
  batch_size: <batch_size>
  ...
logging:
  tensorboard:
  num_images: <num_images>
```
    
# Baselines Stored

|   Model Name   |                                      Algorithm - Architecture                                      |
|:--------------:|:--------------------------------------------------------------------------------------------------:|
|RecurrentVarNet | Recurrent Variational Network [@yiasemis2021recurrent]                                         |
|       RIM      | Recurrent Inference Machine <br>[@beauferris2020multichannel; @LONNING201964]                      |
|     LPDNet     | Learned Primal Dual Network [@lpd2018]                                                             |
| EndToEndVarnet | End-to-end Variational Network [@varnetfastmri]                                                    |
|     XPDNet     | X - Primal Dual Network [@ramzi2021xpdnet]                                                         |
|     KIKINet    | Kspace-Image-Kspace-Image Network [@kiki2018]                                                      |
|   JointICNet   | Joint Deep Model-based MR Image and Coil <br>Sensitivity Reconstruction Network [@Jun_2021_CVPR]   |
| MultiDomainNet | Feature-level multi-domain learning with <br>standardization for multi-channel data [@fastmri2021] |
|     UNet2d     | U-Net for MRI Reconstruction [@zbontar2019fastmri]                                                 |    
 
# Research projects using DIRECT

DIRECT is the main software used for research by the MRI Reconstruction team of the Innovation Centre for Artificial Intelligence (ICAI) - AI for Oncology group of the Netherlands Cancer Institute (NKI). 

## Challenges
DIRECT has been used for MRI Reconstruction result submissions in the fastMRI challenge  [@fastmri2021] and the Multi-Coil MRI Reconstruction challenge [@beauferris2020multichannel]. 
    
## Publications
Papers using DIRECT:

* @yiasemis2021deep (presented in SPIE Medical Imaging Conference 2022)
* @yiasemis2021recurrent (to be presented in CVPR Conference 2022)


# References
