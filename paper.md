---
title: "DIRECT: Deep Image REConstruction toolkit"

tags:

- Python
- Pytorch
- Deep Learning
- Inverse Problem Solver
- Image Processing
- Deep MRI reconstruction
- Accelerated MRI

authors:

- name: George Yiasemis^[first author] 
  orchid: 0000-0002-1348-8987
  affiliation: "1, 2"
- name: Nikita Moriakov^[co-author]
  affiliation: "1, 4"
- name: Dimitrios Karkalousos^[co-author]
  affiliation: 3
- name: Matthan Caan^[co-author]
  affiliation: 3
- name: Jonas Teuwen^[supervisor]
  affiliation: "1, 2, 4"
   
affiliations:

- name: Netherlands Cancer Institute
  index: 1
- name: University of Amsterdam
  index: 
- name: Amsterdam UMC, Biomedical Engineering and Physics
  index: 3
- name: Radboud University Medical Center
  index: 4


date: 30 October 2021

bibliography: paper.bib

# Summary
`DIRECT` is a Python, end-to-end pipeline for solving Inverse Problems emerging in Imaging Processing. It is built with PyTorch and stores state-of-the-art Deep Learning imaging inverse problem solvers such as denoising, dealiasing and reconstruction. By defining a forward linear or non-linear operator, `DIRECT` can be used for training models for recovering images such as MRIs from partially observed or noisy input data. Additionally, it provides the user with the functionality to load saved weights of pre-trained models to be used for inference. Furthermore, it offers functions for peparing and pre-processing data such as `.h5` files into PyTorch Datasets compatible with the software's training pipeline but also allows for flexibility to work with any kind of PyTorch Dataset. In order for the user to view the proccess of their experiments, it allows for continuous visualisation of training and validation metrics as well as image predictions utilising Tensorboard.

# Statement of need
A plethora of image processing problems arising in biology, chemistry and medicine can be defined as Inverse Problems. Inverse Problems aim in recovering a signal $\vec{x} \, \in \, \mathcal{X}$ (e.g. an image) that can not  be directly observed from a set of measurements $\vec{y} \, \in \, \mathcal{Y}$ and is subject to a given corruption process known as the forward model $$\tag{1} \vec{y} \, = \, \mathcal{A}(\vec{x}) \,+\,\vec{n},$$ where $\mathcal{A}$ is the forward operator and $\vec{n}$ is some noise, oftenly assumed to be additive Gaussian noise. Equation (1) is usually ill-posed and therefore an explicit solution is hard to find. Instead, Inverse Problems in Imaging are tipically solved by minimizing an objective function $\mathcal{J}$ which is consisted of a data-fidelity term $\mathcal{L}$ and a regularization term $\mathcal{R}$ (also known as Variational Problems):
$$\tag{2}  \min_{\vec{x} \, \in \, \mathcal{X}} \mathcal{J}(x) \, = \, \min_{\vec{x} \, \in \,  \mathcal{X}} \mathcal{L}\big(\vec{x}, \, \mathcal{A}(\vec{y})\big) \,+\, \lambda \mathcal{R}(\vec{x}),\quad \lambda \, \ge \, 0.$$ 

Accelerated Magnetic Ressonance Image (MRI) reconstruction, that is, reconstructing an MR image from a set of partially observed (or undersampled) $k$-space measurements, is par excellence an example of Inverse Problems.  Conventional approaches of solving this class of Inverse Problems include Parallel Imaging (PI) and Compressed Sensing (CS). Combining these methods with Deep Learning Inverse Problem solvers can aid in providing reconstructed images with high fidelity from highly undersampled measurements. More specifically, given multicoil ($n_c$) undersampled $k$-space measurements $\vec{y} \,=\,\{\vec{y}_{i=1}^{n_c}\}$ as input,  these models aim to predict the reconstructed picture $\vec{x}$. As `DIRECT` stores several state-of-the-art [baselines](#baselines-stored), it is an essential tool for any research team working with partially observed $k$-space data.

# Functionality

# Baselines Stored

 - Recurrent Inference Machine (RIM) [@beauferris2020multichannel; @LONNING201964]
 - End-to-end Variational Network (EndToEndVarNet) [@varnetfastmri]
 - Learned Primal Dual Network (LDPNet) [@lpd2018]
 - X-Primal Dual Network (XPDNet) [@ramzi2021xpdnet]
 - KIKI-Net [@kiki2018]
 - U-Net [@zbontar2019fastmri]
 - Joint-ICNet [@Jun_2021_CVPR]
 - AIRS Medical fastmri model (MultiDomainNet) [@fastmri2021]

# Research projects using `DIRECT`

`DIRECT` is the main software used for research by the MRI Reconstruction team of the Innovation Centre for Artificial Intelligence (ICAI) - AI for Oncology group of the Netherlands Cancer Institute (NKI).

Papers fully or partially making use of results output by `DIRECT` include @LONNING201964, @putzky2019irim, @fastmri2021, @beauferris2020multichannel, @yiasemis2021deep .


# Acknowledgements


# References
