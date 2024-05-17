JSSL: Joint Supervised and Self-supervised Learning for MRI Reconstruction
==========================================================================

This project contains necessary configuration files to reproduce experiments of the paper
`JSSL: Joint Supervised and Self-supervised Learning for MRI Reconstruction <https://arxiv.org/abs/2311.15856>`_.


.. figure:: https://github.com/NKI-AI/direct/assets/71031687/97f1fa27-f4e4-44e2-b54a-7ac149f3c01d
   :alt: fig
   :name: fig1

   Figure 1: The training process for the proposed Joint Supervised and Self-supervised Learning (JSSL) method is 
   divided into two phases: (1) Supervised Learning (SL) using fully sampled k-space data from proxy datasets. 
   During this phase, the model is trained to predict fully sampled data from retrospectively subsampled proxy data. 
   (2) Self-supervised Learning.


Setting up data directory
-------------------------

Prerequisites
~~~~~~~~~~~~~

Before you begin, make sure you have downloaded the all three fastMRI datasets - brain, knee and prostate. 
For more information for downloading you can check out the `fastMRI website <https://fastmri.med.nyu.edu/>`_.

Prostate Dataset Setup
~~~~~~~~~~~~~~~~~~~~~~

In this project, we use the `prostate dataset <https://www.nature.com/articles/s41597-024-03252-w>`_. from the fastMRI 
as the target dataset. After downloading the dataset, you need to convert the data to necessary format. 
Raw T2 prostate k-space data are provided with dimensions (averages, slices, coils, readout, phase). It is required to
convert the data to h5 format with dimensions (slices, coils, readout, phase) to match proxy datasets (brain and knee).
To do that you can use the GRAPPA T2 reconstruction code provided in the fastMRI Prostate `repository 
<https://github.com/cai2r/fastMRI_prostate/blob/main/fastmri_prostate/reconstruction/t2/prostate_t2_recon.py>`_.


Assumed Base Path Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To run the JSSL experiments with the provided configuration files in direct it's recommended that all data are stored
in the same directory or symlinked to the same directory. The configuration files assume that the data is stored in the
following structure:

.. code-block:: text

   base_path
   ├── training
   │   ├── file_brain_AXT1_<..>.h5
   │   ├── file_brain_AXFLAIR_<..>.h5
   │   ├── ...
   │   ├── file100<...>.h5 (knee)
   │   ├── file101<...>.h5 (knee)
   │   ├── ...
   │   ├── file_prostate_AXT2_<...>.h5
   │   └──  ...
   ├── validation
   │   ├── file_prostate_AXT2_<...>.h5
   │   └──  ...
   ├── test
   │   ├── file_prostate_AXT2_<...>.h5
   |   └──  ...


The `base_path` is the root directory where all the data is stored. The `base_path` should contain
three subdirectories: `training`, `validation` and `test`. The `training` directory should contain all the training data
for the three datasets. The `validation` and `test` directories should contain the validation and test data for the
prostate dataset. The filenames should be the same as the ones in the fastMRI dataset.


Filenames lists
~~~~~~~~~~~~~~~
Provided configuration files assume that `direct/projects/JSSL/lists/` contains `.lst` files for each dataset
which contains the list of filenames for training, validation and test datasets. Each list categorizes the filenames
based on the dataset they belong to, as well as shape and number of coils. This is necessary for the dataloader to
collate the data for batch size greater than 1. The filenames should be the same as the ones in the data directory. 

Experiments
-----------

Configuration Files
~~~~~~~~~~~~~~~~~~~

We provide configuration files for JSSL, SSL and all ablation settings as presented in the main paper, 
as well as the supplementary material. The configuration files are stored
in the `JSSL projects folder <https://github.com/NKI-AI/direct/tree/main/projects/JSSL>`_.


Training
~~~~~~~~

In `direct/` run the following command to commence training:

.. code-block:: bash

    direct train <output_folder> \
      --training-root <base_path>/training/ \
      --validation-root <base_path>/validation/ \
      --cfg projects/JSSL/configs/<name_of_experiment>.yaml \
      --num-gpus <number_of_gpus> \
      --num-workers <number_of_workers> --resume


For instance, to perform JSSL training with the vSHARP model, replace `<name_of_experiment>.yaml`
with `vsharp_jssl.yaml`.

Note
~~~~
Note that `jssl`, `sl` and `ssl` in the configuration file names stand for Joint Supervised and Self-supervised 
Learning, Supervised Learning, and Self-supervised Learning, respectively.