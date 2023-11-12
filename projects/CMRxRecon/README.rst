Deep Cardiac MRI Reconstruction with ADMM (CMRxRecon Challenge 2023)
=====================================================================

.. figure:: https://github.com/NKI-AI/direct/assets/71031687/40460397-acb0-402e-bd22-0e7b547e61e5
   :alt: fig
   :name: fig1

   Figure 1: Pipeline of our proposed methods.

.. figure:: https://github.com/NKI-AI/direct/assets/71031687/24e68d2a-4d94-42b3-9560-f68661753ad9
   :alt: tabs
   :name: fig2

   Figure 2: Average results and inference times.


This project contains necessary configuration files to reproduce experiments of 2nd top-ranking approach
to both tasks of CMRxRecon Challenge 2023 as presented in `Deep Cardiac MRI Reconstruction with ADMM
<https://arxiv.org/abs/2310.06628>`_.
This will also help you set up the training and inference data directories.

Setting up data directory
-------------------------

This project aims to help you set up directories for training and validation
data using the specified directory structures necessary to run with DIRECT.
You will need to run this process twice: once for "Cine" data and once for "Mapping" data.

Prerequisites
~~~~~~~~~~~~~

Before you begin, make sure you have downloaded the CMRxRecon Challenge
data. Check `https://cmrxrecon.github.io/Challenge.html`_ for more information.

Assumed Base Path Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The script assumes that CMRxRecon data is organized according to the following directory structure:

.. code-block:: plaintext

    base_path
    ├── MultiCoil
    │   ├── Cine_or_Mapping
    │   │   ├── TrainingSet
    │   │   │   ├── FullSample
    │   │   │   ├── AccFactor04
    │   │   │   ├── AccFactor08
    │   │   │   └── AccFactor10
    │   │   ├── ValidationSet
    │   │   │   ├── AccFactor04
    │   │   │   ├── AccFactor08
    │   │   │   └── AccFactor10
    │   │   ├── TestSet
    │   │   │   ├── AccFactor04
    │   │   │   ├── AccFactor08
    │   │   │   └── AccFactor10

Symlinks Path Structure
~~~~~~~~~~~~~~~~~~~~~~~

The script will create symbolic links (symlinks) in a target directory with the following structure:

.. code-block:: plaintext

    target_path
        target_path
        ├── MultiCoil
        │   ├── training
        │   │   ├── P001_T1map.h5
        │   │   ├── with_masks_P001_T1map.h5
        │   │   ├── P001_cine_sax.h5
        │   │   ├── with_masks_P001_cine_sax.h5
        │   ├── Cine_or_Mapping
        │   │   ├── validation
        │   │   │   ├── AccFactor04
        │   │   │   |   ├── P001_<..>.h5
        │   │   │   ├── AccFactor08
        │   │   │   |   ├── P001_<..>.h5
        │   │   │   └── AccFactor10
        │   │   │   |   ├── P001_<..>.h5
        │   │   ├── test
        │   │   │   ├── AccFactor04
        │   │   │   |   ├── P001_<..>.h5
        │   │   │   ├── AccFactor08
        │   │   │   |   ├── P001_<..>.h5
        │   │   │   └── AccFactor10
        │   │   │   |   ├── P001_<..>.h5

Create Symbolic Directories
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following creates files of fully sampled data with the respective masks, and creates
symbolic paths of data in single directories to be used with DIRECT.

.. code-block:: bash

    python3 create_data_dir.py --base_path path_to_base_data --target_path path_to_target_directory --data_type Cine
    python3 create_data_dir.py --base_path path_to_base_data --target_path path_to_target_directory --data_type Mapping

Experiments
-----------

Configuration Files
~~~~~~~~~~~~~~~~~~~

We provide configuration files for DIRECT for experiments presented in `Deep Cardiac MRI Reconstruction with ADMM
<https://arxiv.org/abs/2310.06628>`_ in the `CMRxRecon configs folder <https://github.com/NKI-AI/direct/tree/main/projects/CMRxRecon>`_.

Training
~~~~~~~~

In `direct/` run the following command to begin training on the training data.

.. code-block:: bash

    direct train <output_folder> \
                --training-root <target_path>/MultiCoil/training/ \
                --validation-root <target_path>/MultiCoil/training/ \
                --cfg projects/CMRxRecon/configs/base_<name_of_experiment>.yaml \
                --num-gpus <number_of_gpus> \
                --num-workers <number_of_workers> --resume

Note that for validation a subset of the training data is used since full validation data have not been released.

Inference
~~~~~~~~~

Note that inference is performed for a single dataset, therefore a single acceleration factor.
For example, the following entry for `inference` will perform predictions for acceleration factor of 4x
on validation data. Change `kspace_key: kspace_sub04` to `kspace_key: kspace_sub08` for 8x and
`kspace_key: kspace_sub10` for 10x.

.. code-block:: yaml

    inference:
        batch_size: 8
        dataset:
            name: CMRxRecon
            kspace_key: kspace_sub04
            compute_mask: true
            transforms:
                cropping:
                    crop: null
                sensitivity_map_estimation:
                    estimate_sensitivity_maps: true
                normalization:
                    scaling_key: masked_kspace
                    scale_percentile: 0.99
                masking: null
            text_description: inference-4x
        crop: null

In `direct/` run the following command to perform inference, for instance on 4x:

.. code-block:: bash

    direct predict <output_directory>
                --checkpoint <path_or_url_to_checkpoint> \
                --cfg projects/CMRxRecon/configs/base_<name_of_experiment>.yaml \
                --data-root <target_path>/MultiCoil/<Cine_or_Mapping>/validation/AccFactor<04_or_08_or_10> \
                --num-gpus <number_of_gpus> \
                --num-workers <number_of_workers> \
                [--other-flags]

Note
~~~~

Fully sampled validation dataset and Test data have note be released yet by the CMRxRecon team.


Citing this work
----------------

Please use the following BiBTeX entries if you use our proposed methods in your work:

.. code-block:: BibTeX

    @article{yiasemis2023deep,
        title = {Deep Cardiac MRI Reconstruction with ADMM},
        author = {George Yiasemis and Nikita Moriakov and Jan-Jakob Sonke and Jonas Teuwen},
        year = {2023},
        month = {Oct},
        eprint = {2310.06628},
        archivePrefix = {arXiv},
        journal = {arXiv.org},
        doi = {10.48550/arXiv.2310.06628},
        url = {https://doi.org/10.48550/arXiv.2310.06628},
        note = {arXiv:2310.06628 [eess.IV]},
        primaryClass = {eess.IV}
    }

    @article{DIRECTTOOLKIT,
        doi = {10.21105/joss.04278},
        url = {https://doi.org/10.21105/joss.04278},
        year = {2022},
        publisher = {The Open Journal},
        volume = {7},
        number = {73},
        pages = {4278},
        author = {George Yiasemis and Nikita Moriakov and Dimitrios Karkalousos and Matthan Caan and Jonas Teuwen},
        title = {DIRECT: Deep Image REConstruction Toolkit}, journal = {Journal of Open Source Software}
    }
