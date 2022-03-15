=======================================================================================================================================
Recurrent Variational Network: A Deep Learning Inverse Problem Solver applied to the task of Accelerated MRI Reconstruction (CVPR 2022)
=======================================================================================================================================

This folder contains the training code specific for our experiments presented in our paper
`Recurrent Variational Network: A Deep Learning Inverse Problem Solver applied to the task of Accelerated MRI Reconstruction <https://arxiv.org/abs/2111.09639>`__ (pre-print version) accepted in CVPR 2022.

.. image::  https://user-images.githubusercontent.com/71031687/158409764-f83df10f-1118-4e9f-9131-2946120c4ff5.png

    Reconstruction of :math:`5\times` data.
    

Datasets
--------
* For the proposed model, the comparison, and ablation studies we used the `Calgary-Campinas public brain multi-coil MRI dataset <https://sites.google.com/view/calgary-campinas-dataset/home>`__ which was released as part of an accelerated MRI reconstruction challenge. The dataset is consisted of 67  3D raw k-space volumes. After cropping the 100 outer slices, these amount to 10,452 slices of fully sampled k-spaces which we randomly split into training (47 volumes), validation (10 volumes) and test (10 volumes) sets (see `lists/ <https://github.com/NKIAI/direct/tree/main/projects/cvpr2022_recurrentvarnet/calgary_campinas/lists>`__). Sub-sampling was performed by applying the Poisson disk distribution sub-sampling masks provided by the challange.

* For additional experiments we used the AXT1 brain `FastMRI dataset <https://fastmri.org/dataset/>`_. The dataset was consisted of 3D raw k-space volumes:
    
  * Training Set: 248 volumes (3844 slices)  
  * Validation Set: 92 volumes (1428 slices) split in half to create new Validation and Test sets.
  For this dataset random Cartesian sub-sampling was performed. 

Training
--------

The standard training command ``direct train`` can be used for training. Configurations can be found in the project folder.

After downloading the data to ``<data_root>`` a command such as the one below is used to train our proposed model on the Calgary Campinas Dataset:

.. code-block:: bash

    direct train <data_root>/Train/ \
                <data_root>/Val/ \
                <output_folder> \
                --cfg /direct/projects/calgary_campinas/configs/base_recurrentvarnet.yaml \
                --num-gpus <number_of_gpus> \
                --num-workers <number_of_workers> \

To train a model used for the comparison or ablation studies in the paper (Section 4) a command such as the one below is used:

.. code-block:: bash

    direct train <data_root>/Train/ \
                <data_root>/Val/ \
                <output_folder> \
                --cfg /direct/projects/calgary_campinas/configs/<ablation_or_comparisons>/base_<model_name>.yaml \
                --num-gpus <number_of_gpus> \
                --num-workers <number_of_workers> \

To train a model used for the additional experiments on the FastMRI AXT1 brain Dataset as in the paper (Appendix B) a command such as the one below is used:

.. code-block:: bash

    direct train <data_root>/Train/ \
                <data_root>/Val/ \
                <output_folder> \
                --cfg /direct/projects/fastmri/AXT1_brain/configs/base_<model_name>.yaml \
                --num-gpus <number_of_gpus> \
                --num-workers <number_of_workers> \

For further information about training see `Training <https://docs.aiforoncology.nl/direct/training.html>`__.

During training, training loss, validation metrics and validation image predictions are logged. Additionally, `Tensorboard <https://docs.aiforoncology.nl/direct/tensorboard.html>`__ allows for visualization of the above.
