=======================================================================================================================================
Recurrent Variational Network: A Deep Learning Inverse Problem Solver applied to the task of Accelerated MRI Reconstruction (CVPR 2022)
=======================================================================================================================================

This folder contains the training code specific for our experiments presented in our paper
`Recurrent Variational Network: A Deep Learning Inverse Problem Solver applied to the task of Accelerated MRI Reconstruction <https://arxiv.org/abs/2111.09639>`__ (pre-print version) accepted in CVPR 2022.

Dataset
-------
* For the comparison and ablation studies we used the `Calgary-Campinas public brain multi-coil MRI dataset <https://sites.google.com/view/calgary-campinas-dataset/home>`__
which was released as part of an accelerated MRI reconstruction challenge.
The dataset is consisted of 67  3D raw k-space volumes collected on a Cartesian grid (equidistant). After cropping the 100 outer slices, these amount to 10,452 slices of
fully sampled k-spaces which we randomly
split into training (47 volumes), validation (10 volumes) and test (10 volumes) sets (see `lists/ <https://github.com/NKIAI/direct/tree/main/projects/cvpr2022_recurrentvarnet/calgary_campinas/lists>`__).

* For additional

Training
--------

The standard training command ``direct train`` can be used. Training model configurations can be found in the `configs/ <configs>`__ folder.

After downloading the data to ``<data_root>`` a command such as the one below is used to train our proposed model:

.. code-block:: bash

    direct train <data_root>/Train/ \
                <data_root>/Val/ \
                <output_folder> \
                --cfg /direct/projects/calgary_campinas/configs/base_recurrentvarnet.yaml \
                --num-gpus <number_of_gpus> \
                --num-workers <number_of_workers> \
                --resume

To train a model used for the comparison or ablation studies in the paper (Section 4) a command such as the one below is used:

.. code-block:: bash

    direct train <data_root>/Train/ \
                <data_root>/Val/ \
                <output_folder> \
                --cfg /direct/projects/calgary_campinas/configs/<ablation_or_comparisons>/base_<model_name>.yaml \
                --num-gpus <number_of_gpus> \
                --num-workers <number_of_workers> \
                --resume


For further information about training see `training <../../docs/training.rst>`__.

During training, training loss, validation metrics and validation image predictions are logged. Additionally, `Tensorboard <https://docs.aiforoncology.nl/direct/tensorboard.html>`__ allows for visualization of the above.
