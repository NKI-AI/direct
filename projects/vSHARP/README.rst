===============================================================================================
vSHARP: variable Splitting Half-quadratic ADMM algorithm for Reconstruction of inverse-Problems
===============================================================================================

This folder contains the training code specific for reproduction of our experiments as presented in our paper
`vSHARP: variable Splitting Half-quadratic ADMM algorithm for Reconstruction of inverse-Problems (pre-print) <https://arxiv.org/abs/2309.09954>`__.

.. figure:: https://github.com/NKI-AI/direct/assets/71031687/493701b6-6efa-427d-9b4f-94a0ebcf3142
   :alt: fig
   :name: fig1

   Figure 1: Overview of our proposed method vSHARP.

Dataset
=======
* For the proposed model, the comparison, and ablation studies we used the `fastMRI prostate T2 dataset <https://arxiv.org/abs/2304.09254>`__.
To constract the training, validation and test data we used code provided in https://github.com/cai2r/fastMRI_prostate
from the raw ismrmd data format.

* We employed a retrospective Cartesian equispaced scheme to undersample our data.

Training
========

Assuming data are stored in ``data_root` the standard training command ``direct train`` can be used for training.

Our model and baselines configuration files can be found in the
`vSHARP project folder <https://github.com/NKI-AI/direct/tree/main/projects/vSHARP>`_.

To train vSHARP or the any of the baselines presented in the paper use the following command:

.. code-block:: bash

    direct train <output_folder> \
                --training-root /.../data_root/<training_data_directory> \
                --validation-root /.../data_root/<validation_data_directory>  \
                --cfg projects/vSHARP/fastmri_prostate/base_<name_of_model>.yaml \
                --num-gpus <number_of_gpus> \
                --num-workers <number_of_workers> \


For further information about training see `Training <https://docs.aiforoncology.nl/direct/training.html>`__.

During training, training loss, validation metrics and validation image predictions are logged.
Additionally, `Tensorboard <https://docs.aiforoncology.nl/direct/tensorboard.html>`__ allows for visualization of the above.

Inference
=========

To perform inference on test set run:

.. code-block:: bash

    direct predict <output_directory> \
                --checkpoint <path_or_url_to_checkpoint> \
                --cfg projects/vSHARP/fastmri_prostate/base_<name_of_model>.yaml \
                --data-root /.../data_root/<validation_data_directory> \
                --num-gpus <number_of_gpus> \
                --num-workers <number_of_workers> \
                [--other-flags]
