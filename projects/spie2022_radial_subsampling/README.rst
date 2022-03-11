===========================================================
Deep MRI Reconstruction with Radial Subsampling (SPIE 2022)
===========================================================

This folder contains the training code specific for our experiments presented in our paper `Deep MRI Reconstruction with Radial Subsampling <https://arxiv.org/abs/2108.07619>`__ accepted in SPIE 2022.

Dataset
-------
We used the `Calgary-Campinas public brain multi-coil MRI dataset <https://sites.google.com/view/calgary-campinas-dataset/home>`__ which was released as part of an accelerated MRI reconstruction challenge.
The dataset is consisted of 67  3D raw k-space volumes collected on a Cartesian grid (equidistant). After cropping the 100 outer slices, these amount to 10,452 slices of fully sampled k-spaces which we randomly
split into training (40 volumes), validation (14 volumes) and test (13 volumes) sets (see `lists/ <https://github.com/NKI-AI/direct/tree/main/projects/spie_radial_subsampling/lists/>`__).

Training
--------

The standard training command ``direct train`` can be used. Training model configurations can be found in the `configs/ <configs>`__ folder.

After downloading the data to ``<data_root>`` a command such as the one below is used to train the `RIM` models:

.. code-block:: bash

    direct train <data_root>/Train/ \
                <data_root>/Val/ \
                <output_folder> \
                --cfg /direct/projects/spie2022_radial_subsampling/configs/base_<radial_OR_rectilinear>.yaml \
                --num-gpus <number_of_gpus> \
                --num-workers <number_of_workers> \
                --resume


For further information about training see `training <../../docs/training.rst>`__.

The validation volumes can be computed using ``predict_val.py`` (see `Validation <#validation>`__).

During training, training loss, validation metrics and validation image predictions are logged. Additionally, `Tensorboard <https://docs.aiforoncology.nl/direct/tensorboard.html>`__ allows for visualization of the above.

Note
~~~~

* For the experiments with Radial Subsampling we used ``base_radial.yaml`` to train the ``RadialRIM``
* For the experiments with Rectilinear Subsampling we used ``base_rectilinear.yaml`` to train the ``RectRIM``


Inference
---------

Validation
~~~~~~~~~~

To make predictions on the validation set (14 volumes, see `lists/val/ <https://github.com/NKI-AI/direct/tree/main/projects/spie_radial_subsampling/lists/val>`__) a command such as the one below is used to perform inference on dataset with index ``<dataset_validation_index>`` as various datasets can be defined in the training configuration file.

.. code-block:: bash

    cd projects/spie_radial_subsampling/
    python3 predict_val.py <data_root>/Val/ \
                    <output_directory> \
                    <experiment_directory_containing_checkpoint> \
                    --checkpoint <checkpoint> \
                    --validation-index <dataset_validation_index> \
                    --num-gpus <number_of_gpus> \
                    --num-workers <number_of_workers> \

Test
~~~~

To make predictions on the test set (13 volumes, `lists/test/ <https://github.com/NKI-AI/direct/tree/main/projects/spie_radial_subsampling/lists/test>`__) a command such as
the one below is used to perform inference on the inference dataset as defined in the ``inference`` section of the ``inference/configs/<acceleration>x/base_<subsampling_type>.yaml`` configuration file.

.. code-block:: bash

    cd projects/spie_radial_subsampling/
    direct predict <data_root>/Test/ \
                    <output_directory> \
                    <experiment_directory_containing_checkpoint> \
                    --cfg configs/inference/<acceleration>x/base_<radial_OR_rectilinear>.yaml
                    --checkpoint <checkpoint> \
                    --num-gpus <number_of_gpus> \
                    --num-workers <number_of_workers> \
