==========================
Calgary-Campinas challenge
==========================

This folder contains the training code specific for the `Calgary Campinas challenge <https://sites.google.com/view/calgary-campinas-dataset/mr-reconstruction-challenge>`__.

Training
--------

The standard training command ``direct train`` can be used. Training model configurations can be found in the `configs/ <configs>`__ folder.

After downloading the data to ``<data_root>`` a command such as the one below is used to train a ``<model_name>``:

.. code-block:: bash

   direct train <data_root>/Train/ \
                       <data_root>/Val/ \
                       <output_folder> \
                       --name <name> \
                       --cfg /direct/projects/calgary_campinas/configs/base_<model_name>.yaml \
                       --num-gpus <number_of_gpus> \
                       --num-workers <number_of_workers> \
                       --resume

For further information see `training <../../docs/training.rst>`__.

The validation volumes can be computed using ``predict_val.py`` (see `Validation <#validation>`__).

During training, training loss, validation metrics and validation image predictions are logged. Additionally, `Tensorboard <https://docs.aiforoncology.nl/direct/tensorboard.html>`__ allows for visualization of the above.

For our submissions in the challenge we used `base_rim.yaml <configs/base_rim.yaml>`__ and `base_recurrentvarnet.yaml <configs/base_recurrentvarnet.yaml>`__ as the model configurations. As of writing (January 2022) these are among the top results in both Track 1 and Track 2.

Inference
---------

Validation
~~~~~~~~~~

To make predictions on validation data a command such as the one below is used to perform inference on dataset with index ``<dataset_validation_index>`` as various datasets can be defined in the training configuration file.

.. code-block:: bash

   cd projects/
   python3 predict_val.py <data_root>/Val/ \
                       <output_directory> \
                       <experiment_directory_containing_checkpoint> \
                       --checkpoint <checkpoint> \
                       --validation-index <dataset_validation_index> \
                       --num-gpus <number_of_gpus> \
                       --num-workers <number_of_workers> \

Test
~~~~

The masks are not provided for the test set, and need to be pre-computed using `compute_masks.py <compute_masks.py>`__. These masks should be passed to the ``--masks`` parameter of `predict_test.py <predict_test.py>`__.
