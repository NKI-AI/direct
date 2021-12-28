.. highlight:: shell

========
Training
========

After `installing <../installation.rst>`_ the software and downloading the training and validation data  to `<data_root>`, to train a model you can run the following to train a model you can use the ``direct train`` command.

To train on a single machine run the following code block in your linux machine:

.. code-block:: bash

    direct train <data_root>/Train/ <data_root>/Val/ <experiment_directory> --num-gpus <number_of_gpus> --cfg <path_or_url_to_yaml_file> [--other-flags]
                  
To train on multiple machines run the following code (one command on each machine):

.. code-block:: bash

    (machine0)$ direct train <data_root>/Train/ <data_root>/Val/ <experiment_directory> --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
    (machine1)$ direct train <data_root>/Train/ <data_root>/Val/ <experiment_directory> --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]


The above command will start the training and will create an experiment directory in ``<experiment_directory>/base_<experiment_name>``. If you are performing an experiment on a CPU (not recommended) replace ``--num-gpus <number_of_gpus>`` with ``--device 'cpu:0'``.

In ``<experiment_directory>/base_<experiment_name>`` there will be stored the logs of the experiment, model checkpoints (e.g. ``model_<checkpoint_number>.pt``), training and validation metrics, and a ``config.yaml`` file which includes all the configuration parameters of the experiment (as stated in the ``yaml`` file ``<path_or_url_to_yaml_file>``). 


Training model configurations can be found in the ``projects`` folder.  

During training, training loss, validation metrics and validation image predictions are logged. Additionally, [Tensorboard](#tensorboard) allows for visualization of the above. 
