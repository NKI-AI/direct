Training
========

Experiment Configuration
------------------------

In ```code/experiments/configs/`` <../experiments/configs>`__ are stored
all the configuration files (``.yaml`` extension) for our experiments as
presented in the paper. Each configuration file contains all training
details for each experiment including: \* Model hyper-parameters \*
Training parameters \* Dataset parameters \* etc.

The configuration file we used for our original model is
`base_recurrentvarnet.yaml <../experiments/configs/base_recurrentvarnet.yaml>`__.
Configuration files for the comparison and ablation studies are located
in `comparisons <../experiments/configs/comparisons>`__ and
`ablation <../experiments/configs/ablation>`__ respectively.

Run Experiments
---------------

After `installing <./install.md>`__ the software and downloading the
`dataset <./dataset_download.md>`__, to train a model navigate to the
base code directory ``code/``. Assuming that the data is stored in
``<data_root>``, you can run the following to train a model with name
``<experiment_name>``:

::

   python3 tools/train_model.py <data_root>/Train/ \
               <data_root>/Val/ \ 
               <experiment_directory> \ 
               --cfg experiments/configs/base_<experiment_name>.yaml \
               --num-gpus <number_of_gpus>

The above command will start the training and will create an experiment
directory in ``<experiment_directory>/base_<experiment_name>``. If you
are performing an experiment on a CPU (not recommended) replace
``--num-gpus <number_of_gpus>`` with ``--device 'cpu:0'``.

In ``<experiment_directory>/base_<experiment_name>`` there will be
stored the logs of the experiment, model checkpoints
(e.g. ``model_<checkpoint_number>.pt``), training and validation
metrics, and a ``config.yaml`` file which includes all the configuration
parameters of the experiment (as stated in
``base_<experiment_name>.yaml``). For `inference <./inference.md>`__,
this is the directory you should use for
``<experiment_directory_containing_checkpoint_and_config>``.

Tensorboard
-----------

To visualize training and validation metrics of an experiment in
Tensorboard on your local machine run

::

   tensorboard --logdir <path_to_experiment> --port <port_id> . 

If you are working on a remote host and want to visualize the experiment
on your local machine run

1. ``tensorboard --logdir <path_to_experiment> --port <remote_port_id>``
   on the remote host, and

2. ``ssh -N -f -L localhost:<local_port_id>:localhost:<remote_port_id> <user@remote_host>``
   on your local machine.

3. Navigate to `http://localhost: <http://localhost:local_port_id>`__ on
   your local machine.
