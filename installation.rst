
Installation
============

Requirements
------------


* CUDA ≥ 10.2 supported GPU.
* Linux with Python ≥ 3.8
* PyTorch ≥ 1.6

Install using Docker
--------------------

We provide a `Dockerfile <https://github.com/NKI-AI/direct/tree/main/docker>`_ which install DIRECT with a few commands. While recommended due to the use of specific
pytorch features, DIRECT should also work in a virtual environment.

.. include:: ../docker/README.rst

Install using ``conda``
---------------------------


#.
   First, install conda. Here is a guide on how to install conda on Linux if you don't already have it `here <https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html>`_. If you downloaded conda for the first time it is possible that you will need to restart your machine.  Once you have conda, create a python 3.9 conda environment:

   .. code-block::

      conda create -n myenv python=3.9

   Then, activate the virtual environment ``myenv`` you created where you will install the software:

   .. code-block::

      conda activate myenv

#.
   If you are using GPUs, cuda is required for the project to run. To install `PyTorch <https://pytorch.org/get-started/locally/>`_ with cuda run:

   .. code-block::

      pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

   **otherwise**\ , install the latest PyTorch CPU version (not recommended):

   .. code-block::

      pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

#.
   Clone the repository using ``git clone`` and navigate to ``direct/direct/`` and run

   .. code-block::

      python3 setup.py install

   or

   .. code-block::

      python3 -m pip install -e ".[dev]"

   This will install ``direct`` as a python module.


Using DIRECT with Bazel
----------------------------

DIRECT can also be installed using `bazel <https://bazel.build/>`_. If you want to use bazel, you can follow the 
instructions below.

#.
   Install `bazelisk <https://github.com/bazelbuild/bazelisk>`_ which is a wrapper for bazel that automatically 
   downloads the correct version of bazel for you. You can install it using following the instructions on their
   `GitHub page <https://github.com/bazelbuild/bazelisk>`_.

#.
   Once you have bazelisk installed, you can clone the repository using ``git clone`` and navigate  
   to ``direct/`` and run

   .. code-block::

      bazelisk build //...

   This will build the DIRECT library and create a binary in the `bazel-bin` directory.

#.
   Make sure the tests are passing by running:
   .. code-block::

      bazelisk test //...

#.
   To use DIRECT commands, you follow the normal run commands (e.g., `training <./docs/training.rst>`_ or
   `inference <./docs/inference.rst>`_), but with including the `bazelisk` command. For example,
   to run the `training` command, you can use:

   .. code-block::

      bazelisk run //direct:direct -- train <experiment_directory> --num-gpus <number_of_gpus> \
      --cfg <path_or_url_to_yaml_file> [--training-root <training_data_root> \
      --validation-root <validation_data_root>]  [--other-flags]

Common Installation Issues
--------------------------

If you met issues using DIRECT, please first update the repository to the latest version, and rebuild the docker. When
this does not work, create a GitHub issue so we can see whether this is a bug, or an installation problem.
