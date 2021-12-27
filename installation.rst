
Installation
============

Requirements
------------


* CUDA 10.2 supported GPU.
* Linux with Python ≥ 3.8
* PyTorch ≥ 1.6

Install using Docker
--------------------

We provide a `Dockerfile <docker>`_ which install DIRECT with a few commands. While recommended due to the use of specific
pytorch features, DIRECT should also work in a virtual environment.

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

      pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

   **otherwise**\ , install the CPU PyTorch installation (not recommended):

   .. code-block::

      pip3 install torch==1.10.0+cpu torchvision==0.11.1+cpu torchaudio==0.10.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

#. 
   Clone the repository using ``git clone`` and navigate to ``direct/direct/`` and run

   .. code-block::

      python3 setup.py install

   This will install ``direct`` as a python module.

Common Installation Issues
--------------------------

If you met issues using DIRECT, please first update the repository to the latest version, and rebuild the docker. When
this does not work, create a GitHub issue so we can see whether this is a bug, or an installation problem.
