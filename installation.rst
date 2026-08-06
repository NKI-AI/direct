
Installation
============

Requirements
------------


* CUDA ≥ 12.6 supported GPU (optional, but recommended for training).
* Linux or macOS with Python ≥ 3.12.
* PyTorch ≥ 2.11.
* A working C++20 compiler and CMake ≥ 3.18 (required to build the bundled
  ``scikit-build-core`` / ``nanobind`` C++ extension).

Install using ``uv`` (recommended)
----------------------------------

`uv <https://docs.astral.sh/uv/>`_ is the recommended way to install ``DIRECT``
for both users and developers. It produces fast, reproducible environments
directly from ``pyproject.toml`` and the committed ``uv.lock``.

#.
   Install ``uv`` following the
   `official instructions <https://docs.astral.sh/uv/getting-started/installation/>`_.

#.
   Clone the repository and synchronise the environment:

   .. code-block::

      git clone https://github.com/NKI-AI/direct.git
      cd direct
      uv sync                 # runtime + dev (default groups)
      uv sync --all-groups    # also install docs tooling

   ``uv`` will provision Python 3.12, create ``.venv/``, build the C++
   extension via ``scikit-build-core`` + ``nanobind``, and install all pinned
   dependencies from ``uv.lock``.

#.
   Either activate the environment or prefix commands with ``uv run``:

   .. code-block::

      source .venv/bin/activate
      direct --help

   .. code-block::

      uv run direct --help
      uv run pytest

Install using Docker
--------------------

We provide a `Dockerfile <https://github.com/NKI-AI/direct/tree/main/docker>`_
which installs ``DIRECT`` with a few commands. Recommended when you need a
fully reproducible CUDA stack.

.. include:: ../docker/README.rst

Install using ``conda`` (alternative)
-------------------------------------


#.
   Install conda. Here is a guide on how to install conda on Linux if you don't
   already have it `here <https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html>`_.
   Once you have conda, create a Python 3.12 conda environment:

   .. code-block::

      conda create -n myenv python=3.12

   Then activate the virtual environment ``myenv`` you created where you will
   install the software:

   .. code-block::

      conda activate myenv

#.
   If you are using GPUs, CUDA is required for the project to run. To install
   `PyTorch <https://pytorch.org/get-started/locally/>`_ with CUDA run (adjust
   the index URL for your CUDA version):

   .. code-block::

      pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

   **otherwise**\ , install the latest PyTorch CPU version (not recommended):

   .. code-block::

      pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

#.
   Clone the repository using ``git clone``, navigate to ``direct/`` and run

   .. code-block::

      python3 -m pip install .

   or, for an editable install:

   .. code-block::

      python3 -m pip install --no-build-isolation -e .

   This will install ``direct`` as a Python module. The C++ extensions are
   compiled automatically by ``scikit-build-core`` and require a working C++20
   compiler plus CMake (>= 3.18).

   Development and documentation dependencies are declared as PEP 735
   ``[dependency-groups]`` in ``pyproject.toml``. With ``pip`` ≥ 25.1 you can
   install them with ``pip install --group dev`` / ``--group docs``; otherwise
   use the ``uv`` path above.

Common Installation Issues
--------------------------

``elasticdeform`` and NumPy 2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PyPI wheels for ``elasticdeform`` are still compiled against NumPy 1.x and will
fail to import under NumPy 2 (``numpy.core.multiarray failed to import``).
``uv sync`` builds it from source automatically (see ``tool.uv.no-binary-package``
in ``pyproject.toml``). With ``pip`` / conda, rebuild against your NumPy:

.. code-block::

   pip install --force-reinstall --no-binary=elasticdeform 'elasticdeform>=0.5'

Elastic registration simulation (``registration_simulate_reference: ELASTIC``)
needs this package; other registration modes do not.

If you met other issues using DIRECT, please first update the repository to the
latest version, and rebuild the docker. When this does not work, create a
GitHub issue so we can see whether this is a bug, or an installation problem.
