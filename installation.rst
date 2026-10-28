
Installation
============

Requirements
------------


* CUDA ≥ 12.6 supported GPU (optional, but recommended for training).
* Linux, macOS, or Windows with Python ≥ 3.11 (3.12+ recommended).
* PyTorch ≥ 2.11.
* Only required when building from source (prebuilt wheels are published to
  PyPI): a working C++20 compiler. The build uses ``meson-python`` with Meson
  and ``ninja``, both of which are installed automatically as build
  dependencies. ``nanobind`` is fetched and compiled from a pinned Meson
  subproject, so no separate install is needed.

Install using ``uv`` (recommended)
----------------------------------

`uv <https://docs.astral.sh/uv/>`_ is the recommended way to install ``DIRECT``
for both users and developers. It produces fast, reproducible environments
directly from ``pyproject.toml`` and the committed ``uv.lock``.

#.
   Install ``uv`` following the
   `official instructions <https://docs.astral.sh/uv/getting-started/installation/>`_.

#.
   Clone the repository and synchronise the environment. ``DIRECT`` is built
   without build isolation (so the editable install can rebuild the C++
   extensions on the fly using the environment's own ``ninja``), which means
   the build tooling must be installed *before* the project. Run ``uv sync``
   twice:

   .. code-block::

      git clone https://github.com/NKI-AI/direct.git
      cd direct
      uv sync --no-install-project   # runtime + dev + build tooling
      uv sync                        # build & install DIRECT itself
      uv sync --all-groups           # (optional) also install docs tooling

   ``uv`` will provision Python 3.12, create ``.venv/``, build the C++
   extensions via ``meson-python`` + ``nanobind``, and install all pinned
   dependencies from ``uv.lock``. The two-step sync keeps the editable install
   pointed at the persistent ``.venv`` ninja instead of a throwaway build
   environment, so imports keep working after ``uv`` prunes its cache.

#.
   Either activate the environment or prefix commands with ``uv run``:

   .. code-block::

      source .venv/bin/activate
      direct --help

   .. code-block::

      uv run direct --help
      uv run pytest

Install from PyPI (``pip``)
---------------------------

``DIRECT`` is published to PyPI as ``direct-recon`` (the import package is still
``direct``). On the supported platforms this fetches a prebuilt wheel (``abi3``
on Linux/macOS, Python-version-specific on Windows), so nothing is compiled:

.. code-block::

   pip install direct-recon

.. code-block:: python

   import direct

If no wheel is available for your platform, ``pip`` builds the C++ extensions
from the source distribution via ``meson-python`` + ``nanobind`` in an isolated
build environment; only a working C++20 compiler is required. The conda section
below shows a step-by-step build, including installing PyTorch with CUDA first.

For an editable/development install, install the build tooling first and disable
build isolation so the on-import rebuild keeps working:

.. code-block::

   pip install meson-python meson ninja
   pip install --no-build-isolation -e .

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
   Install ``DIRECT`` from PyPI. On the supported platforms this fetches a
   prebuilt wheel (``abi3`` on Linux/macOS, Python-version-specific on
   Windows), so nothing is compiled:

   .. code-block::

      pip3 install direct-recon

   To build from a checkout instead, clone the repository, navigate to
   ``direct/`` and run

   .. code-block::

      python3 -m pip install .

   ``pip`` builds the C++ extensions automatically via ``meson-python`` +
   ``nanobind`` in an isolated build environment; only a working C++20 compiler
   is required.

   For an editable install, first install the build tooling into the active
   environment and disable build isolation so the on-import rebuild keeps
   working:

   .. code-block::

      python3 -m pip install meson-python meson ninja
      python3 -m pip install --no-build-isolation -e .

   Development and documentation dependencies are declared as PEP 735
   ``[dependency-groups]`` in ``pyproject.toml``. With ``pip`` ≥ 25.1 you can
   install them with ``pip install --group dev`` / ``--group docs``; otherwise
   use the ``uv`` path above.

Common Installation Issues
--------------------------

If you met issues using DIRECT, please first update the repository to the
latest version, and rebuild the docker. When this does not work, create a
GitHub issue so we can see whether this is a bug, or an installation problem.
