:github_url: https://github.com/NKI-AI/direct/


DIRECT documentation
====================
DIRECT is a Python, end-to-end pipeline for solving Inverse Problems emerging in medical imaging.
It is built with `PyTorch <https://pytorch.org>`_ and stores state-of-the-art Deep Learning imaging inverse problem solvers such as denoising,
dealiasing and reconstruction.
By defining a base forward linear or non-linear operator, DIRECT can be used for training models for recovering
images such as MRIs from partially observed or noisy input data.



.. toctree::
   :maxdepth: 1
   :caption: Index

   installation
   getting_started
   authors
   history

.. toctree::
   :maxdepth: 1
   :caption: Training and inference

   training
   inference
   config

.. toctree::
   :maxdepth: 1
   :caption: Add more Datasets and Sub-Samplers

   datasets
   samplers

.. toctree::
   :maxdepth: 1
   :caption: Model Zoo

   model_zoo.rst

.. toctree::
   :maxdepth: 2
   :caption: API Documentation

   modules


Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
