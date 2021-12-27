.. image:: https://github.com/NKI-AI/direct/actions/workflows/tox.yml/badge.svg
   :target: https://github.com/NKI-AI/direct/actions/workflows/tox.yml
   :alt: tox

.. image:: https://github.com/NKI-AI/direct/actions/workflows/pylint.yml/badge.svg
   :target: https://github.com/NKI-AI/direct/actions/workflows/pylint.yml
   :alt: pylint

.. image:: https://github.com/NKI-AI/direct/actions/workflows/black.yml/badge.svg
   :target: https://github.com/NKI-AI/direct/actions/workflows/black.yml
   :alt: black

.. image:: https://codecov.io/gh/NKI-AI/direct/branch/main/graph/badge.svg?token=STYAUFCKJY
   :target: https://codecov.io/gh/NKI-AI/direct
   :alt: codecov


DIRECT: Deep Image REConstruction Toolkit
=========================================

``DIRECT`` is a Python, end-to-end pipeline for solving Inverse Problems emerging in Imaging Processing. It is built with PyTorch and stores state-of-the-art Deep Learning imaging inverse problem solvers such as denoising, dealiasing and reconstruction. By defining a base forward linear or non-linear operator, ``DIRECT`` can be used for training models for recovering images such as MRIs from partially observed or noisy input data.
``DIRECT`` stores inverse problem solvers such as the Learned Primal Dual algorithm, Recurrent Inference Machine and Recurrent Variational Network, which were part of the winning solution in Facebook & NYUs FastMRI challenge in 2019 and the Calgary-Campinas MRI reconstruction challenge at MIDL 2020. For a full list of the baselines currently implemented in DIRECT see `here <#baselines-and-trained-models>`_.

.. raw:: html

   <div align="center">
     <img src=".github/direct.png"/>
   </div>



Installation
------------

See `install.md <install.md>`_.

Quick Start
-----------

See `getting_started.md <getting_started.md>`_\ , check out the `documentation <https://docs.aiforoncology.nl/direct>`_.
In the `projects <projects>`_ folder examples are given on how to train models on public datasets.

Baselines and trained models
----------------------------


* `Recurrent Variational Network (RecurrentVarNet) <https://arxiv.org/abs/2111.09639>`_
* `Recurrent Inference Machine (RIM) <https://www.sciencedirect.com/science/article/abs/pii/S1361841518306078>`_
* `End-to-end Variational Network (VarNet) <https://arxiv.org/pdf/2004.06688.pdf>`_
* `Learned Primal Dual Network (LDPNet) <https://arxiv.org/abs/1707.06474>`_
* `X-Primal Dual Network (XPDNet) <https://arxiv.org/abs/2010.07290>`_
* `KIKI-Net <https://pubmed.ncbi.nlm.nih.gov/29624729/>`_
* `U-Net <https://arxiv.org/abs/1811.08839>`_
* `Joint-ICNet <https://openaccess.thecvf.com/content/CVPR2021/papers/Jun_Joint_Deep_Model-Based_MR_Image_and_Coil_Sensitivity_Reconstruction_Network_CVPR_2021_paper.pdf>`_
* `AIRS Medical fastmri model (MultiDomainNet) <https://arxiv.org/pdf/2012.06318.pdf>`_

We provide a set of baseline results and trained models in the `DIRECT Model Zoo <model_zoo.md>`_.

License
-------

DIRECT is released under the `Apache 2.0 License <LICENSE>`_.

Citing DIRECT
-------------

If you use DIRECT in your own research, or want to refer to baseline results published in the
 `DIRECT Model Zoo <model_zoo.md>`_\ , please use the following BiBTeX entry:

.. code-block:: BibTeX

   @misc{DIRECTTOOLKIT,
     author =       {Yiasemis, George and Moriakov, Nikita and Karkalousos, Dimitrios and Caan, Matthan and Teuwen, Jonas},
     title =        {DIRECT: Deep Image REConstruction Toolkit},
     howpublished = {\url{https://github.com/NKI-AI/direct}},
     year =         {2021}
   }
