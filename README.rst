.. image:: https://github.com/NKI-AI/direct/actions/workflows/tox.yml/badge.svg
   :target: https://github.com/NKI-AI/direct/actions/workflows/tox.yml
   :alt: tox

.. image:: https://github.com/NKI-AI/direct/actions/workflows/pylint.yml/badge.svg
   :target: https://github.com/NKI-AI/direct/actions/workflows/pylint.yml
   :alt: pylint

.. image:: https://github.com/NKI-AI/direct/actions/workflows/black.yml/badge.svg
   :target: https://github.com/NKI-AI/direct/actions/workflows/black.yml
   :alt: black

.. image:: https://api.codacy.com/project/badge/Grade/1c55d497dead4df69d6f256da51c98b7
   :target: https://app.codacy.com/gh/NKI-AI/direct?utm_source=github.com&utm_medium=referral&utm_content=NKI-AI/direct&utm_campaign=Badge_Grade_Settings
   :alt: codacy

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
     <figcaption>Zero-filled reconstruction, Compressed-Sensing (CS) reconstruction using the  `BART toolbox <https://mrirecon.github.io/bart/>`_, Reconstruction using a RIM model trained with DIRECT</figcaption>
   </div>



Installation and Quick Start
----------------------------

Check out the `documentation <https://docs.aiforoncology.nl/direct>`_ for installation and a quick start.

Projects
--------
In the `projects <https://github.com/NKI-AI/direct/tree/main/projects>`_ folder baseline model configurations are provided for each project.

Baselines and trained models
----------------------------

We provide a set of baseline results and trained models in the `DIRECT Model Zoo <https://docs.aiforoncology.nl/direct/model_zoo.html>`_. Baselines and trained models include the `Recurrent Variational Network (RecurrentVarNet) <https://arxiv.org/abs/2111.09639>`_, the `Recurrent Inference Machine (RIM) <https://www.sciencedirect.com/science/article/abs/pii/S1361841518306078>`_, the `End-to-end Variational Network (VarNet) <https://arxiv.org/pdf/2004.06688.pdf>`_, the `Learned Primal Dual Network (LDPNet) <https://arxiv.org/abs/1707.06474>`_, the `X-Primal Dual Network (XPDNet) <https://arxiv.org/abs/2010.07290>`_, the `KIKI-Net <https://pubmed.ncbi.nlm.nih.gov/29624729/>`_, the `U-Net <https://arxiv.org/abs/1811.08839>`_, the `Joint-ICNet <https://openaccess.thecvf.com/content/CVPR2021/papers/Jun_Joint_Deep_Model-Based_MR_Image_and_Coil_Sensitivity_Reconstruction_Network_CVPR_2021_paper.pdf>`_, and the `AIRS Medical fastmri model (MultiDomainNet) <https://arxiv.org/pdf/2012.06318.pdf>`_.

License and usage
-----------------

DIRECT is not intended for clinical use. DIRECT is released under the `Apache 2.0 License <LICENSE>`_.

Citing DIRECT
-------------

If you use DIRECT in your own research, or want to refer to baseline results published in the `DIRECT Model Zoo <model_zoo.rst>`_\ , please use the following BiBTeX entry:

.. code-block:: BibTeX

   @misc{DIRECTTOOLKIT,
     author =       {Yiasemis, George and Moriakov, Nikita and Karkalousos, Dimitrios and Caan, Matthan and Teuwen, Jonas},
     title =        {DIRECT: Deep Image REConstruction Toolkit},
     howpublished = {\url{https://github.com/NKI-AI/direct}},
     year =         {2021}
   }
