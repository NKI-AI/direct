.. raw:: html

   <div align="center" style="background-color:#e0e0e0; padding:20px;">
     <img src="https://github.com/NKI-AI/direct/assets/71031687/4a12e495-0b58-490c-b2ed-96dcaa619a80"/>
     <br>
     <figcaption style="color:#333; margin-top:10px;">DIRECT: Deep Image REConstruction Toolkit</figcaption>
   </div>


.. raw:: html

   <div align="center">

   <br />

   <a href="https://doi.org/10.21105/joss.04278">
   <img src="https://joss.theoj.org/papers/10.21105/joss.04278/status.svg" alt="JOSS"></a>
   <a href="https://github.com/NKI-AI/direct/actions/workflows/tox.yml">
   <img src="https://github.com/NKI-AI/direct/actions/workflows/tox.yml/badge.svg" alt="TOX"></a>
   <a href="https://github.com/NKI-AI/direct/actions/workflows/pylint.yml">
   <img src="https://github.com/NKI-AI/direct/actions/workflows/pylint.yml/badge.svg" alt="Pylint"></a>

   </div>

   <br />



DIRECT: Deep Image REConstruction Toolkit
=========================================

``DIRECT`` is a Python, end-to-end pipeline for solving Inverse Problems emerging in Imaging Processing.
It is built with PyTorch and stores state-of-the-art Deep Learning imaging inverse problem solvers such as denoising, dealiasing and reconstruction.
By defining a base forward linear or non-linear operator, ``DIRECT`` can be used for training models for recovering images such as MRIs from partially observed or noisy input data.
``DIRECT`` stores inverse problem solvers such as the vSHARP, Learned Primal Dual algorithm, Recurrent Inference Machine and Recurrent Variational Network, which were part of the winning solutions in Facebook & NYUs FastMRI challenge in 2019, the Calgary-Campinas MRI reconstruction challenge at MIDL 2020 and the CMRxRecon challenge 2023.
For a full list of the baselines currently implemented in DIRECT see `here <#baselines-and-trained-models>`_.

.. raw:: html

   <div align="center">
     <img src=".github/direct.png"/>
     <figcaption>Zero-filled reconstruction, Compressed-Sensing (CS) reconstruction using the BART toolbox, Reconstruction using a RIM model trained with DIRECT</figcaption>
   </div>



Installation and Quick Start
----------------------------

Check out the `documentation <https://docs.aiforoncology.nl/direct>`_ for installation and a quick start.

Projects
--------
In the `projects <https://github.com/NKI-AI/direct/tree/main/projects>`_ folder baseline model configurations are provided for each project.

Baselines and trained models
----------------------------

We provide a set of baseline results and trained models in the `DIRECT Model Zoo <https://docs.aiforoncology.nl/direct/model_zoo.html>`_. Baselines and trained models include the `vSHARP <https://arxiv.org/abs/2309.09954>`_, `Recurrent Variational Network (RecurrentVarNet) <https://arxiv.org/abs/2111.09639>`_, the `Recurrent Inference Machine (RIM) <https://www.sciencedirect.com/science/article/abs/pii/S1361841518306078>`_, the `End-to-end Variational Network (VarNet) <https://arxiv.org/pdf/2004.06688.pdf>`_, the `Learned Primal Dual Network (LDPNet) <https://arxiv.org/abs/1707.06474>`_, the `X-Primal Dual Network (XPDNet) <https://arxiv.org/abs/2010.07290>`_, the `KIKI-Net <https://pubmed.ncbi.nlm.nih.gov/29624729/>`_, the `U-Net <https://arxiv.org/abs/1811.08839>`_, the `Joint-ICNet <https://openaccess.thecvf.com/content/CVPR2021/papers/Jun_Joint_Deep_Model-Based_MR_Image_and_Coil_Sensitivity_Reconstruction_Network_CVPR_2021_paper.pdf>`_, and the `AIRS Medical fastmri model (MultiDomainNet) <https://arxiv.org/pdf/2012.06318.pdf>`_.

License and usage
-----------------

DIRECT is not intended for clinical use. DIRECT is released under the `Apache 2.0 License <LICENSE>`_.

Citing DIRECT
-------------

If you use DIRECT in your own research, or want to refer to baseline results published in the `DIRECT Model Zoo <model_zoo.rst>`_\ , please use the following BiBTeX entry:

.. code-block:: BibTeX

    @article{DIRECTTOOLKIT,
        doi = {10.21105/joss.04278},
        url = {https://doi.org/10.21105/joss.04278},
        year = {2022},
        publisher = {The Open Journal},
        volume = {7},
        number = {73},
        pages = {4278},
        author = {George Yiasemis and Nikita Moriakov and Dimitrios Karkalousos and Matthan Caan and Jonas Teuwen},
        title = {DIRECT: Deep Image REConstruction Toolkit},
        journal = {Journal of Open Source Software}
    }
