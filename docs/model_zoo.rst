Model Zoo and Baselines
=======================

Introduction
------------

Pretrained DIRECT models are hosted on Hugging Face under
`NKI-AI <https://huggingface.co/NKI-AI>`_. Each collection is a set of
inference-ready ``<name>.yaml`` / ``<name>.pt`` pairs (configuration plus
weights). Download a repository, then pass those two files to
``direct predict``.

The older ``files.aiforoncology.nl`` zip archives (``config.yaml`` plus
``model_{iteration}.pt``) are no longer the supported distribution.

Download and run
----------------

Install the Hugging Face CLI if you do not already have it, then download a
collection:

.. code-block:: bash

   pip install huggingface_hub

   hf download NKI-AI/direct-calgary-campinas --local-dir ./calgary

   direct predict ./predictions \
       --cfg ./calgary/rim_5x.yaml \
       --checkpoint ./calgary/rim_5x.pt \
       --data-root /path/to/calgary_campinas \
       --num-gpus 1

The first argument to ``direct predict`` is the **prediction output
directory**. You can also browse a repository on the Hub and download
individual files from the web UI.

Collections
-----------

.. list-table::
   :header-rows: 1
   :widths: 32 38 30

   * - Hub repository
     - Contents
     - Project
   * - `NKI-AI/direct-calgary-campinas <https://huggingface.co/NKI-AI/direct-calgary-campinas>`_
     - Calgary-Campinas challenge baselines (5× / 10×)
     - `projects/calgary_campinas <https://github.com/NKI-AI/direct/tree/main/projects/calgary_campinas>`_
   * - `NKI-AI/direct-cmrxrecon-challenge23 <https://huggingface.co/NKI-AI/direct-cmrxrecon-challenge23>`_
     - CMRxRecon 2023 vSHARP (cine / mapping)
     - `projects/CMRxRecon <https://github.com/NKI-AI/direct/tree/main/projects/CMRxRecon>`_
   * - `NKI-AI/direct-cvpr2022-recurrentvarnet <https://huggingface.co/NKI-AI/direct-cvpr2022-recurrentvarnet>`_
     - RecurrentVarNet (CVPR 2022) and paper baselines
     - `projects/cvpr2022_recurrentvarnet <https://github.com/NKI-AI/direct/tree/main/projects/cvpr2022_recurrentvarnet>`_
   * - `NKI-AI/direct-vsharp-multianatomy <https://huggingface.co/NKI-AI/direct-vsharp-multianatomy>`_
     - vSHARP models for brain, knee, prostate, breast, cardiac, and a universal checkpoint
     - `projects/vSHARP <https://github.com/NKI-AI/direct/tree/main/projects/vSHARP>`_
   * - `NKI-AI/direct-e2e-ads-recon <https://huggingface.co/NKI-AI/direct-e2e-ads-recon>`_
     - End-to-end adaptive sampling and reconstruction
     - `projects/e2e_ads_recon <https://github.com/NKI-AI/direct/tree/main/projects/e2e_ads_recon>`_
   * - `NKI-AI/direct-e2e-ads-recon-reg <https://huggingface.co/NKI-AI/direct-e2e-ads-recon-reg>`_
     - Adaptive sampling, reconstruction, and registration
     - `projects/e2e_ads_recon_reg <https://github.com/NKI-AI/direct/tree/main/projects/e2e_ads_recon_reg>`_
   * - `NKI-AI/direct-modulated-convolution <https://huggingface.co/NKI-AI/direct-modulated-convolution>`_
     - Conditional vSHARP with modulated convolutions (knee / prostate)
     - `projects/modulated_convolution <https://github.com/NKI-AI/direct/tree/main/projects/modulated_convolution>`_
   * - `NKI-AI/direct-uniform <https://huggingface.co/NKI-AI/direct-uniform>`_
     - UNIFORM multi-organ / multi-contrast vSHARP (brain, knee, prostate, cardiac)
     - `projects/UNIFORM <https://github.com/NKI-AI/direct/tree/main/projects/UNIFORM>`_

Challenge Poisson-disk masks used by Calgary-Campinas models are also on the
Hub as
`NKI-AI/direct-mri-masks <https://huggingface.co/datasets/NKI-AI/direct-mri-masks>`_.

How to read the tables
----------------------

* **Name** is the Hub file stem. Weights are ``{name}.pt`` and the matching
  inference config is ``{name}.yaml``.
* Training YAML templates for a project still live under
  ``projects/{project}/``. The Hub files are the inference configs that pin
  one acceleration (and ACS fraction, when applicable).

Calgary-Campinas
----------------

`Calgary-Campinas MR reconstruction challenge <https://sites.google.com/view/calgary-campinas-dataset/mr-reconstruction-challenge>`_
baselines. Models were trained on 47 multi-coil (12-coil) brain volumes with
retrospective 5× or 10× undersampling using the challenge Poisson-disk masks.
Metrics below are on the 20-volume validation set.

Each checkpoint was trained at a **single** challenge rate. Use the matching
``*_5x`` or ``*_10x`` pair; do not swap accelerations on the same weights.

.. code-block:: bash

   hf download NKI-AI/direct-calgary-campinas --local-dir ./calgary

Validation set (12 coils, 20 volumes)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Model
     - Name
     - Acceleration
     - SSIM
     - pSNR
     - VIF
     - NMSE
   * - RecurrentVarNet
     - `recurrentvarnet_5x <https://huggingface.co/NKI-AI/direct-calgary-campinas/blob/main/recurrentvarnet_5x.pt>`_
     - 5×
     - 0.943
     - 36.1
     - 0.964
     - \-
   * - RecurrentVarNet
     - `recurrentvarnet_10x <https://huggingface.co/NKI-AI/direct-calgary-campinas/blob/main/recurrentvarnet_10x.pt>`_
     - 10×
     - 0.911
     - 33.0
     - 0.926
     - \-
   * - LPDNet
     - `lpdnet_5x <https://huggingface.co/NKI-AI/direct-calgary-campinas/blob/main/lpdnet_5x.pt>`_
     - 5×
     - 0.937
     - 35.6
     - 0.953
     - \-
   * - LPDNet
     - `lpdnet_10x <https://huggingface.co/NKI-AI/direct-calgary-campinas/blob/main/lpdnet_10x.pt>`_
     - 10×
     - 0.901
     - 32.2
     - 0.919
     - \-
   * - IterDualNet
     - `iterdualnet_5x <https://huggingface.co/NKI-AI/direct-calgary-campinas/blob/main/iterdualnet_5x.pt>`_
     - 5×
     - 0.936
     - 35.2
     - 0.973
     - 0.0051
   * - IterDualNet
     - `iterdualnet_10x <https://huggingface.co/NKI-AI/direct-calgary-campinas/blob/main/iterdualnet_10x.pt>`_
     - 10×
     - 0.898
     - 31.9
     - 0.930
     - 0.0112
   * - ConjGradNet
     - `conjgradnet_5x <https://huggingface.co/NKI-AI/direct-calgary-campinas/blob/main/conjgradnet_5x.pt>`_
     - 5×
     - 0.937
     - 35.51
     - 0.964
     - 0.0047
   * - ConjGradNet
     - `conjgradnet_10x <https://huggingface.co/NKI-AI/direct-calgary-campinas/blob/main/conjgradnet_10x.pt>`_
     - 10×
     - 0.918
     - 32.3
     - 0.918
     - 0.010
   * - RIM
     - `rim_5x <https://huggingface.co/NKI-AI/direct-calgary-campinas/blob/main/rim_5x.pt>`_
     - 5×
     - 0.932
     - 35.0
     - 0.964
     - \-
   * - RIM
     - `rim_10x <https://huggingface.co/NKI-AI/direct-calgary-campinas/blob/main/rim_10x.pt>`_
     - 10×
     - 0.891
     - 31.7
     - 0.911
     - \-
   * - VarNet
     - `varnet_5x <https://huggingface.co/NKI-AI/direct-calgary-campinas/blob/main/varnet_5x.pt>`_
     - 5×
     - 0.917
     - 33.3
     - 0.937
     - \-
   * - VarNet
     - `varnet_10x <https://huggingface.co/NKI-AI/direct-calgary-campinas/blob/main/varnet_10x.pt>`_
     - 10×
     - 0.862
     - 29.9
     - 0.861
     - \-
   * - Joint-ICNet
     - `jointicnet_5x <https://huggingface.co/NKI-AI/direct-calgary-campinas/blob/main/jointicnet_5x.pt>`_
     - 5×
     - 0.904
     - 32.0
     - 0.940
     - \-
   * - Joint-ICNet
     - `jointicnet_10x <https://huggingface.co/NKI-AI/direct-calgary-campinas/blob/main/jointicnet_10x.pt>`_
     - 10×
     - 0.854
     - 29.4
     - 0.853
     - \-
   * - XPDNet
     - `xpdnet_5x <https://huggingface.co/NKI-AI/direct-calgary-campinas/blob/main/xpdnet_5x.pt>`_
     - 5×
     - 0.907
     - 32.3
     - 0.965
     - \-
   * - XPDNet
     - `xpdnet_10x <https://huggingface.co/NKI-AI/direct-calgary-campinas/blob/main/xpdnet_10x.pt>`_
     - 10×
     - 0.855
     - 29.7
     - 0.837
     - \-
   * - KIKI-Net
     - `kikinet_5x <https://huggingface.co/NKI-AI/direct-calgary-campinas/blob/main/kikinet_5x.pt>`_
     - 5×
     - 0.888
     - 29.6
     - 0.919
     - \-
   * - KIKI-Net
     - `kikinet_10x <https://huggingface.co/NKI-AI/direct-calgary-campinas/blob/main/kikinet_10x.pt>`_
     - 10×
     - 0.833
     - 27.5
     - 0.856
     - \-
   * - U-Net
     - `unet_5x <https://huggingface.co/NKI-AI/direct-calgary-campinas/blob/main/unet_5x.pt>`_
     - 5×
     - 0.871
     - 29.5
     - 0.895
     - \-
   * - U-Net
     - `unet_10x <https://huggingface.co/NKI-AI/direct-calgary-campinas/blob/main/unet_10x.pt>`_
     - 10×
     - 0.821
     - 27.8
     - 0.837
     - \-

CMRxRecon Challenge 2023
------------------------

`CMRxRecon 2023 <https://cmrxrecon.github.io/>`_ vSHARP checkpoint
(`vsharp_2d_dynamic <https://huggingface.co/NKI-AI/direct-cmrxrecon-challenge23>`_).
Use the challenge-provided undersampling masks (YAML ``extra_keys`` such as
``mask04``). Do not add a multi-acceleration training ``masking.accelerations``
list for inference.

.. code-block:: bash

   hf download NKI-AI/direct-cmrxrecon-challenge23 --local-dir ./cmrx23

   direct predict ./predictions \
       --cfg ./cmrx23/vsharp_2d_dynamic.yaml \
       --checkpoint ./cmrx23/vsharp_2d_dynamic.pt \
       --data-root /path/to/cmrxrecon \
       --num-gpus 1

Test-set metrics
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Task
     - Model
     - Name
     - SSIM
     - pSNR
     - NMSE
   * - Task 1 (Cine)
     - vSHARP 3D
     - `vsharp_2d_dynamic <https://huggingface.co/NKI-AI/direct-cmrxrecon-challenge23/blob/main/vsharp_2d_dynamic.pt>`_
     - 0.988
     - 46.2
     - 0.0037
   * - Task 2 (Mapping)
     - vSHARP 3D
     - `vsharp_2d_dynamic <https://huggingface.co/NKI-AI/direct-cmrxrecon-challenge23/blob/main/vsharp_2d_dynamic.pt>`_
     - 0.984
     - 44.4
     - 0.0043

CVPR 2022 RecurrentVarNet
-------------------------

Checkpoints from
`Recurrent Variational Network <https://arxiv.org/abs/2111.09639>`_
(Yiasemis et al., CVPR 2022):

* ``calgary_campinas/`` — main paper experiments and ablations
  (``recurrentvarnet_shared_weights``, ``*_noRSI``, ``*_noSER``, ``*_T11``)
  plus comparison baselines (RIM, XPDNet, U-Net, VarNet). Inference YAMLs
  default to 5× (10× is commented).
* ``fastmri_axt1/`` — paper appendix on fastMRI brain AXT1. Defaults to 4×
  random Cartesian undersampling (8× commented).

.. code-block:: bash

   hf download NKI-AI/direct-cvpr2022-recurrentvarnet --local-dir ./cvpr_rvn

   direct predict ./predictions \
       --cfg ./cvpr_rvn/calgary_campinas/recurrentvarnet_shared_weights.yaml \
       --checkpoint ./cvpr_rvn/calgary_campinas/recurrentvarnet_shared_weights.pt \
       --data-root /path/to/calgary_campinas \
       --num-gpus 1

vSHARP multi-anatomy
--------------------

Anatomy-specific and universal
`vSHARP <https://arxiv.org/abs/2309.09954>`_ checkpoints
(`NKI-AI/direct-vsharp-multianatomy <https://huggingface.co/NKI-AI/direct-vsharp-multianatomy>`_):
``vsharp_brain``, ``vsharp_knee``, ``vsharp_prostate``, ``vsharp_breast``,
``vsharp_cardiac``, and ``vsharp_universal``.

These models were trained on a mixed acceleration / sampling schedule
(:math:`R \in \{2, 4, 6, 8, 10\}`). Released YAMLs pin **one** acceleration and
**one** ACS fraction (default 4× / 0.08). Keep both lists length 1 when
changing rate. See the model card for the matching ACS values and default
mask per anatomy.

.. code-block:: bash

   hf download NKI-AI/direct-vsharp-multianatomy --local-dir ./vsharp_multianatomy

   direct predict ./predictions \
       --cfg ./vsharp_multianatomy/vsharp_knee.yaml \
       --checkpoint ./vsharp_multianatomy/vsharp_knee.pt \
       --data-root /path/to/fastmri/knee/multicoil_val \
       --num-gpus 1

End-to-end adaptive sampling
----------------------------

Adaptive :math:`k`-space sampling jointly trained with reconstruction
(`MIDL 2026 <https://proceedings.mlr.press/v315/yiasemis26a.html>`_,
`NKI-AI/direct-e2e-ads-recon <https://huggingface.co/NKI-AI/direct-e2e-ads-recon>`_).
Pairs include vSHARP or MEDL reconstructors with ADS in 1D / 2D, unified or
frame-specific, with optional sampling init. Inference YAMLs default to 4×;
other trained rates are commented under ``masking``.

.. code-block:: bash

   hf download NKI-AI/direct-e2e-ads-recon --local-dir ./e2e_ads_recon

   direct predict ./predictions \
       --cfg ./e2e_ads_recon/vsharp_ads_1d.yaml \
       --checkpoint ./e2e_ads_recon/vsharp_ads_1d.pt \
       --data-root /path/to/cmrxrecon \
       --num-gpus 1

The companion collection
`NKI-AI/direct-e2e-ads-recon-reg <https://huggingface.co/NKI-AI/direct-e2e-ads-recon-reg>`_
adds a registration network (`arXiv:2411.18249 <https://arxiv.org/abs/2411.18249>`_).
Those YAMLs enable registration transforms that build a ``reference_image``
(default: drop frame index ``6``). Volumes must have enough temporal frames
for that index.

.. code-block:: bash

   hf download NKI-AI/direct-e2e-ads-recon-reg --local-dir ./e2e_ads_recon_reg

   direct predict ./predictions \
       --cfg ./e2e_ads_recon_reg/vsharp_ads_1d_phase_reg.yaml \
       --checkpoint ./e2e_ads_recon_reg/vsharp_ads_1d_phase_reg.pt \
       --data-root /path/to/cmrxrecon \
       --num-gpus 1

Modulated convolution
---------------------

Conditional vSHARP models with modulated convolutions
(`MIDL 2026 <https://proceedings.mlr.press/v315/moriakov26a.html>`_,
`NKI-AI/direct-modulated-convolution <https://huggingface.co/NKI-AI/direct-modulated-convolution>`_)
for fastMRI knee and prostate. A single checkpoint covers a range of
accelerations seen in training; released YAMLs pin one validation rate
(default 4× / ACS 0.08). Files live under ``knee/`` and ``prostate/``.

.. code-block:: bash

   hf download NKI-AI/direct-modulated-convolution --local-dir ./modconv

   direct predict ./predictions \
       --cfg ./modconv/knee/vsharp_modconv_features_triang_32_8.yaml \
       --checkpoint ./modconv/knee/vsharp_modconv_features_triang_32_8.pt \
       --data-root /path/to/fastmri/knee/multicoil_val \
       --num-gpus 1

UNIFORM (multi-organ vSHARP)
----------------------------

A single vSHARP checkpoint trained jointly on fastMRI brain / knee / prostate
and CMRxRecon cardiac data
(`OpenReview <https://openreview.net/forum?id=I13Y1nU6gs>`__,
`NKI-AI/direct-uniform <https://huggingface.co/NKI-AI/direct-uniform>`__).
Default YAMLs pin 4×; other rates are commented under ``masking``.

.. code-block:: bash

   hf download NKI-AI/direct-uniform --local-dir ./uniform

   direct predict ./predictions \
       --cfg ./uniform/uniform_knee.yaml \
       --checkpoint ./uniform/uniform_vsharp.pt \
       --data-root /path/to/fastmri/knee/multicoil_val \
       --filenames-filter /path/to/filenames.lst \
       --num-gpus 1

License
-------

Check the model card on each Hub repository. The Calgary-Campinas zoo is
released under
`Creative Commons Attribution-ShareAlike 3.0 <https://creativecommons.org/licenses/by-sa/3.0/>`_.
The other DIRECT collections listed here are Apache 2.0.

If you use these models, cite the
`DIRECT toolkit <https://doi.org/10.21105/joss.04278>`_ and the paper named
on the corresponding model card.
