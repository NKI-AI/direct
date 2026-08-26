================================================================================
UNIFORM: Unified Deep Learning for Multi-organ / Multi-contrast MRI Reconstruction
================================================================================

This project reproduces and ships the **UNIFORM** model from:

  `UNIFORM: A Unified Deep Learning Framework for Multi-organ and Multi-contrast MRI Reconstruction <https://openreview.net/forum?id=I13Y1nU6gs>`__
  (`PDF <https://openreview.net/pdf?id=I13Y1nU6gs>`__)

UNIFORM trains a **single** `vSHARP <https://arxiv.org/abs/2309.09954>`__ network jointly on
fastMRI brain / knee / prostate and CMRxRecon cardiac multi-coil data, and supports
accelerations :math:`R \in \{2,4,6,8\}` plus zero-shot self-supervised fine-tuning on
unseen anatomies (e.g. breast).

.. figure:: assets/uniform_overview.png
   :alt: UNIFORM overview — brain, knee, prostate, and cardiac into one vSHARP model
   :width: 95%

   Overview: one UNIFORM vSHARP model for multi-anatomy multi-coil MRI reconstruction.

.. figure:: https://github.com/NKI-AI/direct/assets/71031687/493701b6-6efa-427d-9b4f-94a0ebcf3142
   :alt: vSHARP method overview
   :width: 90%

   Underlying reconstructor: vSHARP (variable Splitting Half-quadratic ADMM). Figure from the
   `vSHARP paper <https://doi.org/10.1016/j.mri.2024.110266>`__.

Paper & resources
=================

* OpenReview forum: https://openreview.net/forum?id=I13Y1nU6gs
* OpenReview PDF: https://openreview.net/pdf?id=I13Y1nU6gs
* Pretrained weights + inference YAMLs (Hub): https://huggingface.co/NKI-AI/direct-uniform
* vSHARP method paper: https://doi.org/10.1016/j.mri.2024.110266 · https://arxiv.org/abs/2309.09954
* DIRECT toolkit: https://github.com/NKI-AI/direct

Model
=====

* Architecture: ``vsharp.vsharp.VSharpNet`` with a **U-Net** image prior
* 12 ADMM steps, no parameter sharing between steps
* Sensitivity network: 2D U-Net (32 filters, 4 pool layers)
* Inference configs set ``image_unet_conv_out_bias: true`` (required for the released weights)

Quick start (inference)
=======================

.. code-block:: bash

   pip install huggingface_hub
   hf download NKI-AI/direct-uniform --local-dir ./uniform

   # Brain — 4× random (default YAML)
   direct predict ./preds/brain \
     --cfg ./uniform/uniform_brain.yaml \
     --checkpoint ./uniform/uniform_vsharp.pt \
     --data-root /path/to/fastmri/brain/multicoil_val \
     --num-gpus 1

   # Knee / prostate — 4× equispaced
   direct predict ./preds/knee \
     --cfg ./uniform/uniform_knee.yaml \
     --checkpoint ./uniform/uniform_vsharp.pt \
     --data-root /path/to/fastmri/knee/multicoil_val \
     --num-gpus 1

   # Cardiac — CMRxRecon ValidationSet FullSample (flatten to P0XX_cine_*.mat)
   direct predict ./preds/cardiac \
     --cfg ./uniform/uniform_cardiac.yaml \
     --checkpoint ./uniform/uniform_vsharp.pt \
     --data-root /path/to/cmrxrecon/ValidationSet/FullSample_flat \
     --num-gpus 1

Change acceleration by editing ``inference.dataset.transforms.masking`` and keeping
**single-element** lists (same pattern as other DIRECT Hub releases):

.. code-block:: yaml

   masking:
     name: FastMRIEquispaced   # brain default YAML uses FastMRIRandom
     accelerations: [8]
     center_fractions: [0.04]

======= =============== ==================
:math:`R` accelerations center_fractions
======= =============== ==================
2×      ``[2]``         ``[0.1]`` (equip) / ``[0.1]`` (random)
4×      ``[4]``         ``[0.08]``
6×      ``[6]``         ``[0.06]``
8×      ``[8]``         ``[0.04]``
======= =============== ==================

Project layout
==============

=============================== ==========================================================
Path                            Description
=============================== ==========================================================
``configs/train_uniform.yaml``  Multi-anatomy training config (current DIRECT format)
``configs/inference/``          Per-anatomy inference YAMLs (4× defaults; ``*_8x.yaml``)
``huggingface/``                Hub package (README, YAMLs, overview figure)
``tools/convert_uniform_checkpoint.py``  Legacy → current state_dict remapper
``tools/convert_legacy_config.py``       Flat → nested transforms helper
``lists/test/``                 Small filename lists for smoke tests
``jobs/``                       Example sbatch scripts (Kosmos)
=============================== ==========================================================

Training
========

Training lists are referenced from ``configs/`` as ``../lists/...``. Many lists overlap
with ``projects/JSSL/lists/``; symlink or copy them before launching:

.. code-block:: bash

   direct train ./experiments/uniform \
     --cfg projects/UNIFORM/configs/train_uniform.yaml \
     --num-gpus 8

See `Training <https://docs.aiforoncology.nl/direct/training.html>`__.

Cardiac data note
=================

For metric checks matching the paper / original Kosmos run, use CMRxRecon
**ValidationSet/FullSample** (e.g. P038 sax shape with phase-encode ``246``), not
TrainingSet FullSample (``162``). Flatten patient folders to ``P038_cine_sax.mat``-style
names for the CMRxRecon dataset loader.

Citing this work
================

.. code-block:: bibtex

   @inproceedings{Yiasemis_UNIFORM,
       title     = {{UNIFORM}: A Unified Deep Learning Framework for Multi-organ and Multi-contrast {MRI} Reconstruction},
       author    = {Yiasemis, George and Ferm, Jonatan and Moriakov, Nikita and Mann, Ritse M. and Sonke, Jan-Jakob and Teuwen, Jonas},
       booktitle = {Medical Imaging with Deep Learning},
       year      = {2025},
       url       = {https://openreview.net/forum?id=I13Y1nU6gs}
   }

   @article{Yiasemis_2025_vSHARP,
       title   = {vSHARP: Variable Splitting Half-quadratic ADMM algorithm for reconstruction of inverse-problems},
       author  = {Yiasemis, George and Moriakov, Nikita and Sonke, Jan-Jakob and Teuwen, Jonas},
       journal = {Magnetic Resonance Imaging},
       volume  = {115},
       pages   = {110266},
       year    = {2025},
       doi     = {10.1016/j.mri.2024.110266}
   }

   @article{DIRECTTOOLKIT,
       doi       = {10.21105/joss.04278},
       url       = {https://doi.org/10.21105/joss.04278},
       year      = {2022},
       publisher = {The Open Journal},
       volume    = {7},
       number    = {73},
       pages     = {4278},
       author    = {George Yiasemis and Nikita Moriakov and Dimitrios Karkalousos and Matthan Caan and Jonas Teuwen},
       title     = {DIRECT: Deep Image REConstruction Toolkit},
       journal   = {Journal of Open Source Software}
   }
