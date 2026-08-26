================================================================================
UNIFORM: Unified Deep Learning for Multi-organ / Multi-contrast MRI Reconstruction
================================================================================

MIDL 2025 short paper:
`UNIFORM: A Unified Deep Learning Framework for Multi-organ and Multi-contrast MRI Reconstruction <https://openreview.net/forum?id=I13Y1nU6gs>`__
(`PDF <https://openreview.net/pdf?id=I13Y1nU6gs>`__)

One `vSHARP <https://arxiv.org/abs/2309.09954>`__ model trained jointly on fastMRI
brain, knee, and prostate plus CMRxRecon cardiac multi-coil data, with retrospective
accelerations :math:`R \in \{2,4,6,8\}`.

**Pretrained weights and inference YAMLs:** `NKI-AI/direct-uniform <https://huggingface.co/NKI-AI/direct-uniform>`__

Training
========

Training lists are referenced as ``../lists/...`` (overlap with ``projects/JSSL/lists/``;
symlink or copy before launching).

.. code-block:: bash

   direct train ./experiments/uniform \
     --cfg projects/UNIFORM/train_uniform.yaml \
     --num-gpus 8

See `Training <https://docs.aiforoncology.nl/direct/training.html>`__.

Inference
=========

Download weights and per-anatomy inference configs from the Hub, then run
``direct predict`` with ``--filenames-filter`` (path to a ``.lst`` of basenames under
``--data-root``):

.. code-block:: bash

   hf download NKI-AI/direct-uniform --local-dir ./uniform

   direct predict ./preds/knee \
     --cfg ./uniform/uniform_knee.yaml \
     --checkpoint ./uniform/uniform_vsharp.pt \
     --data-root /path/to/fastmri/knee/multicoil_val \
     --filenames-filter /path/to/filenames.lst \
     --num-gpus 1

Commented ``inference:`` blocks for each anatomy and acceleration are at the bottom of
``train_uniform.yaml``. For day-to-day inference, use the Hub YAMLs.

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
