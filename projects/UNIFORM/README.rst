================================================================================
UNIFORM: Unified Deep Learning for Multi-organ / Multi-contrast MRI Reconstruction
================================================================================

Accepted at **MIDL 2025**:

  `UNIFORM: A Unified Deep Learning Framework for Multi-organ and Multi-contrast MRI Reconstruction <https://openreview.net/forum?id=I13Y1nU6gs>`__
  (`PDF <https://openreview.net/pdf?id=I13Y1nU6gs>`__)

UNIFORM trains one `vSHARP <https://arxiv.org/abs/2309.09954>`__ reconstructor on multi-coil
fastMRI brain, knee, and prostate plus CMRxRecon cardiac data. A single checkpoint covers
retrospective accelerations :math:`R \in \{2,4,6,8\}` across those anatomies and contrasts.

.. figure:: https://huggingface.co/NKI-AI/direct-uniform/resolve/main/uniform_figure1_pipeline.png
   :alt: UNIFORM training and inference pipeline (Figure 1, MIDL 2025)
   :width: 95%

   Figure 1 from the paper: one model, multiple anatomies and contrasts.

**Pretrained weights and inference YAMLs:** `NKI-AI/direct-uniform <https://huggingface.co/NKI-AI/direct-uniform>`__

Training lists
==============

``uniform.yaml`` references training and validation files through ``../lists/...``. Those
lists are **not** flat file dumps: volumes are grouped by **coil count** and **matrix size**
(phase-encode :math:`\times` readout, e.g. ``768x396``) so every sample in a batch shares the
same k-space geometry. That grouping is what makes ``training.batch_size > 1`` practical.

Each ``text_description`` in the config mirrors a list stem, for example
``train_brain_16_coils_768x396`` or ``train_knee_15_coils_320x320``. Before training you need
to build or symlink compatible ``.lst`` files—many overlap with ``projects/JSSL/lists/``.
Scan your data, bucket filenames by coils and shape, and write one list per bucket.

Training
========

.. code-block:: bash

   direct train ./experiments/uniform \
     --cfg projects/UNIFORM/uniform.yaml \
     --num-gpus 8

See `Training <https://docs.aiforoncology.nl/direct/training.html>`__.

Inference
=========

Download weights and per-anatomy configs from the Hub. Use ``--filenames-filter`` with a
``.lst`` of basenames under ``--data-root`` (inference does not read ``filenames_lists`` from
the YAML):

.. code-block:: bash

   hf download NKI-AI/direct-uniform --local-dir ./uniform

   direct predict ./preds/knee \
     --cfg ./uniform/uniform_knee.yaml \
     --checkpoint ./uniform/uniform_vsharp.pt \
     --data-root /path/to/fastmri/knee/multicoil_val \
     --filenames-filter /path/to/filenames.lst \
     --num-gpus 1

Commented ``inference:`` blocks (brain / knee / prostate / cardiac; 2×–8×) are at the bottom
of ``uniform.yaml``. For routine use, prefer the Hub YAMLs.

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
