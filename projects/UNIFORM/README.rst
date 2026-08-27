================================================================================
UNIFORM: Unified Deep Learning for Multi-organ / Multi-contrast MRI Reconstruction
================================================================================

Accepted at **MIDL 2025**:

  `UNIFORM: A Unified Deep Learning Framework for Multi-organ and Multi-contrast MRI Reconstruction <https://openreview.net/forum?id=I13Y1nU6gs>`__
  (`PDF <https://openreview.net/pdf?id=I13Y1nU6gs>`__)

UNIFORM trains one `vSHARP <https://arxiv.org/abs/2309.09954>`__ reconstructor on multi-coil
fastMRI brain, knee, and prostate plus CMRxRecon cardiac data. A single checkpoint covers
retrospective **2×, 4×, 6×, and 8×** acceleration across those anatomies and contrasts.

.. figure:: https://huggingface.co/NKI-AI/direct-uniform/resolve/main/uniform_figure1_pipeline.png
   :alt: UNIFORM training and inference pipeline (Figure 1, MIDL 2025)
   :width: 95%

   Figure 1 from the paper: one model, multiple anatomies and contrasts.

**Pretrained weights and inference YAMLs:** `NKI-AI/direct-uniform <https://huggingface.co/NKI-AI/direct-uniform>`__

Data layout
===========

Put **all** training volumes in one directory and **all** validation volumes in one directory
(fastMRI brain / knee / prostate and CMRxRecon cardiac can be symlinked into the same roots).
``uniform.yaml`` expects basenames only in the ``.lst`` files; paths come from
``--training-root`` and ``--validation-root``.

.. code-block:: text

   <base_path>/
   ├── training/
   │   ├── file_brain_AXT2_<...>.h5
   │   ├── file100<...>.h5              # knee
   │   ├── file_prostate_AXT2_<...>.h5
   │   ├── P0XX_cine_sax.mat            # cardiac (CMRxRecon)
   │   └── ...
   └── validation/
       ├── file_brain_<...>.h5
       ├── file100<...>.h5
       ├── file_prostate_<...>.h5
       ├── P0XX_cine_<...>.mat
       └── ...

Training lists
==============

``uniform.yaml`` references ``../lists/...`` (relative to ``projects/UNIFORM/``). Each
``.lst`` contains basenames for one **coil-count × matrix-size** bucket so every sample in a
batch shares the same k-space geometry—required for ``training.batch_size > 1``.

List stems mirror ``text_description`` entries in the config, e.g.
``train_brain_16_coils_768x396.lst`` or ``train_knee_15_coils_320x320.lst``. You must build
these lists yourself: scan your data, group by coils and shape, write one ``.lst`` per group.
Many training lists overlap with ``projects/JSSL/lists/`` (symlink or copy into
``projects/UNIFORM/lists/`` before launching).

Training
========

From the ``direct/`` repository root:

.. code-block:: bash

   direct train ./experiments/uniform \
     --training-root /path/to/<base_path>/training \
     --validation-root /path/to/<base_path>/validation \
     --cfg projects/UNIFORM/uniform.yaml \
     --num-gpus <number_of_gpus> \
     --num-workers <number_of_workers>

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
