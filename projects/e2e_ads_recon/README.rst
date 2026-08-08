=================================================================================
End-to-End Adaptive k-Space Sampling and Reconstruction
=================================================================================

Configs and instructions for reproducing experiments from:

`End-to-End Co-Optimization of Adaptive k-space Sampling and Reconstruction for
Dynamic MRI <https://proceedings.mlr.press/v315/yiasemis26a.html>`__
(Yiasemis et al., MIDL 2026, PMLR 315).

* `PMLR proceedings <https://proceedings.mlr.press/v315/yiasemis26a.html>`__
* `OpenReview <https://openreview.net/forum?id=0yrf87zVf2>`__
* Related earlier work: `arXiv:2403.10346 <https://arxiv.org/abs/2403.10346>`__

Paper overview
==============

The paper jointly trains an **adaptive** :math:`k`-space sampler with a
reconstruction network so that acquired lines / pixels are chosen to maximize
reconstruction quality under a fixed acceleration budget. Sampling can be
**static** (one mask for all frames) or **dynamic** (a mask per temporal /
slice frame). Both **1D** (phase-encode lines) and **2D** (Cartesian pixel)
policies are supported.

.. figure:: figures/method_overview.png
   :alt: End-to-end adaptive sampling and reconstruction pipeline
   :width: 95%

   Method overview (paper): ACS / init measurements feed a sampling policy that
   proposes a binary undersampling mask; the masked multi-coil :math:`k`-space
   is reconstructed by an unrolled network (vSHARP or MEDL in the released
   configs). The reconstruction loss back-propagates through the sampler via a
   straight-through estimator.

.. figure:: figures/ads_diagram.png
   :alt: Adaptive sampling policy diagram
   :width: 90%

   Adaptive sampler (paper): a U-Net / MLP policy maps (partially observed)
   :math:`k`-space to per-location probabilities, which are binarized under a
   budget constraint and applied as a sampling mask.

What this enables in DIRECT
===========================

Adaptive sampling is **not** tied to one recon model. Enable it for any engine
that uses ``MRIModelEngine.perform_sampling`` by adding
``additional_models.sampling_model``:

.. code-block:: yaml

   additional_models:
     sampling_model:
       model_name: adaptive.policy.StraightThroughPolicy
       sampling_dimension: ONE_D   # or TWO_D
       sampling_type: STATIC       # or DYNAMIC_2D / DYNAMIC_2D_NON_UNIFORM
       kspace_shape: [512, 246]
       # for DYNAMIC_*:
       # num_time_steps: 11

Use ``STATIC`` with 2D recon models and ``DYNAMIC_2D`` /
``DYNAMIC_2D_NON_UNIFORM`` with 3D / cine models. For paper-style **per-frame
init/ACS masks**, set ``transforms.dynamic_mask: true`` in the YAML.

Released paper configs
======================

Only configs that match the published checkpoints (validated on CMRxRecon) are
kept in this folder:

+----------------------------------+------------------+------------------------+
| Config                           | Recon            | Sampler                |
+==================================+==================+========================+
| ``vsharp_ads_1d.yaml``            | vSHARP 2D/3D     | ADS 1D static          |
+----------------------------------+------------------+------------------------+
| ``medl_ads_1d.yaml``             | MEDL             | ADS 1D static          |
+----------------------------------+------------------+------------------------+
| ``vsharp_ads_1d_dyn.yaml``        | vSHARP 3D        | ADS 1D dynamic         |
+----------------------------------+------------------+------------------------+
| ``medl_ads_1d_dyn.yaml``          | MEDL             | ADS 1D dynamic         |
+----------------------------------+------------------+------------------------+
| ``vsharp_ads_1d_init2.yaml``      | vSHARP 3D        | ADS 1D + init2         |
+----------------------------------+------------------+------------------------+
| ``medl_ads_1d_init2.yaml``        | MEDL             | ADS 1D + init2         |
+----------------------------------+------------------+------------------------+
| ``vsharp_ads_1d_dyn_init2.yaml``  | vSHARP 3D        | ADS 1D dyn + init2     |
+----------------------------------+------------------+------------------------+
| ``medl_ads_1d_dyn_init2.yaml``    | MEDL             | ADS 1D dyn + init2     |
+----------------------------------+------------------+------------------------+
| ``vsharp_ads_2d.yaml``            | vSHARP           | ADS 2D static          |
+----------------------------------+------------------+------------------------+
| ``medl_ads_2d.yaml``             | MEDL             | ADS 2D static          |
+----------------------------------+------------------+------------------------+
| ``vsharp_ads_2d_dyn.yaml``        | vSHARP 3D        | ADS 2D dynamic         |
+----------------------------------+------------------+------------------------+
| ``medl_ads_2d_dyn.yaml``          | MEDL             | ADS 2D dynamic         |
+----------------------------------+------------------+------------------------+

Naming: ``{recon}_{sampler}_{dim}_{extras}.yaml`` — ``ads`` = straight-through
adaptive policy; ``1d`` / ``2d`` = sampling dimension; ``dyn`` = dynamic masks;
``init2`` = ACS / target-acceleration init variant from the paper.

Dataset paths
=============

YAMLs were collected from the original experiment trees. Update
``root`` / list files (``.lst``) under each dataset block for your machine
before training or inference. Typical data: **CMRxRecon** cine SAX.

Training / inference
====================

.. code-block:: bash

   # Train
   direct train <experiment_dir> \
     --cfg projects/e2e_ads_recon/<experiment_name>.yaml \
     --num-gpus <N>

   # Inference / validation from a checkpoint directory
   direct predict <experiment_dir> \
     --checkpoint <path/to/model_*.pt> \
     --cfg projects/e2e_ads_recon/<experiment_name>.yaml \
     --num-gpus <N>

Notes
-----

* Prefer **CPU** (or CUDA) when reproducing paper metrics from released weights;
  Apple MPS can diverge numerically for these models.
* Adaptive masks are **float** (soft / STE) tensors; engines use
  ``1 - sampling_mask.float()`` for DC fill rather than boolean ``~``.
* Checkpoint weights under ``projects/e2e_ads_recon/<name>/`` are gitignored;
  sync them separately if you have access to the release store.
