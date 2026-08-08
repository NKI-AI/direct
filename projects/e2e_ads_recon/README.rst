=================================================================================
End-to-End Co-Optimization of Adaptive k-Space Sampling and Reconstruction
=================================================================================

This folder contains configuration files for reproducing experiments from:

`End-to-End Co-Optimization of Adaptive k-space Sampling and Reconstruction for
Dynamic MRI <https://proceedings.mlr.press/v315/yiasemis26a.html>`__
(Yiasemis et al., MIDL 2026, PMLR 315).

* `PMLR proceedings <https://proceedings.mlr.press/v315/yiasemis26a.html>`__
* `OpenReview <https://openreview.net/forum?id=0yrf87zVf2>`__
* Related earlier work: `arXiv:2403.10346 <https://arxiv.org/abs/2403.10346>`__

The paper jointly trains an adaptive :math:`k`-space sampling policy with a
reconstruction network so that acquired lines or pixels are chosen to maximize
image quality under a fixed acceleration budget. The Adaptive Dynamic Sampler
(ADS) can produce either **unified** patterns (one mask shared across all
temporal frames) or **frame-specific** patterns (a distinct mask per frame).
Both **1D** (phase-encode lines) and **2D** (Cartesian locations) sampling are
supported. Gradients flow from the reconstruction loss through the sampler via
a straight-through estimator.

Paper overview
==============

.. figure:: figures/method_overview.png
   :alt: End-to-end adaptive sampling and reconstruction pipeline
   :width: 95%

   Method overview. ACS / init measurements condition a sampling policy that
   proposes a binary undersampling mask; masked multi-coil :math:`k`-space is
   reconstructed by an unrolled network. The reconstruction loss trains both
   the reconstructor and the sampler end-to-end.

.. figure:: figures/ads_diagram.png
   :alt: Adaptive sampling policy diagram
   :width: 90%

   Adaptive sampler. A convolutional / MLP policy maps partially observed
   :math:`k`-space to per-location probabilities, which are budgeted and
   binarized into a sampling mask (unified or frame-specific).

In DIRECT this is not hard-coded to one reconstructor: any model that goes
through ``MRIModelEngine.perform_sampling`` can attach a sampler under
``additional_models.sampling_model``.

Usage in DIRECT
===============

Enable adaptive sampling by adding a ``sampling_model`` block:

.. code-block:: yaml

   additional_models:
     sampling_model:
       model_name: adaptive.policy.StraightThroughPolicy
       sampling_dimension: ONE_D   # or TWO_D
       sampling_type: STATIC       # unified (shared mask)
       # sampling_type: DYNAMIC_2D # frame-specific (per-frame mask)
       kspace_shape: [512, 246]
       # for frame-specific:
       # num_time_steps: 11

* ``STATIC`` — unified sampling (one pattern for all frames); pairs with 2D
  reconstruction models when the volume is handled frame-wise, or with 3D
  models that share a mask.
* ``DYNAMIC_2D`` / ``DYNAMIC_2D_NON_UNIFORM`` — frame-specific sampling; pair
  with 3D / cine models.
* For paper-style per-frame init / ACS masks under frame-specific sampling, set
  ``transforms.dynamic_mask: true``.

Configs in this folder
======================

Configs corresponding to the paper experiments (CMRxRecon cine). Naming uses
``frame`` for frame-specific sampling; omit it for unified:

.. list-table::
   :header-rows: 1
   :widths: 40 20 40

   * - Config
     - Reconstruction
     - Sampler
   * - ``vsharp_ads_1d.yaml``
     - vSHARP
     - ADS 1D unified
   * - ``medl_ads_1d.yaml``
     - MEDL
     - ADS 1D unified
   * - ``vsharp_ads_1d_frame.yaml``
     - vSHARP
     - ADS 1D frame-specific
   * - ``medl_ads_1d_frame.yaml``
     - MEDL
     - ADS 1D frame-specific
   * - ``vsharp_ads_1d_init2.yaml``
     - vSHARP
     - ADS 1D unified + init2
   * - ``medl_ads_1d_init2.yaml``
     - MEDL
     - ADS 1D unified + init2
   * - ``vsharp_ads_1d_frame_init2.yaml``
     - vSHARP
     - ADS 1D frame-specific + init2
   * - ``medl_ads_1d_frame_init2.yaml``
     - MEDL
     - ADS 1D frame-specific + init2
   * - ``vsharp_ads_2d.yaml``
     - vSHARP
     - ADS 2D unified
   * - ``medl_ads_2d.yaml``
     - MEDL
     - ADS 2D unified
   * - ``vsharp_ads_2d_frame.yaml``
     - vSHARP
     - ADS 2D frame-specific
   * - ``medl_ads_2d_frame.yaml``
     - MEDL
     - ADS 2D frame-specific

Naming scheme: ``{recon}_{sampler}_{dim}[_frame][_{extras}].yaml``

* ``vsharp`` / ``medl`` — reconstruction model
* ``ads`` — straight-through adaptive sampler
* ``1d`` / ``2d`` — sampling dimension
* ``frame`` — frame-specific patterns; omit for unified
* ``init2`` — ACS / target-acceleration initialization variant from the paper

Update dataset ``root`` paths and list files (``.lst``) in each YAML for your
machine before training or inference.

Training and inference
======================

.. code-block:: bash

   direct train <experiment_dir> \
     --cfg projects/e2e_ads_recon/<experiment_name>.yaml \
     --num-gpus <N>

   direct predict <experiment_dir> \
     --checkpoint <path/to/model_*.pt> \
     --cfg projects/e2e_ads_recon/<experiment_name>.yaml \
     --num-gpus <N>

Joint sampling, reconstruction, and registration (companion paper) lives in
``projects/e2e_ads_recon_reg``.
