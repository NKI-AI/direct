=================================================================================
End-to-End Adaptive Sampling, Reconstruction, and Registration
=================================================================================

Configs and instructions for reproducing experiments from:

`Deep End-to-end Adaptive k-Space Sampling, Reconstruction, and Registration for
Dynamic MRI <https://arxiv.org/abs/2411.18249>`__
(Yiasemis et al., arXiv:2411.18249).

* `arXiv PDF <https://arxiv.org/pdf/2411.18249>`__
* Companion sampling+recon work (MIDL 2026):
  `PMLR <https://proceedings.mlr.press/v315/yiasemis26a.html>`__ /
  ``projects/e2e_ads_recon``

Paper overview
==============

This work extends end-to-end adaptive sampling + reconstruction with a
**registration** head that aligns reconstructed cine frames to a reference
cardiac phase. The three modules can be trained jointly (gradients through
recon → sampler) or in a **disjoint** / stage-wise fashion
(``train_end_to_end: false``).

.. figure:: figures/method_overview.png
   :alt: Joint adaptive sampling, reconstruction, and registration overview
   :width: 95%

   Method overview (paper): undersampled dynamic multi-coil :math:`k`-space is
   sampled by an adaptive policy, reconstructed, then registered to a reference
   frame so that both image quality and motion alignment are optimized.

.. figure:: figures/pipeline_diagram.png
   :alt: Full pipeline diagram
   :width: 95%

   Full pipeline (paper): ACS / init mask → sampling policy → reconstruction
   network → registration network producing a displacement field and warped
   moving image.

.. figure:: figures/ads_diagram.png
   :alt: Adaptive sampling policy
   :width: 90%

   Adaptive sampler (paper): same straight-through ADS family as
   ``projects/e2e_ads_recon`` (static or dynamic 1D line sampling).

.. figure:: figures/registration_model.png
   :alt: Registration network
   :width: 90%

   Registration model (paper): U-Net (default in released configs) predicts a
   dense displacement field; photometric and smoothness losses supervise
   warping of the reconstructed moving frames onto the reference.

What this enables in DIRECT
===========================

Assemble the joint pipeline with optional ``additional_models``:

1. ``sampling_model`` — adaptive / dynamic adaptive :math:`k`-space sampler
2. ``model`` — any DIRECT 2D/3D recon network
3. ``registration_model`` — DL registration (U-Net / VoxelMorph / ViT) or
   classical (Demons / optical flow)

.. code-block:: yaml

   additional_models:
     sampling_model:
       model_name: adaptive.policy.StraightThroughPolicy
       sampling_dimension: ONE_D
       sampling_type: DYNAMIC_2D
       kspace_shape: [512, 246]
       num_time_steps: 11
     registration_model:
       model_name: registration.registration.UnetRegistration2dModel
       train_end_to_end: true
       decoupled_training: false
       rec_loss_factor: 1.0
       reg_loss_factor: 1.0
       max_seq_len: 11

Enable registration transforms under each dataset:

.. code-block:: yaml

   transforms:
     registration:
       registration: true
       registration_simulate_reference: FROM_KEY
       registration_simulate_reference_from_key_index: 6
       registration_estimate_displacement: false
     use_acs_as_mask: true

``MRIModelEngine`` runs sampling then reconstruction; when ``ndim == 3`` and
``registration_model`` is set it also runs registration and adds registration
losses. Engines that override ``_do_iteration`` (vSHARP, CIRIM, RIM, MEDL)
call the same hooks.

Released paper configs
======================

Only configs corresponding to validated paper checkpoints are kept:

+-----------------------------------------------+------------------+---------------------------+
| Config                                        | Recon / sampler  | Registration              |
+===============================================+==================+===========================+
| ``vsharp_ads_1d_reg.yaml``                     | vSHARP + ADS 1D  | joint U-Net               |
+-----------------------------------------------+------------------+---------------------------+
| ``varnet_ads_1d_reg.yaml``                     | VarNet + ADS 1D  | joint U-Net               |
+-----------------------------------------------+------------------+---------------------------+
| ``vsharp_ads_1d_dyn_reg.yaml``                 | vSHARP + ADS dyn | joint U-Net               |
+-----------------------------------------------+------------------+---------------------------+
| ``varnet_ads_1d_dyn_reg.yaml``                 | VarNet + ADS dyn | joint U-Net               |
+-----------------------------------------------+------------------+---------------------------+
| ``vsharp_ads_1d_init_reg.yaml``                | ADS + init       | joint U-Net               |
+-----------------------------------------------+------------------+---------------------------+
| ``vsharp_ads_1d_dyn_init_reg.yaml``            | ADS dyn + init   | joint U-Net               |
+-----------------------------------------------+------------------+---------------------------+
| ``vsharp_fixed_1d_reg.yaml``                   | fixed mask       | joint U-Net               |
+-----------------------------------------------+------------------+---------------------------+
| ``vsharp_fixed_1d_dyn_reg.yaml``               | fixed dyn mask   | joint U-Net               |
+-----------------------------------------------+------------------+---------------------------+
| ``vsharp_loupe_1d_reg.yaml``                   | LOUPE            | joint U-Net               |
+-----------------------------------------------+------------------+---------------------------+
| ``vsharp_loupe_1d_dyn_reg.yaml``               | LOUPE dyn        | joint U-Net               |
+-----------------------------------------------+------------------+---------------------------+
| ``vsharp_ads_1d_reg_disjoint.yaml``            | ADS 1D           | disjoint / detached       |
+-----------------------------------------------+------------------+---------------------------+
| ``vsharp_ads_1d_dyn_reg_disjoint.yaml``        | ADS dyn          | disjoint / detached       |
+-----------------------------------------------+------------------+---------------------------+
| ``vsharp_ads_1d_init_reg_disjoint.yaml``       | ADS + init       | disjoint / detached       |
+-----------------------------------------------+------------------+---------------------------+
| ``vsharp_ads_1d_dyn_init_reg_disjoint.yaml``   | ADS dyn + init   | disjoint / detached       |
+-----------------------------------------------+------------------+---------------------------+

Naming: ``{recon}_{sampler}_{mode}_reg[_disjoint].yaml`` — ``ads`` /
``loupe`` / ``fixed``; ``dyn`` = dynamic sampling; ``init`` = ACS init
variant; ``disjoint`` = ``train_end_to_end: false``.

Dataset paths
=============

YAMLs came from ``kosmos:/projects/mri_reconstruction_registration``. Update
dataset ``root`` / list paths for your environment. Typical data: **CMRxRecon**
cine.

Training / inference
====================

.. code-block:: bash

   direct train <experiment_dir> \
     --cfg projects/e2e_ads_recon_reg/<experiment_name>.yaml \
     --num-gpus <N>

   direct predict <experiment_dir> \
     --checkpoint <path/to/model_*.pt> \
     --cfg projects/e2e_ads_recon_reg/<experiment_name>.yaml \
     --num-gpus <N>

Notes
-----

* Prefer **CPU** (or CUDA) when reproducing paper metrics; Apple MPS can
  diverge.
* Registration photometric metrics require ``reference_kspace`` to be
  normalized with the moving k-space (DIRECT does this by default).
* ``train_end_to_end: false`` detaches reconstruction before registration
  (disjoint ablations in the paper).
* ``decoupled_training: true`` alternates recon and registration backward
  passes.
* ``reg_loss_on_target: true`` also warps the GT moving image (``target``)
  with the predicted displacement field.
* Classical registration models have no trainable parameters; only recon
  (+ sampler) are optimized.
* Checkpoint weights under ``projects/e2e_ads_recon_reg/<name>/`` are
  gitignored.
