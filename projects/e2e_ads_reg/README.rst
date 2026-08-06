E2E adaptive sampling + reconstruction + registration
=====================================================

Configs for reproducing experiments from:

`Deep End-to-end Adaptive k-Space Sampling, Reconstruction, and Registration for Dynamic MRI <https://arxiv.org/abs/2411.18249>`_
(arXiv:2411.18249).

YAML files were collected from
``kosmos:/projects/mri_reconstruction_registration``. Update dataset roots and
list paths in each experiment ``.yaml`` for your environment before training.

What this enables
-----------------

The joint pipeline is assembled with **optional** ``additional_models``:

1. ``sampling_model`` — adaptive / dynamic adaptive :math:`k`-space sampler
   (same interface as ``projects/e2e_ads_recon``).
2. Reconstruction model (``model``) — any DIRECT 2D/3D recon network.
3. ``registration_model`` — DL (U-Net / VoxelMorph / ViT) or classical
   (Demons / optical flow) registration, trained end-to-end or decoupled.

Example:

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
     use_acs_as_mask: true   # common for adaptive policy training

``MRIModelEngine`` runs sampling then reconstruction; when ``ndim == 3`` and
``registration_model`` is set it also runs registration and adds registration
losses. Engines that override ``_do_iteration`` (vSHARP, CIRIM, RIM, MEDL)
call the same hooks.


Naming
------

Configs use a short scheme: ``{recon}_{sampler}_{mode}_{extras}.yaml``.

* ``vsharp`` / ``varnet`` — reconstruction model
* ``ads`` — straight-through adaptive sampler; ``loupe`` — parameterized; ``fixed`` — non-adaptive mask
* ``1d`` — line sampling; ``dyn`` — dynamic
* ``reg`` — joint registration (U-Net unless ``vit`` / ``voxelmorph``)
* ``disjoint`` — detached / stage-wise registration (``train_end_to_end: false``)
* ``recA_regB`` — ``rec_loss_factor=A``, ``reg_loss_factor=B``
* ``nosmooth`` / ``l1`` / ``ssim`` / ``large`` — loss or capacity ablations
* ``init`` — ACS/target-acceleration init variant; ``kt`` — k-t mask family

Typical training command
------------------------

.. code-block:: bash

   direct train <experiment_dir> \
     --cfg projects/e2e_ads_reg/<experiment_name>.yaml \
     --num-gpus <N>

Notes
-----

* ``train_end_to_end: false`` detaches the reconstruction before registration
  (disjoint / stage-wise style used in some paper ablations).
* ``decoupled_training: true`` alternates reconstruction and registration
  backward passes.
* ``reg_loss_on_target: true`` also warps the GT moving image (``target``) with
  the predicted displacement field and adds photometric loss vs the reference
  (logged as ``registration_target_*``). Reconstruction registration loss is
  unchanged.
* Classical registration models (Demons, optical flow) have no trainable
  parameters; only reconstruction (+ sampler) are optimized.
