==================
MRI transforms
==================

Every training sample is a dict. :func:`direct.data.mri_transforms.build_mri_transforms` builds a
:class:`~direct.data.mri_transforms.Compose` pipeline that turns fully sampled k-space into the tensors a reconstruction
model consumes: ``masked_kspace``, ``sampling_mask``, ``sensitivity_map``, ``target``, and a ``scaling_factor``.

This page shows the default order, the YAML keys, the sample dict, and how to add a custom transform. The YAML is the
same mapping ``direct train`` flattens into :func:`~direct.data.mri_transforms.build_mri_transforms` (nested sections
are flattened; only the leaf keys are passed).

Pipeline order
==============

For supervised training (``transforms_type: SUPERVISED``, the default):

#. :class:`~direct.data.mri_transforms.ToTensor` — complex ndarray → tensor with a last axis of size ``2``.
#. :class:`~direct.data.mri_transforms.CropKspace` — optional image-domain crop, then FFT back.
#. :class:`~direct.data.mri_transforms.RescaleKspace` / :class:`~direct.data.mri_transforms.PadKspace` — optional.
#. :class:`~direct.data.mri_transforms.RandomRotation` / :class:`~direct.data.mri_transforms.RandomFlip` /
   :class:`~direct.data.mri_transforms.RandomReverse` — optional augmentations (90° rotations only).
#. Zero-padding detection on empty k-space edges.
#. :class:`~direct.data.mri_transforms.CreateSamplingMask` — if ``masking`` is set. See :doc:`sampling_masks`.
#. Coil compression and / or :class:`~direct.data.mri_transforms.PadCoilDimensionModule`.
#. :class:`~direct.data.mri_transforms.EstimateSensitivityMapModule` — ACS-based maps unless you set
   ``estimate_sensitivity_maps: false``.
#. :class:`~direct.data.mri_transforms.ComputeImageModule` — RSS (default) or SENSE ``target``.
#. :class:`~direct.data.mri_transforms.ApplyMaskModule` — writes ``masked_kspace``.
#. Scaling from ``scaling_key`` (default ``masked_kspace``) and :class:`~direct.data.mri_transforms.NormalizeModule`.
#. Drop ``kspace`` and ``acs_mask`` when ``delete_kspace`` / ``delete_acs_mask`` are true (defaults).

Self-supervised SSDU adds a mask split after that; set ``transforms_type: SSL_SSDU``. Registration extras are
documented in :doc:`../e2e_ads_recon_reg`.

Sample keys
===========

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Key
     - Role
   * - ``kspace``
     - Fully sampled multi-coil k-space. Removed after masking if ``delete_kspace: true``.
   * - ``masked_kspace``
     - Undersampled k-space the model sees.
   * - ``sampling_mask``
     - Binary mask, broadcastable over coils.
   * - ``sensitivity_map``
     - Coil sensitivities (ACS estimate, ESPIRiT, unit, or a learned ``sensitivity_model``).
   * - ``target``
     - Supervised image (RSS or SENSE).
   * - ``scaling_factor``
     - Scalar used to normalize tensors.
   * - ``filename``, ``slice_no``
     - Identity; ``use_seed: true`` hashes ``filename`` so every slice of a volume shares a mask.

YAML
====

Nested blocks under ``transforms`` map onto :class:`~direct.data.datasets_config.TransformsConfig`. Enum values use
the **member name** (``SENSE``, ``RSS_ESTIMATE``, ``RANDOM``, ``DISCRETE``).

.. literalinclude:: cfgs/mri_transforms.yaml
   :language: yaml

The example crops ``48×48`` k-space to ``32×32``, applies random 90° rotation and flip, estimates RSS sensitivity
maps, pads the coil axis from 4 to 6, and trains a tiny U-Net. Run it:

.. code-block:: bash

    direct train <experiment_directory> --cfg docs/tutorials/cfgs/mri_transforms.yaml \
        --num-gpus 1 --device cpu --num-workers 0 --name smoke

Useful leaves
=============

* **``cropping.crop``** — ``[height, width]``, ``reconstruction_size``, or ``null``.
* **``cropping.image_center_crop``** — center vs random crop when ``crop`` is a size.
* **``random_augmentations.random_*_probability``** — ``0`` disables that augmentation.
* **``sensitivity_map_estimation.sensitivity_maps_type``** — ``RSS_ESTIMATE``, ``ESPIRIT``, or ``UNIT``.
* **``normalization.scaling_key``** — usually ``masked_kspace``.
* **``pad_coils``** — pad the coil axis so batches with different coil counts can collate
  (:class:`~direct.data.mri_transforms.PadCoilDimensionModule`).
* **``delete_kspace``** — set ``false`` if a k-space loss still needs the fully sampled data.
* **``use_seed``** — ``true`` at validation/inference so masks are reproducible; often ``false`` while training.

Adding a custom transform
=========================

YAML cannot name an arbitrary class. A new transform is a :class:`~direct.utils.DirectTransform` (or
:class:`~direct.utils.DirectModule` if it should run as an ``nn.Module``) with ``__call__(sample) -> sample``. Put it
in :mod:`direct.data.mri_transforms` and insert it in the list inside
:func:`~direct.data.mri_transforms.build_supervised_mri_transforms` (or :func:`~direct.data.mri_transforms.build_mri_transforms`
for SSL). If it needs a config key, add that key to :class:`~direct.data.datasets_config.TransformsConfig`.

.. code-block:: python

    from typing import Any

    from direct.utils import DirectTransform


    class ScaleTarget(DirectTransform):
        """Multiply the supervised target by a constant."""

        def __init__(self, factor: float = 1.0) -> None:
            super().__init__()
            self.factor = factor

        def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
            sample["target"] = sample["target"] * self.factor
            return sample

Keep spatial / coil / complex axis conventions from :class:`~direct.utils.DirectTransform`: for dict samples,
``coil_dim`` is ``1`` and 2D spatial axes are ``(1, 2)`` on k-space **without** a batch dimension.
