===============================
Static vs dynamic reconstruction
===============================

Two different things are easy to mix up:

* **Reconstruction dimensionality** — is each network input one 2D slice, or a 2D+time / 3D volume?
* **Sampling-mask mode** — is the undersampling pattern the same for every frame, or a new pattern per frame?

This page covers both. For the catalog of mask patterns, see :doc:`sampling_masks`.

Tensor shapes
=============

Complex k-space is stored with a trailing real/imag axis of size ``2``. After the MRI transforms, a training sample
looks like:

.. list-table::
   :header-rows: 1
   :widths: 22 18 28 32

   * - Setting
     - Dataset ``ndim``
     - ``kspace`` / ``masked_kspace``
     - Typical use
   * - Static 2D
     - ``2``
     - ``(coil, height, width, 2)``
     - FastMRI / Calgary-Campinas slice-wise training
   * - Dynamic 2D+time
     - ``3``
     - ``(coil, time, height, width, 2)``
     - Cine, CMRxRecon with ``kspace_context: time``
   * - Multislice 3D
     - ``3``
     - ``(coil, slice, height, width, 2)``
     - Volumetric data with ``kspace_context: slice``

The engine reads ``dataset.ndim`` and sets FFT axes accordingly (height/width). 2D reconstruction uses a 2D model
and ``ndim == 2``. **Time, multislice, and 3D (x, y, z) reconstruction need a 3D model** and ``ndim == 3``.

3D models
=========

These reconstruction models take 5-D k-space ``(coil, time_or_slice, height, width, 2)``:

* :class:`~direct.nn.varnet.varnet.EndToEndVarNet3D`
* :class:`~direct.nn.vsharp.vsharp.VSharpNet3D`
* :class:`~direct.nn.medl.medl.MEDL3D`
* :class:`~direct.nn.transformers.transformers.ImageDomainMRIViT3D`
* :class:`~direct.nn.transformers.transformers.KSpaceDomainMRIViT3D`

Do not pair them with a 2D dataset, and do not pair a 2D model
(:class:`~direct.nn.unet.unet_2d.Unet2d`, :class:`~direct.nn.varnet.varnet.EndToEndVarNet`,
:class:`~direct.nn.recurrentvarnet.recurrentvarnet.RecurrentVarNet`) with ``ndim == 3``.

Datasets that load time / multislice / 3D
=========================================

* :class:`~direct.data.datasets.CMRxReconDataset` — ``kspace_context: time`` for 2D+t cine, ``kspace_context: slice``
  for 3D (x, y, z). Omit it (or set ``null``) for per-slice 2D. See :doc:`../cmrxrecon`.
* :class:`~direct.data.datasets.FakeMRIBlobsDataset` — any truthy ``kspace_context`` (for example ``time``) emits one
  full volume per item; unset keeps 2D slices.

:class:`~direct.data.datasets.FastMRIDataset` and :class:`~direct.data.datasets.CalgaryCampinasDataset` are **slice-wise
2D** today. Their ``kspace_context`` integer only stacks a few neighbouring slices around the current index; it is not
a full multislice or 3D volume. Full FastMRI multislice / 3D loading is planned.

Sampling: static mask vs per-frame mask
=======================================

This is independent of the model. On a 2D+t volume you can still reuse **one** mask for every frame
(``masking.mode: static``, the default): the mask broadcasts along time. Or you can draw an independent mask per
frame (``masking.mode: dynamic``). :class:`~direct.common.subsample.KtRadialMaskFunc`,
:class:`~direct.common.subsample.KtUniformMaskFunc`, and :class:`~direct.common.subsample.KtGaussian1DMaskFunc`
are dynamic by construction. Adaptive sampling uses
``transforms.dynamic_mask: true`` instead; see :doc:`../e2e_ads_recon`.

.. figure:: ../_static/tutorials/sampling_masks_dynamic.png
   :alt: Dynamic random-line masks across time frames versus a static mask and ACS
   :align: center

   ``mode: dynamic`` draws a new random-line pattern per time frame (left). A static mask is one pattern reused
   for the whole volume.

Static 2D
=========

Leave ``kspace_context`` unset (or ``0``). Each dataset item is one slice. Pair that with a 2D model and a static mask:

.. code-block:: yaml

    training:
      datasets:
        - name: FastMRI
          transforms:
            masking:
              name: FastMRIRandom
              accelerations: [4]
              center_fractions: [0.08]
              mode: static   # default; one mask, broadcast over coils
    model:
      model_name: unet.unet_2d.Unet2d

On :class:`~direct.data.datasets.FakeMRIBlobsDataset`, ``spatial_shape: [n_slices, height, width]`` without
``kspace_context`` still yields 2D items (dataset length is ``sample_size × n_slices``). The engine log reports
``Data dimensionality: 2.``

Dynamic 2D+time
===============

Set ``kspace_context: time`` so each item is a cine volume, use a 3D model, and (optionally) a per-frame mask:

.. code-block:: yaml

    training:
      datasets:
        - name: CMRxRecon
          kspace_context: time
          transforms:
            masking:
              name: FastMRIRandom
              accelerations: [4]
              center_fractions: [0.08]
              mode: dynamic   # independent mask per time frame
    model:
      model_name: varnet.varnet.EndToEndVarNet3D

Dataset length equals the number of volumes, not ``volumes × time``. The engine log reports
``Data dimensionality: 3.``

Checklist
=========

* 2D reconstruction uses a 2D model and no ``kspace_context`` (or ``0``).
* Time, multislice, and 3D (x, y, z) reconstruction use a 3D model listed above, with
  ``kspace_context: time`` or ``kspace_context: slice`` on CMRxRecon (or a truthy ``kspace_context`` on FakeMRIBlobs).
* Same undersampling on every frame: ``masking.mode: static``.
* Time-varying undersampling: ``masking.mode: dynamic``, or ``KtRadial`` / ``KtUniform`` / ``KtGaussian1D``.
* ``ndim`` comes from the dataset, not from the YAML model name. A 2D engine cannot consume 5-D k-space.
