===================
Shepp Logan Dataset
===================

Details
-------

The Shepp Logan Dataset is based on the implementation of the 3-dimensional Shepp Logan phantom as presented in [1]_.
More specifically, we use from [1]_ the ellipsoid and tissue parameters.

In `DIRECT` are implemented three different Shepp Logan datasets:

* T1-weighted (:class:`SheppLoganT1Dataset`)
* T2-weighted (:class:`SheppLoganT2Dataset`)
* Proton density (:class:`SheppLoganT1Dataset`)

The Shepp Logan Datasets allow for data-less testing of the MRI models.

Use the :class:`SheppLoganDataset`
----------------------------------

Configuration
~~~~~~~~~~~~~

A toy configuration file for training a :class:`Unet2d` model can be found in
`https://github.com/NKI-AI/direct/blob/main/projects/toy/shepp_logan/base_unet.yaml <https://github.com/NKI-AI/direct/blob/main/projects/toy/shepp_logan/base_unet.yaml>`__.

Training
~~~~~~~~

Since data are created on the fly, there is no need to pass to the command line training or validation directory paths.
The following command might be used for training using the Shepp Logan Datasets:

.. code-block::bash

   direct train <output_folder> \
            --name <name> \
            --cfg projects/toy/shepp_logan/base_unet.yaml \
            --num-gpus <number_of_gpus> \
            --num-workers <number_of_workers> \
            [--other-flags]

References
----------

.. [1] Gach, H. Michael, Costin Tanase, and Fernando Boada. "2D & 3D Shepp-Logan phantom standards for MRI." 2008 19th International Conference on Systems Engineering. IEEE, 2008.
