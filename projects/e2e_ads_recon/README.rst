E2E-ADS-Recon experiments
=========================

Configs for reproducing experiments from:

`End-to-End Co-Optimization of Adaptive k-space Sampling and Reconstruction for Dynamic MRI <https://proceedings.mlr.press/v315/yiasemis26a.html>`_
(MIDL 2026 / PMLR 315).

These YAML files were collected from the original experiment directories
(``kosmos:/projects/mri_adaptive_sampling/e2e_ads_recon`` and
``kosmos:/projects/direct/adpt``). Paths to data lists and root directories
inside each experiment ``.yaml`` likely need to be updated for your machine.

What this enables
-----------------

Adaptive (and dynamic adaptive) :math:`k`-space sampling is **not** hard-coded
to a single reconstruction model. Enable it for any reconstruction model by
adding an ``additional_models.sampling_model`` block, for example:

.. code-block:: yaml

   additional_models:
     sampling_model:
       model_name: adaptive.policy.StraightThroughPolicy
       sampling_dimension: ONE_D   # or TWO_D
       sampling_type: STATIC       # or DYNAMIC_2D / DYNAMIC_2D_NON_UNIFORM
       kspace_shape: [512, 246]
       # for DYNAMIC_*:
       # num_time_steps: 11

Use ``sampling_type: STATIC`` with 2D reconstruction models and
``DYNAMIC_2D`` / ``DYNAMIC_2D_NON_UNIFORM`` with 3D / dynamic models.
The engine applies the sampler via ``MRIModelEngine.perform_sampling`` when
``sampling_model`` is present.


Naming
------

Configs use a short scheme: ``{recon}_{sampler}_{mode}_{extras}.yaml``.

* ``vsharp`` / ``varnet`` — reconstruction model
* ``ads`` — straight-through adaptive sampler; ``loupe`` — parameterized; ``fixed`` — non-adaptive mask
* ``1d`` / ``2d`` — sampling dimension
* ``dyn`` — dynamic sampling / masking; omit for static
* ``init`` / ``init2`` — ACS/target-acceleration init variants
* ``kspace`` — stronger k-space supervision setup
* ``kt`` / ``radial`` / ``gauss`` — mask family

Typical training command
------------------------

.. code-block:: bash

   direct train <experiment_dir> \
     --cfg projects/e2e_ads_recon/<experiment_name>.yaml \
     --num-gpus <N>

Replace dataset roots / list files in the YAML before launching.
