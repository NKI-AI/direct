.. highlight:: shell

=======================
Adding your own sampler
=======================

:code:`DIRECT` currently supports sub-samplers only for gridded data (data acquired on an equispaced grid).
To add a new sub-sampler follow the steps below:

- Implement your custom sampler under :code:`direct/common/subsample.py` following the template:

.. code-block:: python

    class MyNewMaskFunc(BaseMaskFunc):
        def __init__(
            self,
            accelerations: Tuple[Number, ...],
            ...
        ):
            super().__init__(
                accelerations=accelerations,
                uniform_range=False,
            )
            ...

        def mask_func(self, shape, return_acs=False, seed=None):
            """
            Main function that outputs the sampling mask and acs_mask.

            Parameters
            ----------

            shape : iterable[int]
                The shape of the mask to be created. The shape should at least 3 dimensions.
                Samples are drawn along the second last dimension.
            seed : int (optional)
                Seed for the random number generator. Setting the seed ensures the same mask is generated
                 each time for the same shape.
            return_acs : bool
                Return the autocalibration signal region as a mask.

            Returns
            -------
            torch.Tensor : the sampling mask

            """
            if len(shape) < 3:
                raise ValueError("Shape should have 3 or more dimensions")

            with temp_seed(self.rng, seed):
                num_rows = shape[-3]
                num_cols = shape[-2]
                center_fraction, acceleration = self.choose_acceleration()

                # Create the mask of shape (1, nx, ny, 1)
                mask = ...

                if return_acs:
                    acs_mask = ...
                    return torch.from_numpy(acs_mask)
            ...

            return torch.from_numpy(mask)


Ideally, your sub-sampler should be able to initialise only with the :code:`accelerations` argument. Otherwise, update :code:`direct/common/subsample_config.py` accordingly with any new keys needed to initialise
your sub-sampler:

.. code-block:: python

    @dataclass
    class MaskingConfig(BaseConfig):
        ...
        <new_keys>: ... = ...


- To use your sub-sampler, you have to request it in the :code:`config.yaml` file. The following shows an example for training:


.. code-block:: yaml

    training:
        datasets:
        -   name: ...
            ...
            transforms:
                ...
                masking:
                    name: MyNew
                    accelerations: [...]
                    ...
