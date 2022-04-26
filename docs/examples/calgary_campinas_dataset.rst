=======================================
Setting up the Calgary-Campinas Dataset
=======================================

Imports
-------

.. code:: ipython3

  from functools import partial

  import numpy as np
  import matplotlib.pyplot as plt


  from direct.data.datasets import CalgaryCampinasDataset
  from direct.data.mri_transforms import build_mri_transforms
  from direct.data.transforms import fft2, ifft2, modulus, root_sum_of_squares
  from direct.common.subsample import CalgaryCampinasMaskFunc

Define forward and backward transforms
--------------------------------------


.. code:: ipython3

  forward_operator = partial(fft2, dim=(1,2), centered=False)
  backward_operator = partial(ifft2, dim=(1,2), centered=False)

Define sub-sampling function
----------------------------

.. code:: ipython3

  mask_func = CalgaryCampinasMaskFunc(accelerations=[5, 10])

Define MRI transforms
---------------------

.. code:: ipython3

  transforms = build_mri_transforms(
      forward_operator=forward_operator,
      backward_operator=backward_operator,
      mask_func=mask_func,
      scaling_key="masked_kspace",
  )

Create the Dataset
------------------

.. code:: ipython3

  # root pointing to the Calgary-Campinas data directory
  root = "<...>"
  dataset = CalgaryCampinasDataset(
      root = root,
      transform=transforms,
      crop_outer_slices=True,
  )

Investigate a sample
--------------------

.. code:: ipython3

  sample = dataset[20]

  print("Sample keys: ", sample.keys())

Visualizations
~~~~~~~~~~~~~~

.. code:: ipython3

  plt.imshow(sample["target"], cmap='gray')
  plt.axis("off")
  plt.title("Target")
  plt.show()

  plt.imshow(sample["sampling_mask"].squeeze(), cmap='gray')
  plt.axis("off")
  plt.title("Sub-Sampling Mask")
  plt.show()

.. code:: ipython3

  masked_kspace = sample["masked_kspace"]
  print("K-space shape: (Coils, Height, Width, Complex) = ", tuple(masked_kspace.shape))

  backprojected_kspace = backward_operator(masked_kspace)

  fig, axs = plt.subplots(1, masked_kspace.shape[0], figsize=(20,50))

  for i in range(masked_kspace.shape[0]):
      axs[i].imshow(modulus(backprojected_kspace[i]))
      axs[i].axis("off")
      axs[i].set_title(f"Coil {i+1}")
  plt.show()
