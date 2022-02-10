.. role:: raw-html-m2r(raw)
   :format: html


Model Zoo and Baselines
=======================

Introduction
------------

This file documents baselines created with the DIRECT project. You can download the parameters and weights of these
models in a ``.zip`` file by pressing on the hyperlink of the checkpoint. Each file contains the model checkpoint(s), a
configuration file ``config.yaml`` with the model parameters used to load the model for inference and validation metrics.

How to read the tables
----------------------


* "Name" refers to the name of the config file which is saved in ``projects/{project_name}/configs/{name}.yaml``
* Checkpoint is the integer representing the model weights saved in ``model_{iteration}.pt``  as that iteration.

License
-------

All models made available through this page are licensed under the\ :raw-html-m2r:`<br>`
`Creative Commons Attribution-ShareAlike 3.0 license <https://creativecommons.org/licenses/by-sa/3.0/>`_.

Baselines
---------

Calgary-Campinas MR Image Reconstruction `Challenge <https://sites.google.com/view/calgary-campinas-dataset/mr-reconstruction-challenge>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Models were trained on the Calgary-Campinas brain dataset. Training included 47 multicoil (12 coils) volumes that were either 5x or 10x accelerated by retrospectively applying masks provided by the Calgary-Campinas team.

Validation Set (12 coils, 20 Volumes)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Model
     - Name
     - Acceleration
     - Checkpoint
     - SSIM
     - pSNR
     - VIF
   * - RecurrentVarNet
     - recurrentvarnet
     - 5x
     - `148500 <https://s3.aiforoncology.nl/direct-project/recurrentvarnet.zip>`_
     - 0.943
     - 36.1
     - 0.964
   * - RecurrentVarNet
     - recurrentvarnet
     - 10x
     - `107000 <https://s3.aiforoncology.nl/direct-project/recurrentvarnet.zip>`_
     - 0.911
     - 33.0
     - 0.926
   * - LPDNet
     - lpd
     - 5x
     - `96000 <https://s3.aiforoncology.nl/direct-project/lpdnet.zip>`_
     - 0.937
     - 35.6
     - 0.953
   * - LPDNet
     - lpd
     - 10x
     - `97000 <https://s3.aiforoncology.nl/direct-project/lpdnet.zip>`_
     - 0.901
     - 32.2
     - 0.919
   * - RIM
     - rim
     - 5x
     - `89000 <https://s3.aiforoncology.nl/direct-project/rim.zip>`_
     - 0.932
     - 35.0
     - 0.964
   * - RIM
     - rim
     - 10x
     - `63000 <https://s3.aiforoncology.nl/direct-project/rim.zip>`_
     - 0.891
     - 31.7
     - 0.911
   * - VarNet
     - varnet
     - 5x
     - `4000 <https://s3.aiforoncology.nl/direct-project/varnet.zip>`_
     - 0.917
     - 33.3
     - 0.937
   * - VarNet
     - varnet
     - 10x
     - `3000 <https://s3.aiforoncology.nl/direct-project/varnet.zip>`_
     - 0.862
     - 29.9
     - 0.861
   * - Joint-ICNet
     - jointicnet
     - 5x
     - `43000 <https://s3.aiforoncology.nl/direct-project/jointicnet.zip>`_
     - 0.904
     - 32.0
     - 0.940
   * - Joint-ICNet
     - jointicnet
     - 10x
     - `42500 <https://s3.aiforoncology.nl/direct-project/jointicnet.zip>`_
     - 0.854
     - 29.4
     - 0.853
   * - XPDNet
     - xpdnet
     - 5x
     - `16000 <https://s3.aiforoncology.nl/direct-project/xpdnet.zip>`_
     - 0.907
     - 32.3
     - 0.965
   * - XPDNet
     - xpdnet
     - 10x
     - `14000 <https://s3.aiforoncology.nl/direct-project/xpdnet.zip>`_
     - 0.855
     - 29.7
     - 0.837
   * - KIKI-Net
     - kikinet
     - 5x
     - `44500 <https://s3.aiforoncology.nl/direct-project/kikinet.zip>`_
     - 0.888
     - 29.6
     - 0.919
   * - KIKI-Net
     - kikinet
     - 10x
     - `44500 <https://s3.aiforoncology.nl/direct-project/kikinet.zip>`_
     - 0.833
     - 27.5
     - 0.856
   * - MultiDomainNet
     - multidomainnet
     - 5x
     - `50000 <https://s3.aiforoncology.nl/direct-project/multidomainnet.zip>`_
     - 0.864
     - 28.7
     - 0.912
   * - MultiDomainNet
     - multidomainnet
     - 10x
     - `50000 <https://s3.aiforoncology.nl/direct-project/multidomainnet.zip>`_
     - 0.810
     - 26.8
     - 0.812
   * - U-Net
     - unet
     - 5x
     - `10000 <https://s3.aiforoncology.nl/direct-project/unet.zip>`_
     - 0.871
     - 29.5
     - 0.895
   * - U-Net
     - unet
     - 10x
     - `6000 <https://s3.aiforoncology.nl/direct-project/unet.zip>`_
     - 0.821
     - 27.8
     - 0.837

