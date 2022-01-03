Quick Start
===========
This gives a brief quick start - introduction on how to download public datasets such as the Calgary-Campinas and FastMRI multi-coil MRI data and train models implemented in ``DIRECT``.

1. Downloading and Preparing MRI datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Multi-coil Calgary-Campinas dataset can be obtained following the instructions `here <https://sites.google.com/view/calgary-campinas-dataset/download>`_  and the FastMRI dataset can be obtained from `here <https://fastmri.org>`_ by filling in their form.
Data should be arranged into training and validation folders. The testing set is not strictly required, and definitely not during training, if you do not want to compute the
test set results.

**Note:** Preferably use a fast drive, for instance an SSD to store these files to make sure  to get the maximal performance.

2. Install ``DIRECT``
^^^^^^^^^^^^^^^^^^^^^

Follow the instructions in `installation docs <https://docs.aiforoncology.nl/direct/installation.html>`_.

3. Training and Inference
^^^^^^^^^^^^^^^^^^^^^^^^^

3.1 Preparing a configuration file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To run experiments a configuration file must be created. For a sample configuration file please refer to our `docs <https://docs.aiforoncology.nl/direct/config.html>`_.

3.2 Projects
~~~~~~~~~~~~
In the `projects folder <https://github.com/NKI-AI/direct/tree/main/projects>`_ folder you can find examples of baseline configurations for our experiments.

Instructions on how to train a model or perform inference can be found in the `docs <https://docs.aiforoncology.nl/direct/>`_.
