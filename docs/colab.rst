.. highlight:: shell

============================
Use DIRECT with Google Colab
============================



1. First mount your Google drive in Colab and create a directory named, e.g. `DIRECT`, and `cd` there:

.. code-block:: python3
    
    %cd /content/drive/MyDrive/DIRECT/

This `notebook <https://colab.research.google.com/notebooks/io.ipynb>`_ can help with mounting your Google drive.


2. Clone the repo:

.. code-block:: python

    !git clone https://github.com/NKI-AI/direct.git

3. Copy paste and run the following

.. code-block:: python

    !wget -O mini.sh https://repo.anaconda.com/miniconda/Miniconda3-py38_4.8.2-Linux-x86_64.sh
    !chmod +x mini.sh
    !bash ./mini.sh -b -f -p /usr/local
    !conda install -q -y jupyter
    !conda install -q -y google-colab -c conda-forge
    !python -m ipykernel install --name "py38" --user

The above block is needed to install python 3.8 in Colab as it runs using Python 3.7.

4. Run the following to install the latest `PyTorch` version:

.. code-block:: python

    !pip3 uninstall torch
    !pip3 uninstall torchvision
    !pip3 uninstall torchaudio
    !pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

5.  Navigate to the repo:

.. code-block:: python

    %cd direct/

6. Install package.

.. code-block:: python

    !python3 setup.py install

OR

.. code-block:: python

    !python3 -m pip install -e ".[dev]"

7. Run experiments using the configuration files in the `projects <https://github.com/NKI-AI/direct/tree/main/projects>`_ folder,
or you can set up your own configuration files following our `template <https://docs.aiforoncology.nl/direct/config.html>`_.
