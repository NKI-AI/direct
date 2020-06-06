# Using DIRECT

This document gives a brief introduction on how to train a Recurrent Inference Machine on the single coil
FastMRI knee dataset.

- For general information about DIRECT, please see [`README.md`](README.md).
- For installation instructions for DIRECT, please see [`install.md`](install.md).

## Notebooks
Example [notebooks](notebooks) are provided.
- [FastMRIDataset](notebooks/FastMRIDataset.ipynb): in this notebook the functionality of the `FastMRIDataset` class is
described.


## Training
### 1. Prepare dataset
The dataset can be obtained from https://fastmri.org by filling in their form, download the singlecoil knee train and validation
 data using the `curl` command they provide in the e-mail you will receive. Unzip the files using:

```shell
tar xvf singlecoil_train.tar.gz
tar xfv singlecoil_val.tar.gz
```
The testing set is not strictly required, and definitely not during training, if you do not want to compute the
test set results.

**Note:** Preferably use a fast drive, for instance an SSD to store these files to make sure  to get the maximal performance.

#### 1.1 Generate metadata
As you will likely train several models on the same dataset it might be convenient to compile a dataset description.

**TODO:** Add dataset description.


### 2. Build docker engine
Follow the instructions in the [docker](docker) subfolder, and make sure to mount the data and output directory
(using `--volume`).
