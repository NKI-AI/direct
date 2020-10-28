# History

## 0.2
Many new features have been added, of which most will likely have introduced breaking changes. Several performance
issues have been addressed.

An improved version to the winning solution for the [Calgary-Campinas challenge](https://sites.google.com/view/calgary-campinas-dataset/mr-reconstruction-challenge) is also added to v0.2, including model weights.


### New features
* Baseline model for the Calgary-Campinas challenge (see model_zoo.md)
* Added FastMRI 2020 dataset.
* Challenge metrics for FastMRI and the Calgary-Campinas.
* Allow initialization from zero-filled or external input.
* Allow initialization from something else than the zero-filled image in `train_rim.py` by passing a directory.
* Refactoring environment class allowing the use of different models except RIM.
* Added inference key to the configuration which sets the proper transforms to be used during training, this became
necessary when we introduced the possibility to have multiple training and validation sets, created a inference script
honoring these changes.
* Separate validation and testing scripts for the Calgary-Campinas challenge.

### Technical changes in functions
* `direct.utils.io.write_json` serializes real-valued numpy and torch objects.
* `direct.utils.str_to_class` now supports partial argument parsing, e.g. `fft2(centered=False)` will be properly parsed
in the config.
* Added support for regularizers.
* Engine is now aware of the backward and forward operators, these are not passed in the environment anymore, but are
properties of the engine.
* PyTorch 1.6 and Python 3.8 are now required.

### Work in progress
* Added a preliminary version of a 3D RIM version. This includes changing several transforms to versions being dimension
independent and also intends to support 3D + time data.

### Bugfixes
* Fixed progressive slowing down during validation by refactoring engine and turning lists of dataloaders
into a generator, also disabled memory pinning to alleviate this problem.
* Fixed a bug that when initializing from a previous checkpoint additional models were not loaded.
* Fixed normalization of the sensitivity map in `rim_engine.py`.
* `direct.data.samplers.BatchVolumeSampler` returned wrong length which caused dropping of volumes during validation.


## 0.1.2
* Bugfixes in FastMRI dataset class
* Improvements in logging.
* Allow the reading of subsets of the dataset by providing lists.
* Parses the ISMRMD header for matrix output size instead of having predefined values.
* Add the ability to add additional models to the engine class by configuration.


## 0.1.1
* Mixed precision support (based on PyTorch 1.6).
* Bugfixes in 0.1.

## 0.1
* Configurable loss, metrics.
* Updated logging.
* Fixed several bugs.

## 0.0.1
* First released version on GitHub.
