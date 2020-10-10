# History

## 0.2 (WORK IN PROGRESS)
Many new features have been added, of which most will likely have introduced breaking changes. Several performance
issues have been addressed.

### New features
* Added FastMRI 2020 dataset.
* Allow initialization from zero-filled or external input.
* Allow initialization from something else than the zero-filled image in `train_rim.py` by passing a directory.
* Added regularizers.
* Refactoring environment class allowing the use of different models except RIM.
* Added inference key to the configuration which sets the proper transforms to be used during training, this became
necessary when we introduced the possibility to have multiple training and validation sets, created a inference script
honoring these changes.
* Engine is now aware of the backward and forward operators, these are not passed in the environment anymore, but are
properties of the engine.
* `direct.utils.str_to_class` now supports partial argument parsing, e.g. `fft2(centered=False)` will be properly parsed
in the config.
* PyTorch 1.6 and Python 3.8 are now required.

### Work in progress
* Added a preliminary version of a 3D RIM version. This includes changing several transforms to versions being dimension
independent and also intends to support 3D + time data.

### Bugfixes
* Fixed progressive slowing down during validation by refactoring engine and turning lists of dataloaders
into a generator, also disabled memory pinning to alleviate this problem.
* Fixed a bug that when initializing from a previous checkpoint additional models were not loaded.
* Fixed normalization of the sensitivity map in `rim_engine.py`.

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
