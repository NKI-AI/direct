# Calgary-Campinas challenge
This folder contains the training code specific for the [Calgary Campinas challenge](https://sites.google.com/view/calgary-campinas-dataset/home/mr-reconstruction-challenge).

## Training
The standard training script `train_rim.py` in [tools/](tools) can be used. If you want, the validation volumes can be computed using `run_rim.py`.
For our submission we used [base.yaml](configs/base.yaml) for model configuration.

## Prediction
The masks are not provided for the test set, and need to be pre-computed using [compute_masks.py](compute_masks.py).
These masks should be passed to the `--masks` parameter of [predict_test.py](predict_test.py).

