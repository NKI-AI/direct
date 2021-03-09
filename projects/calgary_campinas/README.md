# Calgary-Campinas challenge

This folder contains the training code specific for
the [Calgary Campinas challenge](https://sites.google.com/view/calgary-campinas-dataset/home/mr-reconstruction-challenge)
. As of writing (25 Oct 2020) this is the top result in both Track 1 and Track 2.

## Training

The standard training script `train_rim.py` in [tools/](tools) can be used. If you want, the validation volumes can be
computed using `predict_val.py`. During training, validation metrics will be logged, these match the challenge metrics.
For our submission we used [base.yaml](configs/base.yaml) as model configuration.

After downloading the data to `<data_root>` a command such as the one below was used (running in the docker container,
which maps the code to `direct`):

```
cd /direct/tools
python train_rim.py <data_root>/Train/ \
                    <data_root>/Val/ \
                    <output_folder> \
                    --name base \
                    --cfg /direct/projects/calgary_campinas/configs/base.yaml \
                    --num-gpus 4 \
                    --num-workers 8 \
                    --resume
```

Additional options can be found using `python train_rim.py --help`.

## Prediction

The masks are not provided for the test set, and need to be pre-computed using [compute_masks.py](compute_masks.py).
These masks should be passed to the `--masks` parameter of [predict_test.py](predict_test.py).

