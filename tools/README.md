# Direct tools

Scripts are provided:
- To train and test a recurrent inference machine (`train_rim.py` and `val_rim.py`).
The latter script predicts the output based on the validation set as given in the config.
- To extract the best checkpoint based on `metrics.json`, use `parse_metrics_log.py`.


## Tips and tricks

- We are using a lot of experimental features in pytorch, to reduce such warnings you can use
 `export PYTHONWARNINGS="ignore"` in the shell before execution.
