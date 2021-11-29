# Direct tools

Scripts are provided:
- To train a model use `train_model.py`.
- To extract the best checkpoint based on `metrics.json`, use `parse_metrics_log.py`.


## Tips and tricks

- We are using a lot of experimental features in pytorch, to reduce such warnings you can use
 `export PYTHONWARNINGS="ignore"` in the shell before execution.
