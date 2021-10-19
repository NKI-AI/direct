# Calgary-Campinas challenge
This folder contains the training code specific for the [Calgary Campinas challenge](https://sites.google.com/view/calgary-campinas-dataset/mr-reconstruction-challenge).

## Training
The standard training script `train_model.py` in [tools/](tools) can be used. Training model configurations can be found in the [configs/](configs) folder.  

During training, training loss, validation metrics and validation image predictions are logged. Additionally, [Tensorboard](#tensorboard) allows for visualization of the above. 

After downloading the data to `<data_root>` a command such as the one below is used to train a `<model_name>` (running in the docker container, which maps the code to `direct`):
```
cd /direct/tools
python3 train_model.py <data_root>/Train/ \
                    <data_root>/Val/ \
                    <output_folder> \
                    --name <name> \
                    --cfg /direct/projects/calgary_campinas/configs/base_<model_name>.yaml \
                    --num-gpus 4 \
                    --num-workers 8 \
                    --resume
```
Additional options can be found using `python train_model.py --help`.

The validation volumes can be computed using `predict_val.py` (see [Validation](#validation)).

For our submission in the challenge we used [base_rim.yaml](configs/base_rim.yaml) as the model configuration. As of writing (October 2021) this is the top result in both Track 1 and Track 2.

## Inference

### Validation
To make predictions on validation data a command such as the one below is used to perform inference on dataset with index `<dataset_validation_index>` as various datasets can be defined in the training configuration file.
```
cd projects/calgary_campinas
python3 predict_val.py <data_root>/Val/ \
                    <output_directory> \
                    <experiment_directory_containing_checkpoint> \
                    --checkpoint <checkpoint> \
                    --validation-index <dataset_validation_index> \
                    --name <name> \
                    --num-gpus 4 \
                    --num-workers 8 \
```
### Test 
The masks are not provided for the test set, and need to be pre-computed using [compute_masks.py](compute_masks.py).
These masks should be passed to the `--masks` parameter of [predict_test.py](predict_test.py).

##  Tensorboard
To visualize training and validation metrics of an experiment in Tensorboard on your local machine run
```
tensorboard --logdir <path_to_experiment> --port <port_id> . 
```
If you are working on a remote host and want to visualize the experiment on your local machine run 

 1. ```tensorboard --logdir <path_to_experiment> --port <remote_port_id>``` on the remote host, and

 2. ```ssh -N -f -L localhost:<local_port_id>:localhost:<remote_port_id> <user@remote_host>``` on your local machine.

 3. Navigate to [http://localhost:<local_port_id>](http://localhost:local_port_id) on your local machine.

![direct_tensorboard](https://user-images.githubusercontent.com/71031687/137918503-84b894e4-b9db-42cd-8e94-03bb098171fa.gif)
