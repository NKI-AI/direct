.. highlight:: shell

===============
Config Template
===============

.. code-block:: yaml
  
  model:
  model_name: <nn_model_path>
  model_parameter_1: <nn_model_paramter_1>
  model_parameter_2: <nn_model_paramter_2>
  ...
  
  additional_models:
    sensitivity_model:
      model_name: <nn_sensitivity_model_path>
      ...

  physics:
    forward_operator: fft2(centered=False)
    backward_operator: ifft2(centered=False)
    ...

  training:
    datasets:
    - name: Dataset1
      lists:
      - <path_to_list_1_for_Dataset1>
      - <path_to_list_2_for_Dataset1>
      transforms:
        estimate_sensitivity_maps: <true_or_false>
        scaling_key: <scaling_key>
        image_center_crop: <true_or_false>
        masking:
          name: MaskingFunctionName
          accelerations: [acceleration_1, accelaration_2, ...]
          ...
      ...
    - name: Dataset2
      lists:
      ...
      transforms:
        ...
        masking:
          name: MaskingFunctionName
          accelerations: [acceleration_1, accelaration_2, ...]
          ...
      ...
    optimizer: <optimizer>
    lr: <learning_rate>
    batch_size: <batch_size>
    lr_step_size: <lr_step_size>
    lr_gamma: <lr_gamma>
    lr_warmup_iter: <num_warmup_iterations>
    num_iterations: <num_iterations>
    validation_steps: <num_val_steps>
    loss:
      losses:
      - function: <fun1_as_in_model_engine>
        multiplier: <multiplier_1>
      - function: <fun2_as_in_model_engine>
        multiplier: <multiplier_2>
    checkpointer:
      checkpoint_steps: <num_checkpointer_steps>
    metrics: [<metric_1, metric_2, ...]
    ...

  validation:
    datasets:
    - name: ValDataset1
      transforms:
        ...
        masking:
          ...
      text_description: <val_description_1>
      ...
    - name: ValDataset2
      transforms:
        ...
        masking:
          ...
        text_description: <val_description_2>
        ...
    - name: ...
    ...
    batch_size: <val_batch_size>
    metrics:
    - val_metric_1
    - val_metric_2
    - ...
    ...

  inference:
    dataset:
      name: InferenceDataset
      lists: ...
      transforms:
        masking:
          ...
        ...
      text_description: >inference_description>
      ...
    batch_size: <batch_size>
    ...

  logging:
    tensorboard:
      num_images: <num_images>
