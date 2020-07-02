# coding=utf-8
# Copyright (c) DIRECT Contributors

from dataclasses import dataclass, field
from omegaconf import MISSING

from direct.config import BaseConfig
from direct.data.datasets_config import DatasetConfig

from typing import Optional, List, Callable


@dataclass
class TensorboardConfig(BaseConfig):
    num_images: int = 8


@dataclass
class CheckpointerConfig(BaseConfig):
    checkpoint_steps: int = 500


@dataclass
class LossConfig(BaseConfig):
    crop: List[int] = (0, 0)


@dataclass
class TrainingConfig(BaseConfig):
    # Dataset
    dataset: DatasetConfig = DatasetConfig()

    # Optimizer
    optimizer: str = 'Adam'
    lr: float = 5e-4
    weight_decay: float = 1e-6
    batch_size: int = 2

    # LR Scheduler
    lr_step_size: int = 5000
    lr_gamma: float = 0.5
    lr_warmup_iter: int = 500

    num_iterations: int = 50000

    # Validation
    validation_steps: int = 1000

    # Gradient
    gradient_steps: int = 1
    gradient_clipping: float = 0.0
    gradient_debug: bool = False

    # Loss
    loss: LossConfig = LossConfig()

    # Checkpointer
    checkpointer: CheckpointerConfig = CheckpointerConfig()


@dataclass
class ValidationConfig(BaseConfig):
    datasets: List[DatasetConfig] = field(default_factory=lambda: [DatasetConfig()])


@dataclass
class ModelConfig(BaseConfig):
    pass


@dataclass
class ModalityConfig(BaseConfig):
    forward_operator: str = 'fft2'
    backward_operator: str = 'ifft2'


@dataclass
class DefaultConfig(BaseConfig):
    debug: bool = False

    model_name: str = MISSING
    model: ModelConfig = MISSING

    modality: ModalityConfig = ModalityConfig()

    training: TrainingConfig = TrainingConfig()  # This should be optional.
    validation: ValidationConfig = ValidationConfig()  # This should be optional.

    tensorboard: TensorboardConfig = TensorboardConfig()
