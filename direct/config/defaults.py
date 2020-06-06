# coding=utf-8
# Copyright (c) DIRECT Contributors

from dataclasses import dataclass
from omegaconf import MISSING

from direct.config import BaseConfig
from direct.common.subsample_config import MaskingConfig
from direct.data.datasets_config import DatasetConfig

from typing import Optional, List


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
class ModelConfig(BaseConfig):
    pass


@dataclass
class DefaultConfig(BaseConfig):
    debug: bool = False


    model_name: str = MISSING
    # SOLVER: SolverConfig = MISSING
    model: ModelConfig = MISSING
    dataset: DatasetConfig = DatasetConfig()

    training: TrainingConfig = TrainingConfig()  # This should be optional.
    masking: MaskingConfig = MaskingConfig()

    tensorboard: TensorboardConfig = TensorboardConfig()








