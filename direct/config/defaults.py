# Copyright 2025 AI for Oncology Research Group. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""direct.config.defaults module."""

from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING

from direct.config import BaseConfig
from direct.data.datasets_config import DatasetConfig


@dataclass
class TensorboardConfig(BaseConfig):
    """TensorboardConfig."""

    num_images: int = 8


@dataclass
class LoggingConfig(BaseConfig):
    """LoggingConfig."""

    log_as_image: list[str] | None = None
    # How often (in iterations) to flush scalars / write TensorBoard. Default: 20.
    log_interval: int = 20
    tensorboard: TensorboardConfig = field(default_factory=TensorboardConfig)


@dataclass
class FunctionConfig(BaseConfig):
    """FunctionConfig."""

    function: str = MISSING
    multiplier: float = 1.0
    # Optional tensor keys for loss comparison. When omitted, defaults are inferred
    # from ``function`` (image → output_image/target, kspace → output_kspace/kspace,
    # displacement_field → displacement_field/displacement_field).
    source_key: str | None = None
    target_key: str | None = None


@dataclass
class CheckpointerConfig(BaseConfig):
    """CheckpointerConfig."""

    checkpoint_steps: int = 500
    # Keep only the newest N ``model_*.pt`` files on disk (plus ``last_model.txt``).
    # ``None`` / ``0`` keeps all checkpoints. Default is ``None``.
    max_to_keep: int | None = None


@dataclass
class LossConfig(BaseConfig):
    """LossConfig."""

    crop: str | None = None
    losses: list[Any] = field(default_factory=lambda: [FunctionConfig()])


@dataclass
class TrainingConfig(BaseConfig):
    # Dataset
    """TrainingConfig."""

    datasets: list[Any] = field(default_factory=lambda: [DatasetConfig()])

    # model_checkpoint gives the checkpoint from which we can load the *model* weights.
    model_checkpoint: str | None = None

    # Optimizer
    optimizer: str = "Adam"
    lr: float = 5e-4
    weight_decay: float = 1e-6
    batch_size: int = 2

    # LR Scheduler
    lr_step_size: int = 5000
    lr_gamma: float = 0.5
    lr_warmup_iter: int = 500

    # Stochastic weight averaging
    swa_start_iter: int | None = None

    num_iterations: int = 50000

    # Validation
    validation_steps: int = 1000

    # Gradient
    gradient_steps: int = 1
    gradient_clipping: float = 0.0
    gradient_debug: bool = False

    # Loss
    loss: LossConfig = field(default_factory=LossConfig)

    # Checkpointer
    checkpointer: CheckpointerConfig = field(default_factory=CheckpointerConfig)

    # Metrics
    metrics: list[str] = field(default_factory=list)

    # Regularizers
    regularizers: list[str] = field(default_factory=list)


@dataclass
class ValidationConfig(BaseConfig):
    """ValidationConfig."""

    datasets: list[Any] = field(default_factory=lambda: [DatasetConfig()])
    batch_size: int = 8
    metrics: list[str] = field(default_factory=list)
    regularizers: list[str] = field(default_factory=list)
    crop: str | None = "training"


@dataclass
class CoilSensitivitySimulationConfig(BaseConfig):
    """Simulated receive-coil maps used when synthesizing k-space at predict time.

    ``mode: acs`` keeps maps estimated from the volume (training-style).
    ``birdcage``, ``surface``, and ``biot_savart`` replace those with the
    physical coil simulator. ``empirical`` resizes/compresses ACS maps.
    """

    mode: str = "acs"
    num_coils: int | None = None
    coil_radius: float = 1.5
    coil_size: float = 0.55
    falloff_power: float = 1.5
    phase_strength: float = 0.7
    angular_jitter: float = 0.08
    biot_savart_segments: int = 96
    seed: int | None = 0
    normalize: bool = True


@dataclass
class InferenceConfig(BaseConfig):
    """InferenceConfig."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    batch_size: int = 1
    metrics: list[str] = field(default_factory=list)
    crop: str | None = None
    coil_sensitivity: CoilSensitivitySimulationConfig = field(default_factory=CoilSensitivitySimulationConfig)


@dataclass
class ModelConfig(BaseConfig):
    """ModelConfig."""

    model_name: str = MISSING
    engine_name: str | None = None


@dataclass
class PhysicsConfig(BaseConfig):
    """PhysicsConfig."""

    forward_operator: str = "fft2"
    backward_operator: str = "ifft2"
    use_noise_matrix: bool = False
    noise_matrix_scaling: float | None = 1.0


@dataclass
class DefaultConfig(BaseConfig):
    """DefaultConfig."""

    model: ModelConfig = MISSING
    additional_models: Any | None = None

    physics: PhysicsConfig = field(default_factory=PhysicsConfig)

    # Optional so inference-only YAMLs need not declare training/validation.
    training: TrainingConfig | None = None
    validation: ValidationConfig | None = None

    inference: InferenceConfig | None = None

    logging: LoggingConfig = field(default_factory=LoggingConfig)
