# coding=utf-8
# Copyright (c) DIRECT Contributors
import logging
import pathlib
import sys
import torch
import signal
import direct
import numpy as np
import warnings

from typing import Optional, Dict, Tuple, List, Union
from abc import abstractmethod, ABC

from torch import nn
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.cuda.amp import GradScaler


from direct.data.mri_transforms import AddNames
from direct.data import sampler
from direct.data.datasets import ConcatDataset
from direct.data.sampler import ConcatDatasetBatchSampler
from direct.checkpointer import Checkpointer
from direct.utils.collate import named_collate
from direct.utils import (
    communication,
    prefix_dict_keys,
    evaluate_dict,
    normalize_image,
    str_to_class,
    reduce_list_of_dicts,
)
from direct.data.bbox import crop_to_largest
from direct.utils.io import write_json
from direct.utils.events import (
    get_event_storage,
    EventStorage,
    JSONWriter,
    CommonMetricPrinter,
    TensorboardWriter,
)
from direct.data import transforms
from direct.config.defaults import BaseConfig
from direct.exceptions import ProcessKilledException
from direct.types import PathOrString

from torchvision.utils import make_grid


class Engine(ABC):
    def __init__(
        self,
        cfg: BaseConfig,
        model: nn.Module,
        device: int,
        mixed_precision: bool = False,
        **models: Dict[str, nn.Module],
    ):
        self.logger = logging.getLogger(type(self).__name__)

        self.cfg = cfg
        self.model = model
        self.models = models
        self.device = device

        self.mixed_precision = mixed_precision
        self.checkpointer = None

        # TODO: This might not be needed, if these objects are changed in-place
        self.__optimizer = None
        self.__lr_scheduler = None
        self._scaler = GradScaler(enabled=self.mixed_precision)
        self.__writers = None
        self.__bind_sigint_signal()

        self._ndim = None

    @property
    def real_names(self):
        if self.ndim == 2:
            return ["batch", "height", "width"]
        elif self.ndim == 3:
            return ["batch", "slice", "height", "width"]
        else:
            raise NotImplementedError(f"{self.ndim}D named data is not yet supported")

    @property
    def complex_names(self):
        if self.ndim == 2:
            return ["batch", "complex", "height", "width"]
        elif self.ndim == 3:
            return ["batch", "complex", "slice", "height", "width"]
        else:
            raise NotImplementedError(f"{self.ndim}D named data is not yet supported")

    @property
    def complex_names_complex_last(self):
        if self.ndim == 2:
            return ["batch", "height", "width", "complex"]
        elif self.ndim == 3:
            return ["batch", "slice", "height", "width", "complex"]
        else:
            raise NotImplementedError(f"{self.ndim}D named data is not yet supported")

    @property
    def ndim(self):
        if not self._ndim:
            raise ValueError(f"ndim needs to be set before it can be called.")

        return self._ndim

    @ndim.setter
    def ndim(self, ndim):
        if not isinstance(ndim, int) or ndim <= 0:
            raise ValueError(
                f"ndim has to be an integer larger than 0. Got {ndim}. "
                f"You can for instance use `compute_dimensionality_from_sample` "
                f"to determine the dimensionality or set it explicitly."
            )

        self._ndim = ndim

    @staticmethod
    def compute_dimensionality_from_sample(sample):
        """
        Computes the dimensionality of the data from a single sample by looking at all the names in the tensor dict
        and finding the tensor with the largest dimension in the dict.

        Parameters
        ----------
        sample : dict

        Returns
        -------
        int : the sample dimensionality
        """
        relevant_names = ["slice", "height", "width", "time"]
        ndim = 0
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                curr_dimensionality = sum([_ in v.names for _ in relevant_names])
                ndim = max(curr_dimensionality, ndim)

        return ndim

    @abstractmethod
    def build_loss(self) -> Dict:
        pass

    @staticmethod
    def build_metrics(metrics_list) -> Dict:
        if not metrics_list:
            return {}

        # _metric is added as only keys containining loss or metric are logged.
        metrics_dict = {
            curr_metric + "_metric": str_to_class("direct.functionals", curr_metric)
            for curr_metric in metrics_list
        }
        return metrics_dict

    @abstractmethod
    def _do_iteration(self, *args, **kwargs) -> Tuple[torch.Tensor, Dict]:
        """
        This is a placeholder for the iteration function. This needs to perform the backward pass.
        If using mixed-precision you need to implement `autocast` as well in this function.
        It is recommended you raise an error if `self.mixed_precision` is true but mixed precision is not available.
        """
        pass

    @torch.no_grad()
    def predict(
        self,
        dataset: Dataset,
        experiment_directory: pathlib.Path,
        checkpoint_number: int = -1,
        num_workers: int = 6,
    ) -> np.ndarray:
        # TODO: Improve the way of checkpointing
        self.logger.info(f"Predicting...")
        self.checkpointer = Checkpointer(
            self.model, experiment_directory, save_to_disk=False
        )

        # Do not load again if we already have loaded the checkpoint.
        if self.checkpointer.checkpoint_loaded is not checkpoint_number:
            self.checkpointer.load(
                iteration=checkpoint_number, checkpointable_objects=None
            )

        batch_sampler = self.build_batch_sampler(
            dataset,
            batch_size=self.cfg.validation.batch_size,
            sampler_type="sequential",
            limit_number_of_volumes=None,
        )
        # TODO: Batch size can be much larger, perhaps have a different batch size during evaluation.
        data_loader = self.build_loader(
            dataset, batch_sampler=batch_sampler, num_workers=num_workers
        )
        loss, output = self.evaluate(
            data_loader, loss_fns=None, crop=None, is_validation_process=False
        )

        return output

    @staticmethod
    def build_loader(
        dataset: Dataset,
        batch_sampler: Optional[Sampler] = None,
        num_workers: int = 6,
    ) -> DataLoader:
        # TODO(jt): Custom memory pinning.
        loader = DataLoader(
            dataset=dataset,
            sampler=None,
            batch_size=1,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=named_collate,
            drop_last=False,
            shuffle=False,
            pin_memory=True,
        )

        return loader

    @staticmethod
    def build_batch_sampler(
        dataset: Union[Dataset, List[Dataset]],
        batch_size: int,
        sampler_type: str,
        **kwargs,
    ) -> Sampler:
        if sampler_type == "random":
            if not isinstance(dataset, List) or any(
                [not isinstance(_, Dataset) for _ in dataset]
            ):
                raise ValueError(
                    f"Random sampler requires a list of datasets as input."
                )
            batch_sampler = ConcatDatasetBatchSampler(
                datasets=dataset, batch_size=batch_size
            )
        elif sampler_type == "sequential":
            sampler = direct.data.sampler.DistributedSequentialSampler(
                dataset, **kwargs
            )
            batch_sampler = direct.data.sampler.BatchVolumeSampler(
                sampler,
                batch_size=batch_size,
            )
        else:
            raise ValueError(f"Sampler type {sampler_type} not supported.")

        return batch_sampler

    def training_loop(
        self,
        data_loader: DataLoader,
        start_iter: int,
        validation_data_loaders: Optional[List[DataLoader]] = None,
        experiment_directory: Optional[pathlib.Path] = None,
    ):
        self.logger.info(f"Local rank: {communication.get_local_rank()}.")
        self.models_training_mode()

        loss_fns = self.build_loss()
        metric_fns = self.build_metrics(self.cfg.training.metrics)
        storage = get_event_storage()

        total_iter = self.cfg.training.num_iterations  # noqa
        for data, iter_idx in zip(data_loader, range(start_iter, total_iter)):
            data = AddNames()(data)
            if iter_idx == start_iter:
                self.ndim = self.compute_dimensionality_from_sample(data)
                self.logger.info(f"Data dimensionality: {self.ndim}.")

            if iter_idx == 0:
                self.log_first_training_example(data)

            try:
                output, loss_dict = self._do_iteration(data, loss_fns)
            except ProcessKilledException as e:
                # If the process is killed, the output if saved at state iter_idx, which is the current state,
                # so the computation can restart from the last iteration.
                self.logger.exception(f"Exiting with exception: {e}.")
                self.checkpointer.save(iter_idx)  # Save checkpoint at kill. # noqa
                self.write_to_logs()  # TODO: This causes the issue that current metrics are not written,
                # and you end up with an empty line.
                sys.exit(-1)

            # Gradient accumulation
            if (iter_idx + 1) % self.cfg.training.gradient_steps == 0:  # type: ignore
                if self.cfg.training.gradient_steps > 1:  # type: ignore
                    for parameter in self.model.parameters():
                        if parameter.grad is not None:
                            # In-place division
                            parameter.grad.div_(self.cfg.training.gradient_steps)  # type: ignore
                if self.cfg.training.gradient_clipping > 0.0:  # type: ignore
                    self._scaler.unscale_(self.__optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.training.gradient_clipping
                    )

                # Gradient norm
                if self.cfg.training.gradient_debug:  # type: ignore
                    warnings.warn(
                        f"Gradient debug set. This will affect training performance. Only use for debugging."
                        f"This message will only be displayed once."
                    )
                    parameters = list(
                        filter(lambda p: p.grad is not None, self.model.parameters())
                    )
                    gradient_norm = sum(
                        [parameter.grad.data ** 2 for parameter in parameters]
                    ).sqrt()  # typing: ignore
                    storage.add_scalar("train/gradient_norm", gradient_norm)

                # Same as self.__optimizer.step() for mixed precision.
                self._scaler.step(self.__optimizer)
                # Updates the scale for next iteration.
                self._scaler.update()

            # Incorrect inference by mypy and pyflake
            self.__lr_scheduler.step()  # type: ignore # noqa
            storage.add_scalar(
                "lr", self.__optimizer.param_groups[0]["lr"], smoothing_hint=False
            )

            self.__optimizer.zero_grad()  # type: ignore

            # Reduce the loss over all devices
            loss_dict_reduced = communication.reduce_tensor_dict(loss_dict)
            loss_reduced = sum(loss_dict_reduced.values())

            metrics_dict = evaluate_dict(
                metric_fns,
                transforms.modulus_if_complex(output.detach()).rename(None),
                data["target"].rename(None).detach().to(self.device),
                reduction="mean",
            )
            metrics_dict_reduced = (
                communication.reduce_tensor_dict(metrics_dict) if metrics_dict else {}
            )
            storage.add_scalars(
                loss=loss_reduced, **loss_dict_reduced, **metrics_dict_reduced
            )

            if iter_idx > 5 and (
                iter_idx % self.cfg.training.checkpointer.checkpoint_steps == 0
                or (iter_idx + 1) == total_iter
            ):
                self.logger.info(f"Checkpointing at iteration {iter_idx}.")
                self.checkpointer.save(iter_idx)

            if (
                validation_data_loaders is not None
                and iter_idx > 5
                and (
                    iter_idx % self.cfg.training.validation_steps == 0
                    or (iter_idx + 1) == total_iter
                )
            ):
                for (
                    curr_dataset_name,
                    curr_validation_data_loader,
                ) in validation_data_loaders:
                    self.logger.info(
                        f"Evaluating: {curr_dataset_name}..."
                    )  # TODO(jt): Fix with better names and stuff.
                    (
                        curr_val_loss_dict,
                        curr_val_metric_dict_per_case,
                        visualize_slices,
                        visualize_target,
                    ) = self.evaluate(
                        curr_validation_data_loader,
                        loss_fns,
                        is_validation_process=True,
                    )

                    if experiment_directory:
                        # Make dictionary serializable for logging
                        serializable_val_metric_dict = {
                            k0: {k1: float(v1) for k1, v1 in v0.items()}
                            for k0, v0 in curr_val_metric_dict_per_case.items()
                        }
                        write_json(
                            experiment_directory
                            / f"metrics_val_{curr_dataset_name}_{iter_idx}.json",
                            serializable_val_metric_dict,
                        )

                    # Metric dict still needs to be reduced as it gives values *per* data
                    curr_val_metric_dict = reduce_list_of_dicts(
                        list(curr_val_metric_dict_per_case.values()), mode="average"
                    )

                    key_prefix = (
                        "val/" if not curr_dataset_name else f"val/{curr_dataset_name}/"
                    )
                    val_loss_reduced = sum(curr_val_loss_dict.values())
                    storage.add_scalars(
                        **{key_prefix + "loss": val_loss_reduced},
                        **{
                            **prefix_dict_keys(curr_val_metric_dict, key_prefix),
                            **prefix_dict_keys(curr_val_loss_dict, key_prefix),
                        },
                        smoothing_hint=False,
                    )
                    visualize_slices = self.process_slices_for_visualization(
                        visualize_slices, visualize_target
                    )
                    storage.add_image(f"{key_prefix}prediction", visualize_slices)

                    if iter_idx // self.cfg.training.validation_steps - 1 == 0:
                        visualize_target = make_grid(
                            crop_to_largest(visualize_target, pad_value=0),
                            nrow=self.cfg.tensorboard.num_images,
                            scale_each=True,
                        )
                        storage.add_image(f"{key_prefix}target", visualize_target)

                self.logger.info(
                    f"Done evaluation of {curr_dataset_name} at iteration {iter_idx}."
                )
                self.model.train()

            # Log every 20 iterations, or at a validation step or at the end of training.
            if iter_idx > 5 and (
                iter_idx % 20 == 0
                or iter_idx % self.cfg.training.validation_steps == 0
                or (iter_idx + 1) == total_iter
            ):
                self.write_to_logs()

            storage.step()

    def process_slices_for_visualization(self, visualize_slices, visualize_target):
        # Log slices.
        # Compute the difference as well, and normalize for visualization
        difference_slices = [a - b for a, b in zip(visualize_slices, visualize_target)]
        # Normalize slices
        difference_slices = [(d / np.abs(d)) * 0.5 + 0.5 for d in difference_slices]
        visualize_slices = [normalize_image(image) for image in visualize_slices]

        # Visualize slices, and crop to the largest volume
        visualize_slices = make_grid(
            crop_to_largest(visualize_slices + difference_slices, pad_value=0),
            nrow=self.cfg.tensorboard.num_images,
            scale_each=False,
        )
        return visualize_slices

    def models_training_mode(self):
        self.model.train()
        for curr_model in self.models:
            self.models[curr_model].train()

    def models_validation_mode(self):
        self.model.eval()
        for curr_model in self.models:
            self.models[curr_model].eval()

    def models_to_device(self):
        self.model = self.model.to(self.device)
        for curr_model_name in self.models:
            self.models[curr_model_name] = self.models[curr_model_name].to(self.device)

    def train(
        self,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,  # noqa
        training_datasets: List[Dataset],
        experiment_directory: pathlib.Path,
        validation_data: Optional[Dataset] = None,
        resume: bool = False,
        initialization: Optional[PathOrString] = None,
        num_workers: int = 6,
    ) -> None:
        self.logger.info("Starting training.")
        # TODO: Does not need to be member of self.
        self.__optimizer = optimizer
        # TODO: Optimizer and LR scheduler need to be resumed too.
        self.__lr_scheduler = lr_scheduler
        training_data = ConcatDataset(training_datasets)
        self.logger.info(f"Concatenated dataset length: {len(training_data)}.")
        self.logger.info(
            f"Building batch sampler for training set with batch size {self.cfg.training.batch_size}."
        )
        training_sampler = self.build_batch_sampler(
            training_datasets, self.cfg.training.batch_size, "random"
        )
        training_loader = self.build_loader(
            training_data,
            batch_sampler=training_sampler,
            num_workers=num_workers,
        )

        if validation_data:
            validation_loaders = []
            for idx, curr_validation_data in enumerate(validation_data):
                text_dataset_description = curr_validation_data.text_description
                self.logger.info(
                    f"Building dataloader for dataset: {text_dataset_description}."
                )
                curr_batch_sampler = self.build_batch_sampler(
                    curr_validation_data,
                    batch_size=self.cfg.validation.batch_size,
                    sampler_type="sequential",
                    limit_number_of_volumes=None,
                )
                validation_loaders.append(
                    (
                        text_dataset_description,
                        self.build_loader(
                            curr_validation_data,
                            batch_sampler=curr_batch_sampler,
                            num_workers=0,  # num_workers, # TODO(jt): This seems to choke the validation.
                        ),
                    )
                )
        else:
            validation_loaders = None

        self.models_to_device()

        # Optimizer
        self.__optimizer.zero_grad()  # type: ignore

        # Mixed precision setup. This requires the model to be on the gpu.
        git_hash = direct.utils.git_hash()
        extra_checkpointing = {
            "__author__": git_hash if git_hash else "N/A",
            "__version__": direct.__version__,
            "__mixed_precision__": self.mixed_precision,
        }
        if self.mixed_precision:
            # TODO(jt): Check if on GPU
            self.logger.info(f"Using mixed precision training.")

        self.checkpointer = Checkpointer(
            self.model,
            experiment_directory,
            save_to_disk=communication.is_main_process(),
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            scaler=self._scaler,
            **self.models,
            **extra_checkpointing,
        )

        # Load checkpoint
        start_iter = 0
        checkpoint = {}
        if resume:
            self.logger.info("Attempting to resume...")
            # This changes the model inplace
            checkpoint = self.checkpointer.load(iteration="latest")
            if not checkpoint:
                self.logger.info("No checkpoint found. Starting from scratch.")
            else:
                start_iter = checkpoint["iteration"] + 1
                self.logger.info(f"Starting from iteration: {start_iter}.")

        if start_iter > 0 and initialization:
            self.logger.warning(
                f"Initialization checkpoint set to {initialization},"
                f" but model will resume training from previous checkpoint. Initialization ignored."
            )
        elif initialization:
            self.logger.info(f"Initializing from {initialization}...")
            self.checkpointer.load_from_file(initialization)

        if "__version__" in checkpoint:
            self.logger.info(
                f"DIRECT version of checkpoint: {checkpoint['__version__']}."
            )
            if checkpoint["__version__"] != direct.__version__:
                self.logger.warning(
                    f"Current DIRECT version {direct.__version__} is different from the one "
                    f"this checkpoint is saved with ({checkpoint['__version__']}. This can be fine, "
                    f"but beware that this can be a source of confusion."
                )

        if "__author__" in checkpoint:
            self.logger.info(f"Git hash of checkpoint: {checkpoint['__author__']}.")
            if checkpoint["__author__"] != direct.utils.git_hash():
                self.logger.warning(
                    f"Current git hash {direct.utils.git_hash()} is different from the one "
                    f"this checkpoint is saved with ({checkpoint['__author__']}. This can be fine, "
                    f"but beware that this can be a source of confusion."
                )

        if "__datetime__" in checkpoint:
            self.logger.info(f"Checkpoint created at: {checkpoint['__datetime__']}.")

        if "__mixed_precision__" in checkpoint:
            if (not self.mixed_precision) and checkpoint["__mixed_precision__"]:
                self.logger.warning(
                    f"Mixed precision training is not enabled, yet saved checkpoint requests this"
                    f"Will now enable mixed precision."
                )
                self.mixed_precision = True
            elif not checkpoint["__mixed_precision__"] and self.mixed_precision:
                self.logger.warning(
                    f"Mixed precision levels of training and loading checkpoint do not match. "
                    f"Requested mixed precision but checkpoint is saved without. "
                    f"This will almost surely lead to performance degradation."
                )

        self.logger.info(f"World size: {communication.get_world_size()}.")
        self.logger.info(f"Device count: {torch.cuda.device_count()}.")
        if communication.get_world_size() > 1:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[communication.get_local_rank()],
                broadcast_buffers=False,
            )

        # World size > 1 if distributed mode, else allow a DataParallel fallback, can be convenient for debugging.
        elif torch.cuda.device_count() > 1 and communication.get_world_size() == 1:
            self.model = DataParallel(self.model)

        self.__writers = (
            [
                JSONWriter(experiment_directory / "metrics.json"),
                CommonMetricPrinter(self.cfg.training.num_iterations),
                TensorboardWriter(experiment_directory / "tensorboard"),
            ]
            if communication.is_main_process()
            else []
        )

        with EventStorage(start_iter):
            self.training_loop(
                training_loader,
                start_iter,
                validation_loaders,
                experiment_directory=experiment_directory,
            )

        self.logger.info("Training completed.")

    @abstractmethod
    def evaluate(self, *args, **kwargs):  # noqa
        pass

    @abstractmethod
    def process_output(self, *args, **kwargs):  # noqa
        # Typically use this to scale data back to the original range.
        pass

    def log_process(self, idx, total):
        if idx % (total // 5) == 0 or total == (idx + 1):
            self.logger.info(f"Progress: {(idx + 1) / total * 100:.2f}%.")

    def log_first_training_example(self, data):
        storage = get_event_storage()
        self.logger.info(
            f"First case: slice_no: {data['slice_no'][0]}, filename: {data['filename'][0]}."
        )

        # TODO(jt): Cleaner, loop over types of images
        first_sampling_mask = data["sampling_mask"][0][0]
        first_acs_mask = data["acs_mask"][0][0] if "acs_mask" in data else None
        first_target = data["target"][0]
        first_masked_image = data["masked_image"][0]

        if self.ndim == 3:
            first_sampling_mask = first_sampling_mask[0]
            if first_acs_mask:
                first_acs_mask = first_acs_mask[0]

            num_slices = first_target.shape[first_target.names.index("slice")]
            first_target = first_target[num_slices // 2]
            first_masked_image = first_masked_image[num_slices // 2]
        elif self.ndim > 3:
            raise NotImplementedError

        storage.add_image(
            "train/mask", first_sampling_mask[..., 0].rename(None).unsqueeze(0)
        )
        if first_acs_mask:
            storage.add_image(
                "train/acs_mask",
                first_acs_mask[..., 0].rename(None).unsqueeze(0),
            )
        storage.add_image(
            "train/target",
            normalize_image(first_target.rename(None).unsqueeze(0)),
        )
        storage.add_image(
            "train/masked_image",
            normalize_image(
                transforms.modulus_if_complex(first_masked_image)
                .rename(None)
                .unsqueeze(0)
            ),
        )
        self.write_to_logs()

    def write_to_logs(self):
        if self.__writers is not None:
            for writer in self.__writers:
                writer.write()

    def __bind_sigint_signal(self):
        """Bind SIGINT signal to handle preemption or other kill of the process."""

        def raise_process_killed_error(signal_id, _):
            """Raise the ProcessKilledError."""
            self.logger.info(
                f"Received {signal.Signals(signal_id).name}. Shutting down..."
            )
            raise ProcessKilledException(signal_id, signal.Signals(signal_id).name)

        signal.signal(signalnum=signal.SIGINT, handler=raise_process_killed_error)

    # TODO(jt): Extend
    # def __repr__(self):
    #     pass
