# coding=utf-8
# Copyright (c) DIRECT Contributors
import logging
import pathlib
import sys
import torch
import signal
import direct
import numpy as np

from typing import Optional, Dict, Tuple, List
from apex import amp
from abc import abstractmethod, ABC

from torch import nn
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Sampler, BatchSampler

from direct.data.mri_transforms import AddNames
from direct.data import sampler
from direct.checkpointer import Checkpointer
from direct.utils.collate import named_collate
from direct.utils import communication, prefix_dict_keys, evaluate_dict, normalize_image
from direct.utils.events import get_event_storage, EventStorage, JSONWriter, CommonMetricPrinter, TensorboardWriter
from direct.data import transforms
from direct.config.defaults import BaseConfig
from direct.exceptions import ProcessKilledException
from direct.types import PathOrString


class Engine(ABC):
    def __init__(self, cfg: BaseConfig,
                 model: nn.Module,
                 device: int, mixed_precision: bool = False):
        self.logger = logging.getLogger(type(self).__name__)

        self.cfg = cfg
        self.model = model
        self.device = device

        self.mixed_precision = mixed_precision
        self.checkpointer = None

        # TODO: This might not be needed, if these objects are changed in-place
        self.__optimizer = None
        self.__lr_scheduler = None
        self.__writers = None
        self.__bind_sigint_signal()

    @abstractmethod
    def build_loss(self) -> Dict:
        pass

    @abstractmethod
    def build_metrics(self) -> Dict:
        pass

    @abstractmethod
    def _do_iteration(self, *args, **kwargs) -> Tuple[torch.Tensor, Dict]:
        pass

    @torch.no_grad()
    def predict(self, dataset: Dataset, experiment_directory: pathlib.Path,
                checkpoint_number: int = -1, num_workers: int = 6) -> np.ndarray:
        # TODO: Improve the way of checkpointing
        self.logger.info(f'Predicting...')
        self.checkpointer = Checkpointer(
            self.model, experiment_directory, save_to_disk=False)

        # Do not load again if we already have loaded the checkpoint.
        if self.checkpointer.checkpoint_loaded is not checkpoint_number:
            self.checkpointer.load(iteration=checkpoint_number, checkpointable_objects=None)

        sampler = self.build_sampler(dataset, 'sequential', limit_number_of_volumes=None)
        batch_sampler = BatchSampler(sampler, batch_size=self.cfg.validation.batch_size, drop_last=False)
        # TODO: Batch size can be much larger, perhaps have a different batch size during evaluation.
        data_loader = self.build_loader(dataset, batch_sampler=batch_sampler, num_workers=num_workers)
        loss, output = self.evaluate(
            data_loader, loss_fns=None, volume_metrics=None, crop=None, is_validation_process=False)

        return output

    @staticmethod
    def build_loader(dataset: Dataset,
                     sampler: Optional[Sampler] = None,
                     batch_size: int = 1,
                     batch_sampler: Optional[Sampler] = None,
                     num_workers: int = 6,
                     drop_last: bool = False) -> DataLoader:
        # TODO(jt): Custom memory pinning.
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=named_collate,
            drop_last=drop_last,
            pin_memory=True,
        )
        return loader

    @staticmethod
    def build_sampler(dataset: Dataset, sampler_type: str, **kwargs) -> Sampler:
        if sampler_type == 'random':
            sampler: Sampler = direct.data.sampler.DistributedSampler(
                len(dataset), shuffle=True, seed=None, **kwargs)  # TODO: Set seed
        elif sampler_type == 'sequential':
            sampler: Sampler = direct.data.sampler.DistributedSequentialSampler(dataset, **kwargs)
        else:
            raise ValueError(f'Sampler type {sampler_type} not supported.')

        return sampler

    def training_loop(self,
                      data_loader: DataLoader,
                      start_iter: int,
                      validation_data_loaders: Optional[List[DataLoader]] = None):
        self.logger.info(f'Local rank: {communication.get_local_rank()}.')
        self.model.train()

        loss_fns = self.build_loss()
        metric_fns = self.build_metrics()
        storage = get_event_storage()

        total_iter = self.cfg.training.num_iterations # noqa
        for data, iter_idx in zip(data_loader, range(start_iter, total_iter)):
            data = AddNames()(data)
            if iter_idx == 0:
                # TODO: This is too MRI specific.
                self.logger.info(f"First case: slice_no: {data['slice_no'][0]}, filename: {data['filename'][0]}.")
                storage.add_image('train/mask', data['sampling_mask'][0, ..., 0])
                if 'acs_mask' in data:
                    storage.add_image('train/acs_mask', data['acs_mask'][0, ..., 0])
                if 'sensitivity_map' in data:
                    storage.add_image(
                        'train/sensitivity_map', normalize_image(
                            transforms.modulus_if_complex(data['sensitivity_map'][0][0]).rename(None).unsqueeze(0)))

                storage.add_image('train/target', normalize_image(data['target'][0].rename(None).unsqueeze(0)))
                storage.add_image(
                    'train/masked_image', normalize_image(
                        transforms.modulus_if_complex(data['masked_image'][0]).rename(None).unsqueeze(0)))
                self.write_to_logs()

            try:
                output, loss_dict = self._do_iteration(data, loss_fns)
            except ProcessKilledException as e:
                # If the process is killed, the output if saved at state iter_idx, which is the current state,
                # so the computation can restart from the last iteration.
                self.logger.exception(f'{e}.')
                self.checkpointer.save(iter_idx)  # Save checkpoint at kill. # noqa
                self.write_to_logs()  # TODO: This causes the issue that current metrics are not written,
                # and you end up with an empty line.
                sys.exit(f'Exited with exception: {e}')

            # Gradient accumulation
            if (iter_idx + 1) % self.cfg.training.gradient_steps == 0:  # type: ignore
                # TODO: Is this slow? This is a generator, so should be cheap.
                pass
                # parameter_list = self.model.parameters() if not self.mixed_precision else \
                #     amp.master_params(self.__optimizer)
                # if self.cfg.training.gradient_steps > 1:  # type: ignore
                #     for parameter in parameter_list:
                #         if parameter.grad is not None:
                #             # In-place division
                #             parameter.grad.div_(self.cfg.training.gradient_steps)  # type: ignore
                # if self.cfg.training.gradient_clipping > 0.0:  # type: ignore
                #     torch.nn.utils.clip_grad_norm_(parameter_list, self.cfg.training.gradient_clipping)  # type: ignore
                #
                # # Gradient norm
                # if self.cfg.training.gradient_debug:  # type: ignore
                #     parameters = list(filter(lambda p: p.grad is not None, parameter_list))
                #     gradient_norm = sum([parameter.grad.data ** 2 for parameter in parameters]).sqrt()  # typing: ignore
                #     storage.add_scalar('gradient_norm', gradient_norm)

            self.__optimizer.step()  # type: ignore
            # Incorrect inference by mypy and pyflake
            self.__lr_scheduler.step()  # type: ignore # noqa
            storage.add_scalar('lr', self.__optimizer.param_groups[0]['lr'], smoothing_hint=False)

            self.__optimizer.zero_grad()  # type: ignore

            # Reduce the loss over all devices
            loss_dict_reduced = communication.reduce_tensor_dict(loss_dict)
            loss_reduced = sum(loss_dict_reduced.values())

            metrics_dict = evaluate_dict(
                metric_fns,
                transforms.modulus_if_complex(output.detach()).rename(None),
                data['target'].rename(None).detach().to(self.device),
                reduction='mean'
            )
            metrics_dict_reduced = communication.reduce_tensor_dict(metrics_dict) if metrics_dict else {}
            storage.add_scalars(loss=loss_reduced, **loss_dict_reduced, **metrics_dict_reduced)

            if validation_data_loaders is not None \
                    and iter_idx > 5\
                    and (iter_idx % self.cfg.training.validation_steps == 0 or (iter_idx + 1) == total_iter):
                for curr_dataset_name, curr_validation_data_loader in validation_data_loaders:
                    self.logger.info(f'Evaluating {curr_dataset_name}...')  # TODO(jt): Fix with better names and stuff.
                    curr_val_loss_dict = self.evaluate(
                        curr_validation_data_loader, loss_fns,
                        evaluation_round=iter_idx // self.cfg.training.validation_steps - 1,
                        crop=self.cfg.training.loss.crop)
                    key_prefix = 'val_' if not curr_dataset_name else f'val_{curr_dataset_name}_'
                    storage.add_scalars(**prefix_dict_keys(curr_val_loss_dict, key_prefix), smoothing_hint=False)

                self.logger.info(f'Done evaluation at iteration {iter_idx}.')
                self.model.train()

            if iter_idx > 5 and\
                    (iter_idx % self.cfg.training.checkpointer.checkpoint_steps == 0 or (iter_idx + 1) == total_iter):
                self.logger.info(f'Checkpointing at iteration {iter_idx}.')
                self.checkpointer.save(iter_idx)

            # Log every 20 iterations, or at a validation step or at the end of training.
            if iter_idx > 5 and (iter_idx % 20 == 0 or
                                 iter_idx % self.cfg.training.validation_steps == 0 or
                                 (iter_idx + 1) == total_iter):
                self.write_to_logs()

            storage.step()

    def train(self,
              optimizer: torch.optim.Optimizer,
              lr_scheduler: torch.optim.lr_scheduler._LRScheduler,  # noqa
              training_data: Dataset,
              experiment_directory: pathlib.Path,
              validation_data: Dataset = None,
              resume: bool = False,
              initialization: Optional[PathOrString] = None,
              num_workers: int = 6) -> None:

        # TODO: Does not need to be member of self.
        self.__optimizer = optimizer
        # TODO: Optimizer and LR scheduler need to be resumed too.
        self.__lr_scheduler = lr_scheduler

        training_sampler = self.build_sampler(training_data, 'random')
        # TODO: Configurable
        training_loader = self.build_loader(
            training_data, sampler=training_sampler,
            batch_size=self.cfg.training.batch_size,
            num_workers=num_workers, drop_last=True)

        if validation_data:
            validation_loaders = []
            for idx, curr_validation_data in enumerate(validation_data):
                dataset_description = f'ds{idx}' if len(validation_data) > 1 else None
                self.logger.info(f'Building dataloader for {dataset_description}.')
                curr_validation_sampler = self.build_sampler(
                    curr_validation_data, 'sequential', limit_number_of_volumes=None)
                curr_batch_sampler = BatchSampler(
                    curr_validation_sampler, batch_size=self.cfg.validation.batch_size, drop_last=False)
                validation_loaders.append(
                    (
                        dataset_description,
                        self.build_loader(
                            curr_validation_data, batch_sampler=curr_batch_sampler, num_workers=num_workers)
                    )
                )
        else:
            validation_loaders = None

        self.model = self.model.to(self.device)

        # Optimizer
        self.__optimizer.zero_grad()  # type: ignore

        # Mixed precision setup. This requires the model to be on the gpu.
        git_hash = direct.utils.git_hash()
        extra_checkpointing = {'__author__': git_hash if git_hash else 'N/A'}
        if self.mixed_precision > 0:
            opt_level = f'O{self.mixed_precision}'
            self.logger.info(f'Using apex level {opt_level}.')
            self.model, self.__optimizer = amp.initialize(self.model, self.__optimizer, opt_level=opt_level)
            extra_checkpointing['amp'] = amp
            extra_checkpointing['opt_level'] = opt_level

        self.checkpointer = Checkpointer(
            self.model, experiment_directory,
            save_to_disk=communication.is_main_process(),
            optimizer=optimizer, lr_scheduler=lr_scheduler, **extra_checkpointing)

        # Load checkpoint
        start_iter = 0
        checkpoint = {}
        if resume:
            self.logger.info('Attempting to resume...')
            # This changes the model inplace
            checkpoint = self.checkpointer.load(
                iteration='latest', checkpointable_objects=['amp'] if self.mixed_precision > 0 else [])
            if not checkpoint:
                self.logger.info('No checkpoint found. Starting from scratch.')
            else:
                start_iter = checkpoint['iteration'] + 1
                self.logger.info(f'Starting from iteration: {start_iter}.')

        if start_iter > 0 and initialization:
            self.logger.warning(f'Initialization checkpoint set to {initialization},'
                                f' but model will resume training from previous checkpoint. Initialization ignored.')
        elif initialization:
            self.logger.info(f'Initializing from {initialization}...')
            self.checkpointer.load_from_file(initialization)

        if '__author__' in checkpoint:
            self.logger.info(f"Git hash of checkpoint: {checkpoint['__author__']}")
            if checkpoint['__author__'] != direct.utils.git_hash():
                self.logger.warning(f"Current git hash {direct.utils.git_hash()} is different from the one "
                                    f"this checkpoint is saved with ({checkpoint['__author__']}. This can be fine, "
                                    f"but beware that this can be a source of confusion.")

        if '__datetime__' in checkpoint:
            self.logger.info(f"Checkpoint created at: {checkpoint['__datetime__']}")
        if 'opt_level' in checkpoint:
            if checkpoint['opt_level'] != opt_level:
                self.logger.warning(f"Mixed precision opt-levels do not match. "
                                    f"Requested {opt_level} got {checkpoint['opt_level']} from checkpoint. "
                                    f"This will almost surely lead to performance degradation.")

        self.logger.info(f'World size: {communication.get_world_size()}.')
        self.logger.info(f'Device count: {torch.cuda.device_count()}.')
        if communication.get_world_size() > 1:
            self.model = DistributedDataParallel(
                self.model, device_ids=[communication.get_rank()], broadcast_buffers=False)

        # World size > 1 if distributed mode, else allow a DataParallel fallback, can be convenient for debugging.
        elif torch.cuda.device_count() > 1 and communication.get_world_size() == 1:
            self.model = DataParallel(self.model)

        self.__writers = ([
             JSONWriter(experiment_directory / 'metrics.json'),
             CommonMetricPrinter(self.cfg.training.num_iterations),
             TensorboardWriter(experiment_directory / 'tensorboard')
         ] if communication.is_main_process() else [])

        with EventStorage(start_iter):
            self.training_loop(training_loader, start_iter, validation_loaders)

        self.logger.info('Training completed.')

    @abstractmethod
    def evaluate(self, *args, **kwargs): # noqa
        pass

    @abstractmethod
    def process_output(self, *args, **kwargs): # noqa
        # Typically use this to scale data back to the original range.
        pass

    def log_process(self, idx, total):
        if total % (idx + 1) == 5 or total == (idx + 1):
            self.logger.info(f'Progress: {(idx + 1) / total * 100:.2f}%.')

    def write_to_logs(self):
        if self.__writers is not None:
            for writer in self.__writers:
                writer.write()

    def __bind_sigint_signal(self):
        """Bind SIGINT signal to handle preemption or other kill of the process."""
        def raise_process_killed_error(signal_id, _):
            """Raise the ProcessKilledError."""
            self.logger.info(f'Received {signal.Signals(signal_id).name}. Shutting down...')
            raise ProcessKilledException(signal_id, signal.Signals(signal_id).name)
        signal.signal(signalnum=signal.SIGINT, handler=raise_process_killed_error)

    # TODO(jt): Extend
    # def __repr__(self):
    #     pass
