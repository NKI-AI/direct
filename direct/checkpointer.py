# coding=utf-8
# Copyright (c) DIRECT Contributors
import logging
import pathlib
import torch
import datetime
import warnings

from collections import OrderedDict
from pickle import UnpicklingError
from typing import Union, Optional, Dict, Any

from direct.types import PathOrString

from torch.nn.parallel import DistributedDataParallel
from torch.nn import DataParallel


class Checkpointer:
    def __init__(
        self,
        model: torch.nn.Module,
        save_directory: pathlib.Path,
        save_to_disk: bool = True,
        **checkpointables: Any,
    ):

        self.save_directory = save_directory
        self.logger = logging.getLogger(type(self).__name__)

        if hasattr(model, "module"):
            if not isinstance(model, (DistributedDataParallel, DataParallel)):
                self.logger.warning(
                    f"Model has a `.module` property and is not derived from DistributeDataParallel"
                    f" or DataParallel. This is strange, but assuming the model is in `.module`."
                )
            else:
                pass
            model = model.module

        self.model = model
        self.save_to_disk = save_to_disk
        self.checkpoint_loaded = None
        self.checkpointables = checkpointables

    def load(
        self,
        iteration: Optional[Union[int, str, type]],
        checkpointable_objects: Optional[Any] = None,
    ) -> Dict:
        if (
            iteration is not None
            and not isinstance(iteration, int)
            and iteration != "latest"
        ):
            raise ValueError(
                "Value `iteration` is expected to be either None, an integer or `latest`."
            )

        if iteration is None:
            return {}

        if iteration == "latest" or iteration == -1:
            last_model_text_path = self.save_directory / "last_model.txt"
            self.logger.info("Attempting to load latest model.")
            if last_model_text_path.exists():
                with open(pathlib.Path(last_model_text_path), "r") as f:
                    iteration = int(f.readline())
                    self.logger.info(f"Loading last saved iteration: {iteration}.")

            else:
                self.logger.info(
                    f"Latest model not found. Perhaps `last_model.txt` (path = {last_model_text_path}) "
                    f"is missing? You can try to set an explicit iteration number, or create this file if "
                    f"you believe this is an error. Will not load any model."
                )
                return {}

        checkpoint_path = self.save_directory / f"model_{iteration}.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Requested to load {checkpoint_path}, but does not exist."
            )

        self.logger.info(f"Loaded checkpoint path: {checkpoint_path}.")
        checkpoint = self._load_checkpoint(checkpoint_path)
        checkpoint["iteration"] = iteration

        self.logger.info(f"Loading model...")
        self._load_model(checkpoint)

        for key in (
            self.checkpointables
            if not checkpointable_objects
            else checkpointable_objects
        ):
            if key in checkpoint:
                if key.endswith("__") and key.startswith("__"):
                    continue
                self.logger.info(f"Loading {key}...")
                obj = self.checkpointables[key]
                obj.load_state_dict(checkpoint.pop(key))
            else:
                self.logger.warning(
                    f"Requested to load {key}, but this was not stored."
                )

        self.checkpoint_loaded = iteration
        # Return whatever is left
        return checkpoint

    def load_from_file(self, checkpoint_path: PathOrString) -> None:
        self.logger.info(f"Loaded checkpoint path: {checkpoint_path}.")
        checkpoint = self._load_checkpoint(checkpoint_path)

        self.logger.info(f"Loading model...")
        self._load_model(checkpoint)

    def save(self, iteration: int, **kwargs: Dict[str, str]) -> None:
        # For instance useful to only  have the rank 0 process write to disk.
        if not self.save_to_disk:
            return

        data: Dict[str, Any] = {"model": self.model.state_dict()}
        for key, obj in self.checkpointables.items():
            if key.endswith("__") and key.startswith("__"):
                # Keys of the form __TEXT__ do should not have state
                data[key] = obj
            else:
                if hasattr(obj, "state_dict"):
                    data[key] = obj.state_dict()
                else:
                    self.logger.warning(f"Value of key {key} has no state_dict.")

        data.update(kwargs)

        checkpoint_path = self.save_directory / f"model_{iteration}.pt"
        self.logger.info(f"Saving checkpoint to: {checkpoint_path}.")

        data["__datetime__"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(str(checkpoint_path), "wb") as f:
            torch.save(data, f)

        # noinspection PyTypeChecker
        with open(self.save_directory / "last_model.txt", "w") as f:  # type: ignore
            f.write(str(iteration))  # type: ignore

    def _load_checkpoint(self, checkpoint_path: PathOrString) -> Dict:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

        except UnpicklingError as e:
            self.logger.exception(
                f"Tried to load {checkpoint_path}, but was unable to unpickle: {e}."
            )
            raise

        # Return whatever is left
        return checkpoint

    def _load_model(self, checkpoint: Any) -> None:
        # TODO check: https://github.com/facebookresearch/fvcore/blob/master/fvcore/common/checkpoint.py
        model_state_dict = checkpoint.pop("model")

        # Strip 'module' if present
        if list(model_state_dict.keys())[0].startswith("module."):
            new_ordered_dict = OrderedDict()
            warnings.warn(
                "Weights start with `.module`, suggesting model was saved with DataParallel. "
                "Removing these now, consider adapting this in your code."
            )

            for idx, (k, v) in enumerate(model_state_dict.items()):
                name = k[7:]
                new_ordered_dict[name] = v
            model_state_dict = new_ordered_dict

        incompatible = self.model.load_state_dict(model_state_dict, strict=False)
        if incompatible.missing_keys:
            raise NotImplementedError
        if incompatible.unexpected_keys:
            raise NotImplementedError
