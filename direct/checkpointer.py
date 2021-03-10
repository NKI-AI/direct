# coding=utf-8
# Copyright (c) DIRECT Contributors
import datetime
import logging
import pathlib
import re
import torch
import warnings
from pickle import UnpicklingError
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from typing import Union, Optional, Dict, Any

from direct.types import PathOrString


class Checkpointer:
    def __init__(
        self,
        model: torch.nn.Module,
        save_directory: pathlib.Path,
        save_to_disk: bool = True,
        model_regex: str = "^.*model$",
        **checkpointables: Any,
    ):

        self.logger = logging.getLogger(type(self).__name__)
        self.save_directory = save_directory
        self.model_regex = model_regex

        self.model = self._remove_module_attribute(model)
        for key in checkpointables:
            if re.match(model_regex, key):
                checkpointables[key] = self._remove_module_attribute(checkpointables[key])

        self.save_to_disk = save_to_disk
        self.checkpointables = checkpointables

    @staticmethod
    def _remove_module_attribute(model):
        if hasattr(model, "module"):
            if not isinstance(model, (DistributedDataParallel, DataParallel)):
                warnings.warn(
                    "Model has a `.module` property and is not derived from DistributeDataParallel"
                    f" or DataParallel. This is strange, but assuming the model is in `.module`."
                )

            model = model.module
        return model

    def load(
        self,
        iteration: Optional[Union[int, str, type]],
        checkpointable_objects: Optional[Any] = None,
    ) -> Dict:
        if iteration is not None and not isinstance(iteration, int) and iteration != "latest":
            raise ValueError("Value `iteration` is expected to be either None, an integer or `latest`.")

        if iteration is None:
            return {}

        if iteration in ("latest", -1):
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
        checkpoint = self.load_from_file(checkpoint_path, checkpointable_objects)
        checkpoint["iteration"] = iteration

        self.checkpoint_loaded = iteration
        # Return whatever is left
        return checkpoint

    def load_from_file(
        self,
        checkpoint_path: PathOrString,
        checkpointable_objects=None,
        only_models=False,
    ) -> Dict:
        checkpoint = self._load_checkpoint(checkpoint_path)
        checkpointable_objects = self.checkpointables if not checkpointable_objects else checkpointable_objects

        # TODO: Model and other checkpointable objects should be treated on the same footing
        self.logger.info("Loading model...")
        self._load_model(self.model, checkpoint["model"])

        for key in checkpointable_objects:
            if key not in checkpoint:
                self.logger.warning(f"Requested to load {key}, but this was not stored.")
                continue

            if only_models and not re.match(self.model_regex, key):
                continue

            if key.endswith("__") and key.startswith("__"):
                continue

            self.logger.info(f"Loading {key}...")
            obj = self.checkpointables[key]
            state_dict = checkpoint.pop(key)
            if re.match(self.model_regex, key):
                self.logger.debug(f"key {key} matches regex {self.model_regex}.")
                self._load_model(obj, state_dict)
            else:
                obj.load_state_dict(state_dict)

        return checkpoint

    def _load_model(self, obj, state_dict):
        # https://github.com/facebookresearch/fvcore/blob/master/fvcore/common/checkpoint.py
        # Link has more elaborate checking for incompatibles in _log_incompatible_keys
        incompatible = obj.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            raise NotImplementedError
        if incompatible.unexpected_keys:
            self.logger.warning(f"Unexpected keys provided which cannot be loaded: {incompatible.unexpected_keys}.")

    def load_models_from_file(self, checkpoint_path: PathOrString) -> None:
        _ = self.load_from_file(checkpoint_path, only_models=True)

    def save(self, iteration: int, **kwargs: Dict[str, str]) -> None:
        # For instance useful to only have the rank 0 process write to disk.
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
        if not pathlib.Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Requested to load {checkpoint_path}, but does not exist.")

        self.logger.info(f"Loaded checkpoint path: {checkpoint_path}.")

        try:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

        except UnpicklingError as e:
            self.logger.exception(f"Tried to load {checkpoint_path}, but was unable to unpickle: {e}.")
            raise

        return checkpoint
