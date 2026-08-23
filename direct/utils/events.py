# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
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

# Inspired on
# https://github.com/facebookresearch/detectron2/blob/45808c0ed68332cdb4c55801f1e2934d58231d35/detectron2/utils/events.py
# https://github.com/facebookresearch/fvcore/blob/master/fvcore/common/history_buffer.py
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Licensed under Apache 2.0
# Changes make here:
# - removed PathManager
# - changed formatting to coding style of the rest of this library
# - Added typing, changed put to add.

"""direct.utils.events module."""

import datetime
import json
import logging
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch

_CURRENT_STORAGE_STACK: list[Any] = []


def get_event_storage():
    """Get event storage.

    Returns:
            The :class:`EventStorage` object that's currently being used.
            Throws an error if no :class`EventStorage` is currently enabled.
    """
    if len(_CURRENT_STORAGE_STACK) == 0:
        raise ValueError("get_event_storage() has to be called inside a 'with EventStorage(...)' context!")
    return _CURRENT_STORAGE_STACK[-1]


class EventWriter:
    """Base class for writers that obtain events from :class:`EventStorage` and process them."""

    def write(self):
        """Write.

        Returns:
            ``None``.

        Raises:
            NotImplementedError: If the operation cannot be completed.
        """
        raise NotImplementedError

    def close(self):
        """Close.

        Returns:
            ``None``.
        """


class JSONWriter(EventWriter):
    """Write scalars to a json file.

    It saves scalars as one json per line (instead of a big json) for easy parsing.

    Examples parsing such a json file:

    .. code-block:: none

        $ cat metrics.json | jq -s '.[0:2]'
        [
          {
            ``"data_time"``: 0.008433341979980469,
            ``"iteration"``: 20,
            ``"loss"``: 1.9228371381759644,
            ``"loss_box_reg"``: 0.050025828182697296,
            ``"loss_classifier"``: 0.5316952466964722,
            ``"loss_mask"``: 0.7236229181289673,
            ``"loss_rpn_box"``: 0.0856662318110466,
            ``"loss_rpn_cls"``: 0.48198649287223816,
            ``"lr"``: 0.007173333333333333,
            ``"time"``: 0.25401854515075684
          },
          {
            ``"data_time"``: 0.007216215133666992,
            ``"iteration"``: 40,
            ``"loss"``: 1.282649278640747,
            ``"loss_box_reg"``: 0.06222952902317047,
            ``"loss_classifier"``: 0.30682939291000366,
            ``"loss_mask"``: 0.6970193982124329,
            ``"loss_rpn_box"``: 0.038663312792778015,
            ``"loss_rpn_cls"``: 0.1471673548221588,
            ``"lr"``: 0.007706666666666667,
            ``"time"``: 0.2490077018737793
          }
        ]

        $ cat metrics.json | jq '.loss_mask'
        0.7126231789588928
        0.689423680305481
        0.6776131987571716
        ...
    """

    def __init__(self, json_file: Path | str, window_size: int = 2):
        """Initialize the instance.

        Args:
                    json_file: Path to the JSON file. Data will be appended if it exists
                    window_size: Window size of median smoothing for variables for which `smoothing_hint` is ``True``.
                    validation: If true, will only log keys starting with val_

        Returns:
            ``None``.
        """

        # Handle is kept open for the writer's lifetime and closed in ``close``.
        self._file_handle = open(json_file, "a", encoding="utf-8")  # noqa: SIM115  # pylint: disable=consider-using-with
        self._window_size = window_size

    def write(self):
        """Write.

        Returns:
            ``None``.
        """
        storage = get_event_storage()
        to_save = {"iteration": storage.iter}
        to_save.update(storage.latest_with_smoothing_hint(self._window_size))
        self._file_handle.write(json.dumps(to_save, sort_keys=True) + "\n")
        self._file_handle.flush()
        try:
            os.fsync(self._file_handle.fileno())
        except AttributeError:
            pass

    def close(self):
        """Close.

        Returns:
            ``None``.
        """
        self._file_handle.close()


class TensorboardWriter(EventWriter):
    """Write all scalars to a tensorboard file."""

    def __init__(self, log_dir: Path | str, window_size: int = 20, **kwargs):
        """Initialize the instance.

        Args:
                    log_dir: The directory to save the output events.
                    window_size: The scalars will be median-smoothed by this window size.
                    kwargs: other arguments passed to `torch.utils.tensorboard.SummaryWriter(...)`

        Returns:
            ``None``.
        """
        self._window_size = window_size
        from torch.utils.tensorboard import SummaryWriter

        self._writer = SummaryWriter(str(log_dir), **kwargs)

    def write(self):
        """Write.

        Returns:
            ``None``.
        """
        storage = get_event_storage()
        for k, v in storage.latest_with_smoothing_hint(self._window_size).items():
            self._writer.add_scalar(k, v, storage.iter)

        if len(storage.vis_data) >= 1:
            for img_name, img, step_num in storage.vis_data:
                self._writer.add_image(img_name, img, step_num)
            storage.clear_images()

    def close(self):
        """Close.

        Returns:
            ``None``.
        """
        if hasattr(self, "_writer"):  # doesn't exist when the code fails at import
            self._writer.close()


class CommonMetricPrinter(EventWriter):
    """Print **common** metrics to the terminal, including iteration time, ETA, memory, all losses, and the learning
    rate.

    To print something different, please implement a similar printer by yourself.
    """

    def __init__(self, max_iter):
        """Initialize the instance.

        Args:
                    max_iter ``(int)``: the maximum number of iterations to train.
                        Used to compute ETA.

        Returns:
            ``None``.
        """
        self.logger = logging.getLogger(type(self).__name__)
        self._max_iter = max_iter
        self._last_write = None

    def write(self):
        """Write.

        Returns:
            ``None``.
        """
        storage = get_event_storage()
        iteration = storage.iter

        try:
            data_time = storage.history("data_time").avg(20)
        except KeyError:
            # they may not exist in the first few iterations (due to warmup)
            # or when SimpleTrainer is not used
            data_time = None

        eta_string = "N/A"
        try:
            iter_time = storage.history("time").global_avg()
            eta_seconds = storage.history("time").median(1000) * (self._max_iter - iteration)
            storage.add_scalar("eta_seconds", eta_seconds, smoothing_hint=False)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        except KeyError:
            iter_time = None
            # estimate eta on our own - more noisy
            if self._last_write is not None:
                estimate_iter_time = (time.perf_counter() - self._last_write[1]) / (iteration - self._last_write[0])
                eta_seconds = estimate_iter_time * (self._max_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            self._last_write = (iteration, time.perf_counter())

        try:
            lr = f"{storage.history('lr').latest():.6f}"
        except KeyError:
            lr = "N/A"

        if torch.cuda.is_available():
            max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
        else:
            max_mem_mb = None

        metrics_and_losses_string = "  ".join(
            [
                f"{k}: {v.median(20):.6f}"
                for k, v in storage.histories().items()
                if ("loss" in k or "metric" in k or "reg" in k)
            ]
        )

        time_string = f"time: {iter_time:.4f}  " if iter_time is not None else ""
        data_time_string = f"data_time: {data_time:.4f}  " if data_time is not None else ""
        memory_string = f"max_mem: {max_mem_mb:.0f}M" if max_mem_mb is not None else ""

        # no logger here, the code already saves the iterations to json.
        self.logger.info(
            f"eta: {eta_string}  iter: {iteration}  "
            f"{metrics_and_losses_string}  {time_string}{data_time_string}lr: {lr}  {memory_string}"
        )


class EventStorage:
    """The user-facing class that provides metric storage functionalities.

    In the future we may add support for storing / logging other types of data if needed.
    """

    def __init__(self, start_iter=0):
        """Initialize the instance.

        Args:
                    start_iter: The index to start with.

        Returns:
            ``None``.
        """
        self._history = defaultdict(HistoryBuffer)
        self._smoothing_hints = {}
        self._latest_scalars = {}
        self._iter = start_iter
        self._current_prefix = ""
        self._vis_data = []

    def add_image(self, img_name, img_tensor):
        """Add an `img_tensor` to the `_vis_data` associated with `img_name`.

        Args:
            img_name: The name of the input_image to put into tensorboard.
            img_tensor: An `uint8` or `float` Tensor of shape `[channel, height, width]` where `channel` is 3. The input_image
                format should be RGB. The elements in img_tensor can either have values in ``[0, 1]`` (float32) or ``[0, 255]`` ``(``uint8``)``. The
                `img_tensor` will be visualized in tensorboard.

        Returns:
            ``None``.
        """
        self._vis_data.append((img_name, img_tensor, self._iter))

    def clear_images(self):
        """Delete all the stored images for visualization.

        This should be called after images are written to tensorboard.

        Returns:
            ``None``.
        """
        self._vis_data = []

    def add_scalar(self, name, value, smoothing_hint=True):
        """Add a scalar `value` to the `HistoryBuffer` associated with `name`.

        Args:
            name: Name.
            value: Value.
            smoothing_hint: A ``'hint'`` on whether this scalar is noisy and should be smoothed when logged. The hint will be
                accessible through `EventStorage.smoothing_hints`. A writer may ignore the hint and apply custom smoothing rule. It
                Default is ``True`` because most scalars we save need to be smoothed to provide any useful signal.

        Returns:
            The result.
        """
        name = self._current_prefix + name
        history = self._history[name]
        value = float(value)
        history.update(value, self._iter)
        self._latest_scalars[name] = value

        existing_hint = self._smoothing_hints.get(name)
        if existing_hint is not None:
            if existing_hint != smoothing_hint:
                raise AssertionError(f"Scalar {name} was put with a different smoothing_hint!")
        else:
            self._smoothing_hints[name] = smoothing_hint

    def add_scalars(self, *, smoothing_hint=True, **kwargs):
        """Put multiple scalars from keyword arguments.

        Args:
            smoothing_hint: Smoothing hint.
            **kwargs: Kwargs.

        Returns:
            ``None``.

        Examples:
            storage.add_scalars``(loss=my_loss, accuracy=my_accuracy, smoothing_hint=True)``
        """
        for k, v in kwargs.items():
            self.add_scalar(k, v, smoothing_hint=smoothing_hint)

    def add_graph(self, img_name, img_tensor):
        """Add an `img_tensor` to the `_vis_data` associated with `img_name`.

        Args:
            img_name: The name of the input_image to put into tensorboard.
            img_tensor: An `uint8` or `float` Tensor of shape `[channel, height, width]` where `channel` is 3. The input_image
                format should be RGB. The elements in img_tensor can either have values in ``[0, 1]`` (float32) or ``[0, 255]`` ``(``uint8``)``. The
                `img_tensor` will be visualized in tensorboard.

        Returns:
            ``None``.
        """
        self._vis_data.append((img_name, img_tensor, self._iter))

    def history(self, name):
        """History.

        Args:
            name: Name.

        Returns:
                    HistoryBuffer: the scalar history for name
        """
        ret = self._history.get(name, None)
        if ret is None:
            raise KeyError(f"No history metric available for {name}!")
        return ret

    def histories(self):
        """Histories.

        Returns:
                    dict[name -> HistoryBuffer]: the HistoryBuffer for all scalars
        """
        return self._history

    def latest(self):
        """Latest.

        Returns:
                    dict[name -> number]: the scalars that's added in the current iteration.
        """
        return self._latest_scalars

    def latest_with_smoothing_hint(self, window_size=20):
        """Similar to :meth:`latest`, but the returned values are either the un-smoothed original latest value, or a
        median of the given window_size, depend on whether the smoothing_hint is ``True``.

        This provides a default behavior that other writers can use.

        Args:
            window_size: Window size.

        Returns:
            ``None``.
        """
        result = {}
        for k, v in self._latest_scalars.items():
            result[k] = self._history[k].median(window_size) if self._smoothing_hints[k] else v
        return result

    def smoothing_hints(self):
        """Smoothing hints.

        Returns:
                    dict[name -> bool]: the user-provided hint on whether the scalar
                        is noisy and needs smoothing.
        """
        return self._smoothing_hints

    def step(self):
        """User should call this function at the beginning of each iteration, to notify the storage of the start of a
        new iteration.

        The storage will then be able to associate the new data with the correct iteration number.

        Returns:
            ``None``.
        """
        self._iter += 1
        # TODO: This clears validation metrics.
        # self._latest_scalars = {}

    @property
    def vis_data(self):
        """Vis data.

        Returns:
            ``None``.
        """
        return self._vis_data

    @property
    def iter(self):
        """Iter.

        Returns:
            ``None``.
        """
        return self._iter

    def __enter__(self):
        """Enter the runtime context.

        Returns:
            ``None``.
        """
        _CURRENT_STORAGE_STACK.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the runtime context.

        Args:
            exc_type: Exc type.
            exc_val: Exc val.
            exc_tb: Exc tb.

        Returns:
            ``None``.

        Raises:
            AssertionError: If the operation cannot be completed.
        """
        if _CURRENT_STORAGE_STACK[-1] != self:
            raise AssertionError
        _CURRENT_STORAGE_STACK.pop()

    @contextmanager
    def name_scope(self, name):
        """Name scope.

        Args:
            name: Name.

        Returns:
            ``None``.

        Yields:
                    A context within which all the events added to this storage
                    will be prefixed by the name scope.
        """
        old_prefix = self._current_prefix
        self._current_prefix = name.rstrip("/") + "/"
        yield
        self._current_prefix = old_prefix


class HistoryBuffer:
    """Track a series of scalar values and provide access to smoothed values over a window or the global average of the
    series.
    """

    def __init__(self, max_length: int = 1000000) -> None:
        """Initialize the instance.

        Args:
                    max_length: maximal number of values that can be stored in the
                        buffer. When the capacity of the buffer is exhausted, old
                        values will be removed.

        Returns:
            ``None``.
        """
        self._max_length: int = max_length
        self._data: list[tuple[float, float]] = []  # (value, iteration) pairs
        self._count: int = 0
        self._global_avg: float = 0

    def update(self, value: float, iteration: float | None = None) -> None:
        """Add a new scalar value produced at certain iteration.

        If the length of the buffer exceeds self._max_length, the oldest element will be removed from the buffer.

        Args:
            value: Value.
            iteration: Iteration.

        Returns:
            ``None``.
        """
        if iteration is None:
            iteration = self._count
        if len(self._data) == self._max_length:
            self._data.pop(0)
        self._data.append((value, iteration))

        self._count += 1
        self._global_avg += (value - self._global_avg) / self._count

    def latest(self) -> float:
        """Return the latest scalar value added to the buffer.

        Returns:
            The result.
        """
        return self._data[-1][0]

    def median(self, window_size: int) -> float:
        """Return the median of the latest `window_size` values in the buffer.

        Args:
            window_size: Window size.

        Returns:
            The result.
        """
        return np.median([x[0] for x in self._data[-window_size:]])

    def avg(self, window_size: int) -> float:
        """Return the mean of the latest `window_size` values in the buffer.

        Args:
            window_size: Window size.

        Returns:
            The result.
        """
        return float(np.mean([x[0] for x in self._data[-window_size:]]))

    def global_avg(self) -> float:
        """Return the mean of all the elements in the buffer.

        Note that this includes those getting removed due to limited buffer storage.

        Returns:
            The result.
        """
        return self._global_avg

    def values(self) -> list[tuple[float, float]]:
        """Values.

        Returns:
                    list[(number, iteration)]: content of the current buffer.
        """
        return self._data
