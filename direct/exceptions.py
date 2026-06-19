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
import logging


class DirectException(BaseException):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.logger = logging.getLogger(__name__)


class ProcessKilledException(DirectException):
    """The process received SIGINT signal."""

    def __init__(self, signal_id: int, signal_name: str):
        """
        Parameters
        ----------
        signal_id: str
        signal_name: str
        """
        super().__init__()
        self.logger.exception(
            f"Received signal (signal_id = {signal_id} - signal_name = {signal_name}). Critical. Process will stop."
        )
        self.signal_id = signal_id
        self.signal_name = signal_name


class TrainingException(DirectException):
    def __init__(self, message=None):
        super().__init__()
        if message:
            self.logger.exception("TrainingException")
        else:
            self.logger.exception(f"TrainingException: {message}")


class ItemNotFoundException(DirectException):
    def __init__(self, item_name, message=None):
        super().__init__()
        error_name = "".join([s.capitalize() for s in item_name.split(" ")]) + "Exception"
        if message:
            self.logger.exception(error_name)
        else:
            self.logger.exception("%s: %s", error_name, message)
