# coding=utf-8
# Copyright (c) DIRECT Contributors
import logging


class DirectException(BaseException):
    def __init__(self, *args, **kwargs):
        self.logger = logging.getLogger(__name__)


class ProcessKilledException(DirectException):
    """The process received SIGINT signal."""

    def __init__(self, signal_id: int, signal_name: str):
        """
        Parameters
        ----------
        signal_id : str
        signal_name: str
        """
        super().__init__()
        self.logger.exception(
            f"Received signal (signal_id = {signal_id} - signal_name = {signal_name}). "
            "Critical. Process will stop."
        )
        self.signal_id = signal_id
        self.signal_name = signal_name
