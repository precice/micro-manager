#!/usr/bin/env python3
"""
Provides a logging wrapper for the Micro Manager classes.
"""
import logging
import sys


class Logger:
    """
    Provides a logging wrapper for the Micro Manager classes.
    """

    def __init__(
        self, name, log_file=None, rank=0, level=logging.INFO, csv_logger=False
    ):
        """
        Set up a logger.

        Parameters
        ----------
        name : string
            Name of the logger.
        log_file : string
            Name of the log file (default is None).
        rank : int, optional
            Rank of the logger (default is 0).
        level : int, optional
            Logging level (default is logging.INFO).
        csv_logger : bool, optional
            If True, the logger will log in CSV format (default is False).
        """

        self._rank = rank

        if log_file is None:
            handler = logging.StreamHandler(sys.stdout)
        else:
            handler = logging.FileHandler(log_file)

        handler.setLevel(level)

        if csv_logger:
            formatter = logging.Formatter("%(message)s")
        else:
            formatter = logging.Formatter(
                "("
                + str(self._rank)
                + ") %(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%m/%d/%Y %I:%M:%S %p",
            )

        handler.setFormatter(formatter)

        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        self._logger.addHandler(handler)

    def get_logger(self):
        """
        Get the logger.

        Returns
        -------
        logger : object of logging
            Logger defined from the standard package logging
        """
        return self._logger

    def log_info_rank_zero(self, message):
        """
        Rank 0 logs a message.

        Parameters
        ----------
        message : string
            Message to log.
        """
        if self._rank == 0:
            self._logger.info(message)

    def log_info(self, message):
        """
        Log a message.

        Parameters
        ----------
        message : string
            Message to log.
        """
        self._logger.info(message)

    def log_error(self, message):
        """
        Log an error message.

        Parameters
        ----------
        message : string
            Message to log.
        """
        self._logger.error(message)

    def log_warning_rank_zero(self, message):
        """
        Rank 0 logs a warning.

        Parameters
        ----------
        message : string
            Message to log.
        """
        if self._rank == 0:
            self._logger.warning(message)

    def log_warning(self, message):
        """
        Log a warning.

        Parameters
        ----------
        message : string
            Message to log.
        """
        self._logger.warning(message)
