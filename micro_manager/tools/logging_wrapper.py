#!/usr/bin/env python3
"""
Provides a logging wrapper for the Micro Manager classes.
"""
import logging


class Logger:
    """
    Provides a logging wrapper for the Micro Manager classes.
    """

    def __init__(self, name, log_file, rank=0, level=logging.INFO):
        """
        Set up a logger.

        Parameters
        ----------
        name : string
            Name of the logger.
        log_file : string
            Name of the log file.
        level : int, optional
            Logging level (default is logging.INFO).
        """

        self._rank = rank

        handler = logging.FileHandler(log_file)
        handler.setLevel(level)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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

    def log_info_one_rank(self, message):
        """
        Log a message. Only the rank 0 logs the message.

        Parameters
        ----------
        message : string
            Message to log.
        """
        if self._rank == 0:
            self._logger.info(message)

    def log_info_any_rank(self, message):
        """
        Log a message. All ranks log the message.

        Parameters
        ----------
        message : string
            Message to log.
        """
        self._logger.info(message)

    def log_error_any_rank(self, message):
        """
        Log an error message. Only the rank 0 logs the message.

        Parameters
        ----------
        message : string
            Message to log.
        """
        self._logger.error("[" + str(self._rank) + "] " + message)
