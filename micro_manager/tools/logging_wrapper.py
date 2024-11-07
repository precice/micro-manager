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

        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.addHandler(handler)

        return logger

    def log_info_one_rank(self, logger, message):
        """
        Log a message. Only the rank 0 logs the message.

        Parameters
        ----------
        logger : Logger
            Logger object.
        message : string
            Message to log.
        """
        if self._rank == 0:
            logger.info(message)

    def log_info_any_rank(self, logger, message):
        """
        Log a message. All ranks log the message.

        Parameters
        ----------
        logger : Logger
            Logger object.
        message : string
            Message to log.
        """
        logger.info(message)

    def log_error_any_rank(self, logger, message):
        """
        Log an error message. Only the rank 0 logs the message.

        Parameters
        ----------
        logger : Logger
            Logger object.
        message : string
            Message to log.
        """
        logger.error(message)
