#!/usr/bin/env python3
"""
The Micro Manager abstract base class provides an interface for its subclasses.
The Micro Manager base class handles initialization shared by its subclasses (MicroManagerCoupling).
The base class should not be executed on its own. It is meant to be inherited by MicroManagerCoupling.

For more details see the MicroManagerCoupling class or the documentation at https://precice.org/tooling-micro-manager-overview.html.
"""

from mpi4py import MPI
import logging
from abc import ABC, abstractmethod

from .config import Config


class MicroManagerInterface(ABC):
    """
    Abstract base class of Micro Manager classes. Defines interface for Micro Manager classes.
    """

    @abstractmethod
    def initialize(self):
        """
        Initialize micro simulations.
        """
        pass

    @abstractmethod
    def solve(self):
        """
        Solve micro simulations.
        """
        pass


class MicroManager(MicroManagerInterface):
    """
    Micro Manager base class provides common functionalities for its subclasses.
    """

    def __init__(self, config_file):
        """
        Constructor. Initializes member variables and logger shared between all subclasses.

        Parameters
        ----------
        config_file : string
            Name of the JSON configuration file (provided by the user).
        """
        self._comm = MPI.COMM_WORLD
        self._rank = self._comm.Get_rank()
        self._size = self._comm.Get_size()

        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(level=logging.INFO)

        # Create file handler which logs messages
        fh = logging.FileHandler("micro-manager.log")
        fh.setLevel(logging.INFO)

        # Create formatter and add it to handlers
        formatter = logging.Formatter(
            "[" + str(self._rank) + "] %(name)s -  %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter)
        self._logger.addHandler(fh)  # add the handlers to the logger

        self._is_parallel = self._size > 1
        self._micro_sims_have_output = False

        self._local_number_of_sims = 0
        self._global_number_of_sims = 0
        self._is_rank_empty = False

        self._logger.info("Provided configuration file: {}".format(config_file))
        self._config = Config(self._logger, config_file)

        # Data names of data to output to the snapshot database
        self._write_data_names = self._config.get_write_data_names()

        # Data names of data to read as input parameter to the simulations
        self._read_data_names = self._config.get_read_data_names()

        self._micro_dt = self._config.get_micro_dt()

        self._is_micro_solve_time_required = self._config.write_micro_solve_time()

    def initialize(self):
        """
        Initialize micro simulations. Not implemented
        """
        raise NotImplementedError(
            "Initialization of micro simulations is not implemented in base class"
        )

    def solve(self):
        """
        Solve micro simulations. Not implemented
        """
        raise NotImplementedError(
            "Solving micro simulations is not implemented in base class"
        )
