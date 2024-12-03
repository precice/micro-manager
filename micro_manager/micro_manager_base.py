#!/usr/bin/env python3
"""
The Micro Manager abstract base class provides an interface for its subclasses.
The Micro Manager base class handles initialization shared by its subclasses (MicroManagerCoupling).
The base class should not be executed on its own. It is meant to be inherited by MicroManagerCoupling.

For more details see the MicroManagerCoupling class or the documentation at https://precice.org/tooling-micro-manager-overview.html.
"""

from mpi4py import MPI
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

        self._is_parallel = self._size > 1
        self._micro_sims_have_output = False

        self._local_number_of_sims = 0
        self._global_number_of_sims = 0
        self._is_rank_empty = False

        self._config = Config(config_file)

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
