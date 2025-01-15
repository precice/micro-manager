"""
Class GlobalAdaptivityLBCalculator provides methods to adaptively control of micro simulations
in a global way. If the Micro Manager is run in parallel, an all-to-all comparison of simulations
on each rank is done, along with dynamic load balancing.

Note: All ID variables used in the methods of this class are global IDs, unless they have *local* in their name.
"""
import hashlib
from copy import deepcopy
from typing import Dict

import numpy as np
from mpi4py import MPI

from .global_adaptivity import GlobalAdaptivityCalculator

from .tools.misc import divide_in_parts


class GlobalAdaptivityLBCalculator(GlobalAdaptivityCalculator):
    def __init__(
        self,
        configurator,
        global_number_of_sims: int,
        global_ids: list,
        rank: int,
        comm,
    ) -> None:
        """
        Class constructor.

        Parameters
        ----------
        configurator : object of class Config
            Object which has getter functions to get parameters defined in the configuration file.
        logger : object of logging
            Logger defined from the standard package logging
        global_number_of_sims : int
            Total number of simulations in the macro-micro coupled problem.
        global_ids : list
            List of global IDs of simulations living on this rank.
        rank : int
            MPI rank.
        comm : MPI.COMM_WORLD
            Global communicator of MPI.
        """
        super().__init__(configurator, global_number_of_sims, global_ids, rank, comm)
