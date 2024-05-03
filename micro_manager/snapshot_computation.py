#!/usr/bin/env python3
"""
Snapshot Computation is a tool to initialize and adaptively control micro simulations
and create snapshot databases by simulating a macro simulation via prescribes parameters.
This files the class SnapshotComputation which has the following callable public methods:

- solve

This file is directly executable as it consists of a main() function. Upon execution, an object of the class SnapshotComputation is created using a given JSON file,
and the initialize and solve methods are called.

Detailed documentation: https://precice.org/tooling-micro-manager-overview.html
"""

import argparse
import importlib
import os
import sys
from mpi4py import MPI
import numpy as np
import logging
import time

from .config_snapshot import SnapshotConfig
from .read_write_data import HDFParameterfile as hp
from .micro_simulation import create_simulation_class

sys.path.append(os.getcwd())


class SnapshotComputation:
    def __init__(self, config_file: str) -> None:
        """
        Constructor.

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
        fh = logging.FileHandler("snapshot-computation.log")
        fh.setLevel(logging.INFO)

        # Create formatter and add it to handlers
        formatter = logging.Formatter(
            "[" + str(self._rank) + "] %(name)s -  %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter)
        self._logger.addHandler(fh)  # add the handlers to the logger

        self._is_parallel = self._size > 1
        self._micro_sims_have_output = False

        self._logger.info("Provided configuration file: {}".format(config_file))
        self._config = SnapshotConfig(self._logger, config_file)

        # Data names of data to output - TODO: rename but reuse
        self._write_data_names = self._config.get_write_data_names()

        # Data names of data read as macro input parameters - TODO: rename but reuse
        self._read_data_names = self._config.get_read_data_names()

        self._parameter_file = self._config.get_parameter_file_name()

        self._postprocessing = self._config.get_postprocessing()

        self._is_micro_solve_time_required = self._config.write_micro_solve_time()

        self._local_number_of_sims = 0
        self._global_number_of_sims = 0
        self._is_rank_empty = False

        self._initialize()

    # **************
    # Public methods
    # **************

    def solve(self) -> None:
        """
        Solve the problem using given macro parameters.
        - Read macro parameters from the config file, solve micro simulations,
          and write outputs to storage file.
        """

        # TODO Loop over macro parameter increases - potentially enumerate
        for elems in range(self._local_number_of_sims):
            # TODO create micro simulation object
            # Create micro simulation objects
            self._micro_sims = create_simulation_class(self._micro_problem)(
                self._global_ids_of_local_sims[elems]
            )

            micro_sims_input = self._macro_parameters[elems]
            # TODO: replace with own read_data function or just a list of dicts to read data from parameter file
            parameter = ""
            for key, value in micro_sims_input.items():
                parameter += "{} = {}, ".format(key, value)

            micro_sims_output = self._solve_micro_simulations(micro_sims_input)

            # TODO: postprocessing
            if self._postprocessing is True:
                if hasattr(self._micro_problem, "postprocessing") and callable(
                    getattr(self._micro_problem, "postprocessing")
                ):
                    micro_sims_output = self._micro_sims.postprocessing(
                        micro_sims_output
                    )
                else:
                    self._logger.info(
                        "Postprocessing is activated in config file but not available. Skipping postprocessing."
                    )

            # TODO log that the snapshots have been created or that the simulation has crashed
            if micro_sims_output is not None:
                self._data_storage.write_sim_output_to_hdf(
                    self._output_file_path, micro_sims_input, micro_sims_output
                )
                # TODO: Replace with write data to file
                self._logger.info(
                    "Snapshots for macro parameter: {} have been created and stored".format(
                        parameter
                    )
                )
            else:
                self._logger.error(
                    "Simulation with parameter: {} has crashed. Skipping this snapshot".format(
                        parameter
                    )
                )

        # TODO exit loop

        self._logger.info("Snapshot computation finished.")

    # ***************
    # Private methods
    # ***************

    def _initialize(self) -> None:
        """
        Initialize the Snapshot Computation by performing the following tasks:
        - Decompose the domain if the snapshot creation is executed in parallel.
        - Read macro parameter from file
        - Simulate macro information.
        - Create all micro simulation objects and initialize them if an initialize() method is available.
        """

        # Create subdirectory in which the snapshot files are stored

        directory = os.path.dirname(self._parameter_file)
        output_subdirectory = os.path.join(directory, "output")
        os.makedirs(output_subdirectory, exist_ok=True)

        # Create instance for reading and writing data

        self._data_storage = hp(self._logger, self._rank, self._size)

        # Read macro parameters from the parameter file
        self._macro_parameters = self._data_storage.read_parameter_hdf_to_dict(
            self._parameter_file, self._read_data_names
        )

        # Decompose macro parameters if the snapshot creation is executed in parallel
        if self._is_parallel:
            equal_partition = int(len(self._macro_parameters) / self._size)
            rest = len(self._macro_parameters) % self._size
            if self._rank < rest:
                start = self._rank * (equal_partition + 1)
                end = start + equal_partition + 1
            else:
                start = self._rank * equal_partition + rest
                end = start + equal_partition
            self._macro_parameters = self._macro_parameters[start:end]

        # Create file to store output
        self._output_file_path = self._data_storage.create_file(output_subdirectory)

        self._local_number_of_sims = len(self._macro_parameters)
        # TODO: use number of parameters to get number of micro simulations
        self._logger.info(
            "Number of local micro simulations = {}".format(self._local_number_of_sims)
        )

        if self._local_number_of_sims == 0:
            if self._is_parallel:
                self._logger.info(
                    "Rank {} has no micro simulations and hence will not do any computation.".format(
                        self._rank
                    )
                )
                self._is_rank_empty = True
            else:
                raise Exception("Snapshot has no micro simulations.")

        # TODO: I don't think we have communication thus it is not necessary tp know the global idsSnapshot
        nms_all_ranks = np.zeros(self._size, dtype=np.int64)
        # Gather number of micro simulations that each rank has, because this rank needs to know how many micro
        # simulations have been created by previous ranks, so that it can set
        # the correct global IDs
        self._comm.Allgatherv(np.array(self._local_number_of_sims), nms_all_ranks)

        # Get global number of micro simulations
        self._global_number_of_sims = np.sum(nms_all_ranks)

        # Create lists of local and global IDs
        sim_id = np.sum(nms_all_ranks[: self._rank])
        self._global_ids_of_local_sims = []  # DECLARATION
        for i in range(self._local_number_of_sims):
            self._global_ids_of_local_sims.append(sim_id)
            sim_id += 1

        self._micro_sims = [None] * self._local_number_of_sims  # DECLARATION

        self._micro_problem = getattr(
            importlib.import_module(
                self._config.get_micro_file_name(), "MicroSimulation"
            ),
            "MicroSimulation",
        )

        self._micro_sims_have_output = False
        if hasattr(self._micro_problem, "output") and callable(
            getattr(self._micro_problem, "output")
        ):
            self._micro_sims_have_output = True

    def _solve_micro_simulations(self, micro_sims_input: list) -> list:
        """
        Solve all micro simulations and assemble the micro simulations outputs in a list of dicts format.

        Parameters
        ----------
        micro_sims_input : list
            List of dicts in which keys are names of data and the values are the data which are required inputs to
            solve a micro simulation.

        Returns
        -------
        micro_sims_output : list
            List of dicts in which keys are names of data and the values are the data of the output of the micro
            simulations.
        """
        micro_sims_output = [None]
        try:
            start_time = time.time()
            micro_sims_output = self._micro_sims.solve(micro_sims_input)
            end_time = time.time()

            if self._is_micro_solve_time_required:
                micro_sims_output["micro_sim_time"] = end_time - start_time

            return micro_sims_output
        except Exception as e:
            self._logger.error(
                "Micro simulation with input {} has crashed. See next entry on rank for error message".format(
                    micro_sims_input
                )
            )
            self._logger.error(e)
            return None


def main():
    parser = argparse.ArgumentParser(description=".")
    parser.add_argument(
        "config_file",
        type=str,
        help="Path to the JSON config file of the snapshot creation manager.",
    )

    args = parser.parse_args()
    config_file_path = args.config_file
    if not os.path.isabs(config_file_path):
        config_file_path = os.getcwd() + "/" + config_file_path

    manager = SnapshotComputation(config_file_path)

    manager.solve()


if __name__ == "__main__":
    main()
