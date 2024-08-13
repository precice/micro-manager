#!/usr/bin/env python3
"""
Snapshot Computation is a tool within the Micro Manager to initialize micro simulations and create a snapshot database of theirs outputs by running them with a set of prescribed input parameters.
This files the class SnapshotComputation which has the following callable public methods:

- solve
- initialize

Detailed documentation: https://precice.org/tooling-micro-manager-overview.html
"""

import importlib
import os
import sys
import time

import numpy as np

from micro_manager.micro_manager import MicroManager
from .dataset import ReadWriteHDF
from micro_manager.micro_simulation import create_simulation_class

sys.path.append(os.getcwd())


class MicroManagerSnapshot(MicroManager):
    def __init__(self, config_file: str) -> None:
        """
        Constructor.

        Parameters
        ----------
        config_file : string
            Name of the JSON configuration file (provided by the user).
        """
        super().__init__(config_file)
        self._config.read_json_snapshot()

        # Path to the parameter file containing input parameters for micro simulations
        self._parameter_file = self._config.get_parameter_file_name()
        # Get name of pos-processing script
        self._post_processing_file_name = self._config.get_postprocessing_file_name()

        # Check if simulation object can be re-used.
        self._initialize_once = self._config.create_single_sim_object()
        # Collect crashed indices
        self._crashed_snapshots = []  # Declaration

    # **************
    # Public methods
    # **************

    def solve(self) -> None:
        """
        Solve the problem by iterating over a set macro parameters.
        - Create micro simulation object.
        - Post-process micro output.
        - Write output to database.
        - Merge output in parallel run.
        """

        # Loop over all macro parameters
        for elems in range(self._local_number_of_sims):
            # initialize micro simulation
            if elems == 0:
                self._micro_sims = create_simulation_class(self._micro_problem)(
                    self._global_ids_of_local_sims[0]
                )
            else:
                if not self._initialize_once:
                    self._micro_sims = create_simulation_class(self._micro_problem)(
                        self._global_ids_of_local_sims[elems]
                    )

            micro_sims_input = self._macro_parameters[elems]
            # Solve micro simulation
            micro_sims_output = self._solve_micro_simulation(micro_sims_input)

            # Write output to file
            if micro_sims_output is not None:
                # Post-processing
                if self._post_processing_file_name is not None:
                    # Attempt importing post-processing script
                    try:
                        post_processing = getattr(
                            importlib.import_module(
                                self._post_processing_file_name, "Postprocessing"
                            ),
                            "Postprocessing",
                        )
                        if hasattr(post_processing, "postprocessing") and callable(
                            getattr(post_processing, "postprocessing")
                        ):
                            micro_sims_output = post_processing.postprocessing(
                                micro_sims_output
                            )
                        else:
                            self._logger.info(
                                "No post-processing script with the provided path found. Skipping post-processing."
                            )
                            self._post_processing_file_name = None
                    except Exception:
                        self._logger.info(
                            "No post-processing script with the provided path found. Skipping post-processing."
                        )
                        self._post_processing_file_name = None
                self._data_storage.write_output_to_hdf(
                    self._output_file_path,
                    micro_sims_input,
                    micro_sims_output,
                    elems,
                    len(self._macro_parameters),
                )
            # Log error and write macro data to database if simulation has crashed
            else:
                self._logger.info("Skipping snapshot storage for crashed simulation.")
                self._data_storage.write_output_to_hdf(
                    self._output_file_path,
                    micro_sims_input,
                    None,
                    elems,
                    len(self._macro_parameters),
                )

                self._crashed_snapshots.append(elems)

        # Write positions of crashed snapshots in database to database
        if len(self._crashed_snapshots) > 0:
            self._data_storage.write_crashed_snapshots(
                self._output_file_path, self._crashed_snapshots
            )
        self._data_storage.set_status(self._output_file_path, "none")

        # Merge output files
        if self._is_parallel:
            self._logger.info(
                "Snapshots have been computed and stored. Merging output files"
            )
            self._data_storage.set_status(self._output_file_path, "reading/deleting")
            list_of_output_files = self._comm.gather(self._file_name, 0)
            if self._rank == 0:
                self._data_storage.collect_output_files(
                    self._output_subdirectory,
                    list_of_output_files,
                    self._parameter_space_size,
                )
        self._logger.info("Snapshot computation completed.")

    def initialize(self) -> None:
        """
        Initialize the Snapshot Computation by performing the following tasks:
        - Distribute the parameter data equally if the snapshot creation is executed in parallel.
        - Read macro parameter from parameter file.
        - Create output subdirectory and file paths to store output.
        - Import micro simulation.
        """

        # Create subdirectory to store output files in
        directory = os.path.dirname(self._parameter_file)
        self._output_subdirectory = os.path.join(directory, "output")
        os.makedirs(self._output_subdirectory, exist_ok=True)

        # Create object responsible for reading parameters and writing simulation output
        self._data_storage = ReadWriteHDF(self._logger)

        self._parameter_space_size = self._data_storage.get_parameter_space_size(
            self._parameter_file
        )
        # Read macro parameters from the parameter file
        # Decompose parameters if the snapshot creation is executed in parallel
        if self._is_parallel:
            equal_partition = int(self._parameter_space_size / self._size)
            rest = self._parameter_space_size % self._size
            if self._rank < rest:
                start = self._rank * (equal_partition + 1)
                end = start + equal_partition + 1
            else:
                start = self._rank * equal_partition + rest
                end = start + equal_partition
            self._macro_parameters = self._data_storage.read_hdf(
                self._parameter_file, self._read_data_names, start, end
            )
        else:
            self._macro_parameters = self._data_storage.read_hdf(
                self._parameter_file,
                self._read_data_names,
                0,
                self._parameter_space_size,
            )

        # Create database file to store output from a rank in
        if self._is_parallel:
            self._file_name = "snapshot_data_{}.hdf5".format(self._rank)
        else:
            self._file_name = "snapshot_data.hdf5"
        self._output_file_path = os.path.join(
            self._output_subdirectory, self._file_name
        )
        self._data_storage.create_file(self._output_file_path)
        self._logger.info("Output file created: {}".format(self._output_file_path))
        self._local_number_of_sims = len(self._macro_parameters)
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

    # ***************
    # Private methods
    # ***************

    def _solve_micro_simulation(self, micro_sims_input: dict) -> dict | None:
        """
        Solve a single micro simulation.

        Parameters
        ----------
        micro_sims_input : dict
            Keys are names of data and the values are the data which are required inputs to
            solve a micro simulation.

        Returns
        -------
        micro_sims_output : dict | None
            Dict in which keys are names of data and the values are the data of the output of the micro
            simulations. The return type is None if the simulation has crashed.
        """
        try:
            start_time = time.time()
            micro_sims_output = self._micro_sims.solve(micro_sims_input, self._micro_dt)
            end_time = time.time()

            if self._is_micro_solve_time_required:
                micro_sims_output["micro_sim_time"] = end_time - start_time

            return micro_sims_output
        # Handle simulation crash
        except Exception as e:
            self._logger.error(
                "Micro simulation with input {} has crashed. See next entry on this rank for error message".format(
                    micro_sims_input
                )
            )
            self._logger.error(e)
            return None
