"""
Class Config provides functionality to read a JSON file and pass the values to the Micro Manager.
"""

import json
import os
from warnings import warn


class SnapshotConfig:
    """
    Handles the reading of parameters in the JSON configuration file for the snapshot computation provided by the user. This class is based on
    the config class in https://github.com/precice/fenics-adapter/tree/develop/fenicsadapter
    """

    def __init__(self, logger, config_filename):
        """
        Constructor of the Config class.

        Parameters
        ----------
        config_filename : string
            Name of the JSON configuration file
        """
        self._logger = logger

        self._micro_file_name = None

        self._parameter_file_name = None

        self._read_data_names = dict()
        self._write_data_names = dict()

        self._postprocessing = False

        self._output_micro_sim_time = False

        self.read_json(config_filename)

    def read_json(self, config_filename):
        """
        Reads JSON adapter configuration file and saves the data to the respective instance attributes.

        Parameters
        ----------
        config_filename : string
            Name of the JSON configuration file
        """
        folder = os.path.dirname(os.path.join(os.getcwd(), config_filename))
        path = os.path.join(folder, os.path.basename(config_filename))
        with open(path, "r") as read_file:
            data = json.load(read_file)

        # convert paths to python-importable paths
        self._micro_file_name = (
            data["micro_file_name"]
            .replace("/", ".")
            .replace("\\", ".")
            .replace(".py", "")
        )

        self._parameter_file_name = os.path.join(
            folder, data["snapshot_params"]["parameter_file_name"]
        )

        try:
            self._write_data_names = data["snapshot_params"]["write_data_names"]
            assert isinstance(
                self._write_data_names, dict
            ), "Write data entry is not a dictionary"
            for key, value in self._write_data_names.items():
                if value == "scalar":
                    self._write_data_names[key] = False
                elif value == "vector":
                    self._write_data_names[key] = True
                else:
                    raise Exception(
                        "Write data dictionary as a value other than 'scalar' or 'vector'"
                    )
        except BaseException:
            self._logger.error(
                "No write data names provided. Snapshot computation will not yield any results."
            )

        try:
            self._read_data_names = data["snapshot_params"]["read_data_names"]
            assert isinstance(
                self._read_data_names, dict
            ), "Read data entry is not a dictionary"
            for key, value in self._read_data_names.items():
                if value == "scalar":
                    self._read_data_names[key] = False
                elif value == "vector":
                    self._read_data_names[key] = True
                else:
                    raise Exception(
                        "Read data dictionary as a value other than 'scalar' or 'vector'"
                    )
        except BaseException:
            self._logger.error(
                "No read data names provided. Snapshot computation is not able to yield results without information."
            )

        self._postprocessing = data["snapshot_params"]["postprocessing"]

        try:
            diagnostics_data_names = data["diagnostics"]["data_from_micro_sims"]
            assert isinstance(
                diagnostics_data_names, dict
            ), "Diagnostics data is not a dictionary"
            for key, value in diagnostics_data_names.items():
                if value == "scalar":
                    self._write_data_names[key] = False
                elif value == "vector":
                    self._write_data_names[key] = True
                else:
                    raise Exception(
                        "Diagnostics data dictionary as a value other than 'scalar' or 'vector'"
                    )
        except BaseException:
            self._logger.info(
                "No diagnostics data is defined. Micro Manager will not output any diagnostics data."
            )

        try:
            if data["diagnostics"]["output_micro_sim_solve_time"]:
                self._output_micro_sim_time = True
                self._write_data_names["micro_sim_time"] = False
        except BaseException:
            self._logger.info(
                "Micro manager will not output time required to solve each micro simulation in each time step."
            )

    def get_config_file_name(self):
        """
        Get the name of the JSON configuration file.

        Returns
        -------
        config_file_name : string
            Name of the JSON configuration file provided to the Config class.
        """
        return self._config_file_name

    def get_parameter_file_name(self):
        """
        Get the name of the parameter file.

        Returns
        -------
        parameter_file_name : string
            Name of the hdf5 file containing the macro parameters.
        """

        return self._parameter_file_name

    def get_read_data_names(self):
        """
        Get the user defined dictionary carrying information of the data to be read from preCICE.

        Returns
        -------
        read_data_names: dict_like
            A dictionary containing the names of the data to be read from preCICE as keys and information on whether
            the data are scalar or vector as values.
        """
        return self._read_data_names

    def get_write_data_names(self):
        """
        Get the user defined dictionary carrying information of the data to be written to preCICE.

        Returns
        -------
        write_data_names: dict_like
            A dictionary containing the names of the data to be written to preCICE as keys and information on whether
            the data are scalar or vector as values.
        """
        return self._write_data_names

    def get_micro_file_name(self):
        """
        Get the path to the Python script of the micro-simulation.

        Returns
        -------
        micro_file_name : string
            String carrying the path to the Python script of the micro-simulation.
        """
        return self._micro_file_name

    def write_micro_solve_time(self):
        """
        Depending on user input, micro manager will calculate execution time of solve() step of every micro simulation

        Returns
        -------
        output_micro_sim_time : bool
            True if micro simulation solve time is required.
        """
        return self._output_micro_sim_time

    def get_postprocessing(self):
        """
        Depending on user input, snapshot will perform postprocessing for every micro simulation
        Returns
        -------
        postprocessing : bool
            True if postprocessing is required.
        """
        return self._postprocessing
