"""
Class Config provides functionality to read a JSON file and pass the values to the Micro Manager.
"""

import json
import os
from warnings import warn


class Config:
    """
    Handles the reading of parameters in the JSON configuration file provided by the user. This class is based on
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

        self._config_file_name = None
        self._macro_mesh_name = None
        self._read_data_names = dict()
        self._write_data_names = dict()
        self._micro_dt = None

        self._macro_domain_bounds = None
        self._ranks_per_axis = None
        self._micro_output_n = 1
        self._diagnostics_data_names = dict()

        self._output_micro_sim_time = False

        self._interpolate_crash = False

        self._adaptivity = False
        self._adaptivity_type = "local"
        self._data_for_adaptivity = dict()
        self._adaptivity_history_param = 0.5
        self._adaptivity_coarsening_constant = 0.5
        self._adaptivity_refining_constant = 0.5
        self._adaptivity_every_implicit_iteration = False
        self._adaptivity_similarity_measure = "L1"

        # Snapshot information
        self._parameter_file_name = None
        self._postprocessing_file_name = None
        self._initialize_once = False

        self._output_micro_sim_time = False

        self.read_json(config_filename)

    def read_json(self, config_filename):
        """
        Reads JSON configuration file.

        Parameters
        ----------
        config_filename : string
            Name of the JSON configuration file
        """
        self._folder = os.path.dirname(os.path.join(os.getcwd(), config_filename))
        path = os.path.join(self._folder, os.path.basename(config_filename))
        with open(path, "r") as read_file:
            self._data = json.load(read_file)

        # convert paths to python-importable paths
        self._micro_file_name = (
            self._data["micro_file_name"]
            .replace("/", ".")
            .replace("\\", ".")
            .replace(".py", "")
        )

        try:
            self._write_data_names = self._data["coupling_params"]["write_data_names"]
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
            self._logger.info(
                "No write data names provided. Micro manager will only read data from preCICE."
            )

        try:
            self._read_data_names = self._data["coupling_params"]["read_data_names"]
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
            self._logger.info(
                "No read data names provided. Micro manager will only write data to preCICE."
            )

        self._micro_dt = self._data["simulation_params"]["micro_dt"]

        try:
            if self._data["diagnostics"]["output_micro_sim_solve_time"]:
                self._output_micro_sim_time = True
                self._write_data_names["micro_sim_time"] = False
        except BaseException:
            self._logger.info(
                "Micro manager will not output time required to solve each micro simulation in each time step."
            )

    def read_json_micro_manager(self):
        """
        Reads Micro Manager relevant information from JSON configuration file
        and saves the data to the respective instance attributes.
        """
        self._config_file_name = os.path.join(
            self._folder, self._data["coupling_params"]["config_file_name"]
        )
        self._macro_mesh_name = self._data["coupling_params"]["macro_mesh_name"]

        self._macro_domain_bounds = self._data["simulation_params"][
            "macro_domain_bounds"
        ]

        try:
            self._ranks_per_axis = self._data["simulation_params"]["decomposition"]
        except BaseException:
            self._logger.info(
                "Domain decomposition is not specified, so the Micro Manager will expect to be run in serial."
            )

        try:
            if self._data["simulation_params"]["adaptivity"] == "True":
                self._adaptivity = True
                if not self._data["simulation_params"]["adaptivity_settings"]:
                    raise Exception(
                        "Adaptivity is turned on but no adaptivity settings are provided."
                    )
            else:
                self._adaptivity = False
                if self._data["simulation_params"]["adaptivity_settings"]:
                    raise Exception(
                        "Adaptivity settings are provided but adaptivity is turned off."
                    )
        except BaseException:
            self._logger.info(
                "Micro Manager will not adaptively run micro simulations, but instead will run all micro simulations."
            )

        if self._adaptivity:
            if (
                self._data["simulation_params"]["adaptivity_settings"]["type"]
                == "local"
            ):
                self._adaptivity_type = "local"
            elif (
                self._data["simulation_params"]["adaptivity_settings"]["type"]
                == "global"
            ):
                self._adaptivity_type = "global"
            else:
                raise Exception("Adaptivity type can be either local or global.")

            exchange_data = {**self._read_data_names, **self._write_data_names}
            for dname in self._data["simulation_params"]["adaptivity_settings"]["data"]:
                self._data_for_adaptivity[dname] = exchange_data[dname]

            if self._data_for_adaptivity.keys() == self._write_data_names.keys():
                warn(
                    "Only micro simulation data is used for similarity computation in adaptivity. This would lead to the"
                    " same set of active and inactive simulations for the entire simulation time. If this is not intended,"
                    " please include macro simulation data as well."
                )

            self._adaptivity_history_param = self._data["simulation_params"][
                "adaptivity_settings"
            ]["history_param"]
            self._adaptivity_coarsening_constant = self._data["simulation_params"][
                "adaptivity_settings"
            ]["coarsening_constant"]
            self._adaptivity_refining_constant = self._data["simulation_params"][
                "adaptivity_settings"
            ]["refining_constant"]

            if (
                "similarity_measure"
                in self._data["simulation_params"]["adaptivity_settings"]
            ):
                self._adaptivity_similarity_measure = self._data["simulation_params"][
                    "adaptivity_settings"
                ]["similarity_measure"]
            else:
                self._logger.info(
                    "No similarity measure provided, using L1 norm as default"
                )
                self._adaptivity_similarity_measure = "L1"

            adaptivity_every_implicit_iteration = self._data["simulation_params"][
                "adaptivity_settings"
            ]["every_implicit_iteration"]

            if adaptivity_every_implicit_iteration == "True":
                self._adaptivity_every_implicit_iteration = True
            elif adaptivity_every_implicit_iteration == "False":
                self._adaptivity_every_implicit_iteration = False

            if not self._adaptivity_every_implicit_iteration:
                self._logger.info(
                    "Micro Manager will compute adaptivity once at the start of every time window"
                )

            self._write_data_names["active_state"] = False
            self._write_data_names["active_steps"] = False

        if "interpolate_crash" in self._data["simulation_params"]:
            if self._data["simulation_params"]["interpolate_crash"] == "True":
                self._interpolate_crash = True

        try:
            diagnostics_data_names = self._data["diagnostics"]["data_from_micro_sims"]
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
            self._micro_output_n = self._data["diagnostics"]["micro_output_n"]
        except BaseException:
            self._logger.info(
                "Output interval of micro simulations not specified, if output is available then it will be called "
                "in every time window."
            )

    def read_json_snapshot(self):
        self._parameter_file_name = os.path.join(
            self._folder, self._data["coupling_params"]["parameter_file_name"]
        )

        try:
            self._postprocessing_file_name = (
                self._data["snapshot_params"]["post_processing_file_name"]
                .replace("/", ".")
                .replace("\\", ".")
                .replace(".py", "")
            )
        except BaseException:
            self._logger.info(
                "No post-processing file name provided. Snapshot computation will not perform any post-processing."
            )
            self._postprocessing_file_name = None

        try:
            diagnostics_data_names = self._data["diagnostics"]["data_from_micro_sims"]
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
                        "Diagnostics data dictionary has a value other than 'scalar' or 'vector'"
                    )
        except BaseException:
            self._logger.info(
                "No diagnostics data is defined. Snapshot computation will not output any diagnostics data."
            )

        try:
            if self._data["snapshot_params"]["initialize_once"] == "True":
                self._initialize_once = True
        except BaseException:
            self._logger.info(
                "For each snapshot a new micro simulation object will be created"
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

    def get_macro_mesh_name(self):
        """
        Get the name of the macro mesh. This name is expected to be the same as the one defined in the preCICE
        configuration file.

        Returns
        -------
        macro_mesh_name : string
            Name of the macro mesh as stated in the JSON configuration file.

        """
        return self._macro_mesh_name

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

    def get_macro_domain_bounds(self):
        """
        Get the upper and lower bounds of the macro domain.

        Returns
        -------
        macro_domain_bounds : list
            List containing upper and lower bounds of the macro domain.
            Format in 2D is [x_min, x_max, y_min, y_max]
            Format in 2D is [x_min, x_max, y_min, y_max, z_min, z_max]
        """
        return self._macro_domain_bounds

    def get_ranks_per_axis(self):
        """
        Get the ranks per axis for a parallel simulation

        Returns
        -------
        ranks_per_axis : list
            List containing ranks in the x, y and z axis respectively.
        """
        return self._ranks_per_axis

    def get_micro_file_name(self):
        """
        Get the path to the Python script of the micro-simulation.

        Returns
        -------
        micro_file_name : string
            String carrying the path to the Python script of the micro-simulation.
        """
        return self._micro_file_name

    def get_micro_output_n(self):
        """
        Get the micro output frequency

        Returns
        -------
        micro_output_n : int
            Output frequency of micro simulations, so output every N timesteps
        """
        return self._micro_output_n

    def write_micro_solve_time(self):
        """
        Depending on user input, micro manager will calculate execution time of solve() step of every micro simulation

        Returns
        -------
        output_micro_sim_time : bool
            True if micro simulation solve time is required.
        """
        return self._output_micro_sim_time

    def turn_on_adaptivity(self):
        """
        Boolean stating whether adaptivity is ot or not.

        Returns
        -------
        adaptivity : bool
            True is adaptivity settings are done, False otherwise.

        """
        return self._adaptivity

    def get_adaptivity_type(self):
        """
        String stating type of adaptivity computation, either "local" or "global".

        Returns
        -------
        adaptivity_type : str
            Either "local" or "global" depending on the type of adaptivity computation
        """
        return self._adaptivity_type

    def get_data_for_adaptivity(self):
        """
        Get names of data to be used for similarity distance calculation in adaptivity

        Returns
        -------
        data_for_adaptivity : dict_like
            A dictionary containing the names of the data to be used in adaptivity as keys and information on whether
            the data are scalar or vector as values.
        """
        return self._data_for_adaptivity

    def get_adaptivity_hist_param(self):
        """
        Get adaptivity history parameter.
        More details: https://precice.org/tooling-micro-manager-configuration.html#adaptivity

        Returns
        -------
        adaptivity_hist_param : float
            Adaptivity history parameter
        """
        return self._adaptivity_history_param

    def get_adaptivity_coarsening_const(self):
        """
        Get adaptivity coarsening constant.
        More details: https://precice.org/tooling-micro-manager-configuration.html#adaptivity

        Returns
        -------
        adaptivity_coarsening_constant : float
            Adaptivity coarsening constant
        """
        return self._adaptivity_coarsening_constant

    def get_adaptivity_refining_const(self):
        """
        Get adaptivity refining constant.
        More details: https://precice.org/tooling-micro-manager-configuration.html#adaptivity

        Returns
        -------
        adaptivity_refining_constant : float
            Adaptivity refining constant
        """
        return self._adaptivity_refining_constant

    def get_adaptivity_similarity_measure(self):
        """
        Get measure to be used to calculate similarity between pairs of simulations.
        More details: https://precice.org/tooling-micro-manager-configuration.html#adaptivity

        Returns
        -------
        adaptivity_similarity_measure : str
            String of measure to be used in calculating similarity between pairs of simulations.
        """
        return self._adaptivity_similarity_measure

    def is_adaptivity_required_in_every_implicit_iteration(self):
        """
        Check if adaptivity needs to be calculated in every time iteration or every time window.

        Returns
        -------
        adaptivity_every_implicit_iteration : bool
            True if adaptivity needs to be calculated in every time iteration, False otherwise.
        """
        return self._adaptivity_every_implicit_iteration

    def get_micro_dt(self):
        """
        Get the size of the micro time window.

        Returns
        -------
        micro_time_window : float
            Size of the micro time window.
        """
        return self._micro_dt

    def get_parameter_file_name(self):
        """
        Get the name of the parameter file.

        Returns
        -------
        parameter_file_name : string
            Name of the hdf5 file containing the macro parameters.
        """

        return self._parameter_file_name

    def get_postprocessing_file_name(self):
        """
        Depending on user input, snapshot computation will perform post-processing for every micro simulation before writing output to a file.

        Returns
        -------
        postprocessing : str
            Name of post-processing script.
        """
        return self._postprocessing_file_name

    def interpolate_crashed_micro_sim(self):
        """
        Check if user wants crashed micro simulations to be interpolated.

        Returns
        -------
        interpolate_crash : bool
            True if crashed micro simulations need to be interpolated, False otherwise.
        """
        return self._interpolate_crash

    def create_single_sim_object(self):
        """
        Check if multiple snapshots can be computed on a single micro simulation object.

        Returns
        -------
        initialize_once : bool
            True if initialization is done only once, False otherwise.
        """
        return self._initialize_once
