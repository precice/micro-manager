"""
Class Config provides functionality to read a JSON file and pass the values to the Micro Manager.
"""

import json
import os
import importlib.metadata


class Config:
    """
    Handles the reading of parameters in the JSON configuration file provided by the user. This class is based on
    the config class in https://github.com/precice/fenics-adapter/tree/develop/fenicsadapter
    """

    def __init__(self, config_file_name):
        """
        Constructor of the Config class.

        Parameters
        ----------
        config_file_name : string
            Name of the JSON configuration file
        """
        self._config_file_name = config_file_name
        self._logger = None
        self._micro_file_name = None

        self._precice_config_file_name = None
        self._macro_mesh_name = None
        self._read_data_names = None
        self._write_data_names = None
        self._micro_dt = None

        self._macro_domain_bounds = None
        self._ranks_per_axis = None
        self._micro_output_n = 1
        self._diagnostics_data_names = None

        self._output_micro_sim_time = False
        self._mem_usage_output_type = ""
        self._mem_usage_output_n = 1

        self._interpolate_crash = False

        self._adaptivity = False
        self._adaptivity_type = ""
        self._data_for_adaptivity = dict()
        self._adaptivity_n = 1
        self._adaptivity_history_param = 0.5
        self._adaptivity_coarsening_constant = 0.5
        self._adaptivity_refining_constant = 0.5
        self._adaptivity_every_implicit_iteration = False
        self._adaptivity_similarity_measure = "L1"
        self._adaptivity_output_type = ""
        self._adaptivity_output_n = 1

        self._adaptivity_is_load_balancing = False
        self._load_balancing_n = 1
        self._two_step_load_balancing = False
        self._load_balancing_threshold = 0
        self._balance_inactive_sims = False

        # Snapshot information
        self._parameter_file_name = None
        self._postprocessing_file_name = None
        self._initialize_once = False

        self._output_dir = None

        self._lazy_initialization = False

    def set_logger(self, logger):
        """
        Set the logger for the Config class.

        Parameters
        ----------
        logger : object of logging
            Logger defined from the standard package logging
        """
        self._logger = logger

    def _read_json(self, config_file_name):
        """
        Reads JSON configuration file.

        Parameters
        ----------
        config_file_name : string
            Name of the JSON configuration file
        """
        self._logger.log_info_rank_zero(
            "Micro Manager version: "
            + importlib.metadata.version("micro-manager-precice")
        )

        self._folder = os.path.dirname(os.path.join(os.getcwd(), config_file_name))
        path = os.path.join(self._folder, os.path.basename(config_file_name))
        with open(path, "r") as read_file:
            self._data = json.load(read_file)

        self._logger.log_info_rank_zero("Reading JSON configuration file: " + path)

        # convert paths to python-importable paths
        self._micro_file_name = (
            self._data["micro_file_name"]
            .replace("/", ".")
            .replace("\\", ".")
            .replace(".py", "")
        )

        self._logger.log_info_rank_zero(
            "Micro simulation file name: " + self._data["micro_file_name"]
        )

        try:
            self._output_dir = self._data["output_directory"]
            self._logger.log_info_rank_zero(
                "Logging and metrics output directory: " + self._output_dir
            )
        except BaseException:
            self._logger.log_info_rank_zero(
                "No output directory provided. Output (including logging) will be saved in the current working directory."
            )

        try:
            self._mem_usage_output_type = self._data["memory_usage_output_type"]
            if self._mem_usage_output_type not in ["all", "local", "global"]:
                raise Exception(
                    "Memory usage output can be either 'all', 'local' or 'global'."
                )
            self._logger.log_info_rank_zero(
                "Memory usage output type: " + self._mem_usage_output_type
            )
        except BaseException:
            self._logger.log_info_rank_zero(
                "Micro Manager will not output memory usage."
            )

        try:
            self._mem_usage_output_n = self._data["memory_usage_output_n"]
            self._logger.log_info_rank_zero(
                "Memory usage will be output every "
                + str(self._mem_usage_output_n)
                + " time windows."
            )
        except BaseException:
            self._logger.log_info_rank_zero(
                "No output interval for memory usage output provided. Memory usage will be output every time window."
            )

        try:
            self._write_data_names = self._data["coupling_params"]["write_data_names"]
            if not isinstance(self._write_data_names, list):
                raise Exception("Write data entry is not a list")
            self._logger.log_info_rank_zero(
                "Micro Manager is writing the following data: "
                + str(self._write_data_names)
            )
        except BaseException:
            self._logger.log_info_rank_zero(
                "No write data names provided. Micro manager will only read data from preCICE."
            )

        try:
            self._read_data_names = self._data["coupling_params"]["read_data_names"]
            if not isinstance(self._read_data_names, list):
                raise Exception("Read data entry is not a list")
            self._logger.log_info_rank_zero(
                "Micro Manager is reading the following data: "
                + str(self._read_data_names)
            )
        except BaseException:
            self._logger.log_info_rank_zero(
                "No read data names provided. Micro manager will only write data to preCICE."
            )

        self._micro_dt = self._data["simulation_params"]["micro_dt"]

        try:
            if self._data["diagnostics"]["output_micro_sim_solve_time"] == "True":
                self._output_micro_sim_time = True
                self._write_data_names.append("solve_cpu_time")
        except BaseException:
            self._logger.log_info_rank_zero(
                "Micro manager will not output time required to solve each micro simulation."
            )

    def read_json_micro_manager(self):
        """
        Reads Micro Manager relevant information from JSON configuration file
        and saves the data to the respective instance attributes.
        """
        self._read_json(self._config_file_name)  # Read base information

        self._precice_config_file_name = os.path.join(
            self._folder, self._data["coupling_params"]["precice_config_file_name"]
        )
        self._logger.log_info_rank_zero(
            "preCICE configuration file name: " + self._precice_config_file_name
        )

        self._macro_mesh_name = self._data["coupling_params"]["macro_mesh_name"]
        self._logger.log_info_rank_zero("Macro mesh name: " + self._macro_mesh_name)

        self._macro_domain_bounds = self._data["simulation_params"][
            "macro_domain_bounds"
        ]
        self._logger.log_info_rank_zero(
            "Macro domain bounds: " + str(self._macro_domain_bounds)
        )

        try:
            self._ranks_per_axis = self._data["simulation_params"]["decomposition"]
            if not isinstance(self._ranks_per_axis, list):
                raise Exception("Ranks per axis entry is not a list")
            self._logger.log_info_rank_zero(
                "Axis-wise domain decomposition: " + str(self._ranks_per_axis)
            )
        except BaseException:
            self._logger.log_info_rank_zero(
                "Domain decomposition is not specified, so the Micro Manager will expect to be run in serial."
            )

        try:
            if self._data["simulation_params"]["adaptivity"] == "True":
                self._adaptivity = True
                self._logger.log_info_rank_zero(
                    "Micro Manager will adaptively run micro simulations."
                )
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
            self._logger.log_info_rank_zero(
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

            self._logger.log_info_rank_zero("Adaptivity type: " + self._adaptivity_type)

            if (
                self._data["simulation_params"]["adaptivity_settings"].get(
                    "lazy_initialization"
                )
                == "True"
            ):
                self._lazy_initialization = True

            self._logger.log_info_rank_zero(
                "Micro simulations will be created only when they are required to be active for the very first time."
            )

            self._data_for_adaptivity = self._data["simulation_params"][
                "adaptivity_settings"
            ]["data"]

            self._logger.log_info_rank_zero(
                "Data used for adaptivity: " + str(self._data_for_adaptivity)
            )

            if self._data_for_adaptivity == self._write_data_names:
                self._logger.log_info_rank_zero(
                    "Only micro simulation data is used for similarity computation in adaptivity. This would lead to the"
                    " same set of active and inactive simulations for the entire simulation time. If this is not intended,"
                    " please include macro data as well."
                )

            try:
                self._adaptivity_n = self._data["simulation_params"][
                    "adaptivity_settings"
                ]["adaptivity_every_n_time_windows"]
                self._logger.log_info_rank_zero(
                    "Adaptivity will be computed every "
                    + str(self._adaptivity_n)
                    + " time windows."
                )
            except BaseException:
                self._logger.log_info_rank_zero(
                    "No interval for adaptivity computation provided. Adaptivity will be computed in every time window."
                )

            try:
                self._adaptivity_output_type = self._data["simulation_params"][
                    "adaptivity_settings"
                ]["output_type"]
                if self._adaptivity_output_type not in ["all", "local", "global"]:
                    raise Exception(
                        "Adaptivity output type can be either 'all', 'local' or 'global'."
                    )
                self._logger.log_info_rank_zero(
                    "Adaptivity output type: " + self._adaptivity_output_type
                )
            except BaseException:
                self._logger.log_info_rank_zero(
                    "No adaptivity output type provided. Defaulting to 'local'."
                )
                self._adaptivity_output_type = "local"

            try:
                self._adaptivity_output_n = self._data["simulation_params"][
                    "adaptivity_settings"
                ]["output_n"]
                self._logger.log_info_rank_zero(
                    "Adaptivity metrics will be output every "
                    + str(self._adaptivity_output_n)
                    + " time windows."
                )
            except BaseException:
                self._logger.log_info_rank_zero(
                    "No output interval for adaptivity provided. Adaptivity metrics will be output every time window."
                )

            self._adaptivity_history_param = self._data["simulation_params"][
                "adaptivity_settings"
            ]["history_param"]
            self._logger.log_info_rank_zero(
                "Adaptivity history parameter: " + str(self._adaptivity_history_param)
            )

            self._adaptivity_coarsening_constant = self._data["simulation_params"][
                "adaptivity_settings"
            ]["coarsening_constant"]
            self._logger.log_info_rank_zero(
                "Adaptivity coarsening constant: "
                + str(self._adaptivity_coarsening_constant)
            )

            self._adaptivity_refining_constant = self._data["simulation_params"][
                "adaptivity_settings"
            ]["refining_constant"]
            self._logger.log_info_rank_zero(
                "Adaptivity refining constant: "
                + str(self._adaptivity_refining_constant)
            )

            if (
                "similarity_measure"
                in self._data["simulation_params"]["adaptivity_settings"]
            ):
                self._adaptivity_similarity_measure = self._data["simulation_params"][
                    "adaptivity_settings"
                ]["similarity_measure"]
                self._logger.log_info_rank_zero(
                    "Adaptivity similarity measure: "
                    + str(self._adaptivity_similarity_measure)
                )
            else:
                self._logger.log_info_rank_zero(
                    "No similarity measure provided, using L1 norm as default."
                )
                self._adaptivity_similarity_measure = "L1"

            try:
                adaptivity_every_implicit_iteration = self._data["simulation_params"][
                    "adaptivity_settings"
                ]["every_implicit_iteration"]

                if adaptivity_every_implicit_iteration == "True":
                    self._adaptivity_every_implicit_iteration = True
                    self._logger.log_info_rank_zero(
                        "Micro Manager will compute adaptivity in every implicit iteration, if implicit coupling is done."
                    )

                elif adaptivity_every_implicit_iteration == "False":
                    self._adaptivity_every_implicit_iteration = False
                    self._logger.log_info_rank_zero(
                        "Micro Manager will compute adaptivity once at the start of every time window."
                    )
            except:
                self._logger.log_info_rank_zero(
                    "Micro Manager will compute adaptivity once at the start of every time window."
                )
                self._adaptivity_every_implicit_iteration = False

            self._write_data_names.append("active_state")
            self._write_data_names.append("active_steps")

        try:
            if self._data["simulation_params"]["load_balancing"] == "True":
                self._adaptivity_is_load_balancing = True
                self._logger.log_info_rank_zero(
                    "Micro Manager will dynamically balance micro simulations based on the adaptivity computation."
                )
        except BaseException:
            self._logger.log_info_rank_zero(
                "Micro Manager will not dynamically balance micro simulations based on the adaptivity computation."
            )

        if self._adaptivity_is_load_balancing:
            self._load_balancing_n = self._data["simulation_params"][
                "load_balancing_settings"
            ]["load_balancing_n"]
            self._logger.log_info_rank_zero(
                "Load balancing will be done every "
                + str(self._load_balancing_n)
                + " time windows."
            )

            try:
                if (
                    self._data["simulation_params"]["load_balancing_settings"][
                        "two_step_load_balancing"
                    ]
                    == "True"
                ):
                    self._two_step_load_balancing = True
                    self._logger.log_info_rank_zero(
                        "Micro Manager will use two-step load balancing."
                    )
            except BaseException:
                self._logger.log_info_rank_zero(
                    "Two-step load balancing is not specified. Micro Manager will only try to balance the load in one sweep."  # TODO: Need a better log message here.
                )

            try:
                self._load_balancing_threshold = self._data["simulation_params"][
                    "load_balancing_settings"
                ]["balancing_threshold"]
                self._logger.log_info_rank_zero(
                    "Load balancing threshold: " + str(self._load_balancing_threshold)
                )
            except BaseException:
                self._logger.log_info_rank_zero(
                    "No load balancing threshold provided. The threshold will be set to 0."
                )

            try:
                if (
                    self._data["simulation_params"]["load_balancing_settings"][
                        "balance_inactive_sims"
                    ]
                    == "True"
                ):
                    self._balance_inactive_sims = True
                    self._logger.log_info_rank_zero(
                        "Micro Manager will redistribute inactive simulations in the load balancing."
                    )
            except BaseException:
                self._logger.log_info_rank_zero(
                    "Micro Manager will not redistribute inactive simulations in the load balancing. Only active simulations will be redistributed. Note that this may significantly increase the communication cost of the adaptivity."
                )

        if "interpolate_crash" in self._data["simulation_params"]:
            if self._data["simulation_params"]["interpolate_crash"] == "True":
                self._interpolate_crash = True
                self._logger.log_info_rank_zero(
                    "Micro Manager will interpolate output of crashed micro simulations from its neighbors."
                )

        try:
            diagnostics_data_names = self._data["diagnostics"]["data_from_micro_sims"]
            if not isinstance(diagnostics_data_names, list):
                raise Exception("Diagnostics data entry is not a list")
        except BaseException:
            self._logger.log_info_rank_zero(
                "No diagnostics data is defined. Micro Manager will not output any diagnostics data."
            )

        try:
            self._micro_output_n = self._data["diagnostics"]["micro_output_n"]
        except BaseException:
            self._logger.log_info_rank_zero(
                "Output interval of micro simulations not specified, if output is available then it will be called "
                "in every time window."
            )

    def read_json_snapshot(self):
        """
        Reads Snapshot relevant information from JSON configuration file
        """
        self._read_json(self._config_file_name)  # Read base information

        self._logger.log_info_rank_zero(
            "Reading JSON configuration file: " + self._config_file_name
        )

        self._logger.log_info_rank_zero("Micro Manager is running in snapshot mode.")

        self._parameter_file_name = os.path.join(
            self._folder, self._data["coupling_params"]["parameter_file_name"]
        )
        self._logger.log_info_rank_zero(
            "Parameter file name: " + self._parameter_file_name
        )

        try:
            self._postprocessing_file_name = (
                self._data["snapshot_params"]["post_processing_file_name"]
                .replace("/", ".")
                .replace("\\", ".")
                .replace(".py", "")
            )
            self._logger.log_info_rank_zero(
                "Post-processing file name: " + self._postprocessing_file_name
            )
        except BaseException:
            self._logger.log_info_rank_zero(
                "No post-processing file name provided. Snapshot computation will not perform any post-processing."
            )
            self._postprocessing_file_name = None

        try:
            diagnostics_data_names = self._data["diagnostics"]["data_from_micro_sims"]
            if not isinstance(diagnostics_data_names, list):
                raise Exception("Diagnostics data entry is not a list")
            self._logger.log_info_rank_zero(
                "Diagnostics data: " + str(diagnostics_data_names)
            )
        except BaseException:
            self._logger.log_info_rank_zero(
                "No diagnostics data is defined. Snapshot computation will not output any diagnostics data."
            )

        try:
            if self._data["snapshot_params"]["initialize_once"] == "True":
                self._initialize_once = True
                self._logger.log_info_rank_zero(
                    "Micro Manager will initialize only one micro simulations object for snapshot computation."
                )
        except BaseException:
            self._logger.log_info_rank_zero(
                "For each snapshot a new micro simulation object will be created."
            )

    def get_precice_config_file_name(self):
        """
        Get the name of the preCICE XML configuration file.

        Returns
        -------
        config_file_name : string
            Name of the preCICE XML configuration file.
        """
        return self._precice_config_file_name

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

    def get_adaptivity_n(self):
        """
        Get the frequency of adaptivity computation.

        Returns
        -------
        adaptivity_n : int
            Frequency of adaptivity computation, as a multiple of time windows.
        """
        return self._adaptivity_n

    def get_adaptivity_output_type(self):
        """
        Get the type of adaptivity output.

        Returns
        -------
        adaptivity_output_type : str
            Type of adaptivity output, can be "all", "local" or "global".
        """
        return self._adaptivity_output_type

    def get_adaptivity_output_n(self):
        """
        Get the output frequency of adaptivity metrics.

        Returns
        -------
        adaptivity_output_n : int
            Output frequency of adaptivity metrics, so output every N timesteps
        """
        return self._adaptivity_output_n

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

    def is_adaptivity_with_load_balancing(self):
        """
        Check if adaptivity computation needs to be done with load balancing.

        Returns
        -------
        adaptivity_is_load_balancing : bool
            True if adaptivity computation needs to be done with load balancing, False otherwise.
        """
        return self._adaptivity_is_load_balancing

    def get_load_balancing_n(self):
        """
        Get the load balancing frequency.

        Returns
        -------
        load_balancing_n : int
            Load balancing frequency
        """
        return self._load_balancing_n

    def is_load_balancing_two_step(self):
        """
        Check if two-step load balancing is required.

        Returns
        -------
        two_step_load_balancing : bool
            True if two-step load balancing is required, False otherwise.
        """
        return self._two_step_load_balancing

    def get_load_balancing_threshold(self):
        """
        Get the load balancing threshold to control how balanced the micro simulations need to be.

        Returns
        -------
        load_balancing_threshold : float
            Load balancing threshold
        """
        return self._load_balancing_threshold

    def balance_inactive_sims(self):
        """
        Check if inactive simulations are to be redistributed in the load balancing.

        Returns
        -------
        balance_inactive_sims : bool
            True if inactive simulations are to be redistributed in the load balancing, False otherwise.
        """
        return self._balance_inactive_sims

    def initialize_sims_lazily(self):
        """
        Check if simulations are to be created only when they are required to be active for the very first time.

        Returns
        -------
        adaptivity : bool
            True if micro simulations are created only when needed, False otherwise.

        """
        return self._lazy_initialization

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

    def get_output_dir(self):
        """
        Get the name of the output directory.

        Returns
        -------
        output_dir : string
            Name of the output folder.
        """
        return self._output_dir

    def get_memory_usage_output_type(self):
        """
        Get the type of memory usage output.

        Returns
        -------
        mem_usage_output_type : str
            Type of adaptivity output, can be "all", "local" or "global".
        """
        return self._mem_usage_output_type

    def get_memory_usage_output_n(self):
        """
        Get the output frequency of memory usage.

        Returns
        -------
        mem_usage_output_n : int
            Output frequency of memory usage, so output every N timesteps
        """
        return self._mem_usage_output_n
