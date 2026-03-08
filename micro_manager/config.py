"""
Class Config provides functionality to read a JSON file and pass the values to the Micro Manager.
"""

import json
import os
import importlib.metadata


class ConfigError(Exception):
    """
    Raised when a required configuration parameter is missing or has an invalid value.
    """

    pass


def _get_required(data, key, section_path=""):
    """
    Return data[key] or raise ConfigError with clear path.
    """
    if key not in data:
        path = f"{section_path}.{key}" if section_path else key
        raise ConfigError(
            f"Missing required configuration parameter: '{path}'. "
            f"Please ensure this parameter is set in your JSON configuration file."
        )
    return data[key]


def _get_optional(data, key, default, section_path=""):
    """
    Return data[key] if present, else default.
    """
    return data.get(key, default)


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
        self._micro_stateless = False

        self._precice_config_file_name = None
        self._macro_mesh_name = None
        self._read_data_names = None
        self._write_data_names = None
        self._micro_dt = None

        self._macro_domain_bounds = None
        self._ranks_per_axis = None
        self._micro_output_n = 1
        self._diagnostics_data_names = None

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
        self._adaptivity_similarity_measure = "L2rel"
        self._adaptivity_output_type = ""
        self._adaptivity_output_n = 1

        self._adaptivity_is_load_balancing = False
        self._load_balancing_n = 1
        self._load_balancing_threshold = 0
        self._balance_inactive_sims = False

        # Snapshot information
        self._parameter_file_name = None
        self._postprocessing_file_name = None
        self._initialize_once = False
        self._output_file_name = "snapshot_data"

        self._output_dir = None

        self._lazy_initialization = False

        # Model Adaptivity information
        self._m_adap = False
        self._m_adap_micro_file_names = None
        self._m_adap_micro_stateless = None
        self._m_adap_switching_function = None

        # Tasking
        self._task_is_slurm = False
        self._task_backend = "socket"
        self._task_num_workers = 1
        self._task_mpi_impl = "open"
        self._task_pinning_hostfile = "./hosts.micro"

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

        # Mandatory: micro_file_name
        micro_file_name_raw = _get_required(self._data, "micro_file_name")
        self._micro_file_name = (
            micro_file_name_raw.replace("/", ".")
            .replace("\\", ".")
            .replace(".py", "")
        )
        self._logger.log_info_rank_zero(
            "Micro simulation file name: " + micro_file_name_raw
        )

        # Mandatory: coupling_params and simulation_params (used by both coupling and snapshot)
        coupling_params = _get_required(self._data, "coupling_params", "root")
        simulation_params = _get_required(self._data, "simulation_params", "root")

        # Optional: micro_stateless (default: False)
        self._micro_stateless = _get_optional(
            self._data, "micro_stateless", False
        )
        if self._micro_stateless:
            self._logger.log_info_rank_zero(
                "Only creating one full instance of MicroSimulation."
            )
        else:
            self._logger.log_info_rank_zero(
                "Creating an instance of MicroSimulation for each mesh vertex."
            )

        # Optional: output_directory (default: None, logs to current directory)
        self._output_dir = _get_optional(
            self._data, "output_directory", None
        )
        if self._output_dir is not None:
            self._logger.log_info_rank_zero(
                "Logging and metrics output directory: " + self._output_dir
            )
        else:
            self._logger.log_info_rank_zero(
                "No output directory provided. Output (including logging) will be saved in the current working directory."
            )

        # Optional: memory_usage_output_type (default: "")
        self._mem_usage_output_type = _get_optional(
            self._data, "memory_usage_output_type", ""
        )
        if self._mem_usage_output_type:
            if self._mem_usage_output_type not in ["all", "local", "global"]:
                raise ConfigError(
                    "memory_usage_output_type must be one of 'all', 'local', or 'global'. "
                    f"Got: '{self._mem_usage_output_type}'"
                )
            self._logger.log_info_rank_zero(
                "Memory usage output type: " + self._mem_usage_output_type
            )
        else:
            self._logger.log_info_rank_zero(
                "Micro Manager will not output memory usage."
            )

        # Optional: memory_usage_output_n (default: 1)
        self._mem_usage_output_n = _get_optional(
            self._data, "memory_usage_output_n", 1
        )
        self._logger.log_info_rank_zero(
            "Memory usage will be output every "
            + str(self._mem_usage_output_n)
            + " time windows."
        )

        # Optional: write_data_names (default: None, read-only mode)
        self._write_data_names = _get_optional(
            coupling_params, "write_data_names", None, "coupling_params"
        )
        if self._write_data_names is not None:
            if not isinstance(self._write_data_names, list):
                raise ConfigError(
                    "coupling_params.write_data_names must be a list."
                )
            self._logger.log_info_rank_zero(
                "Micro Manager is writing the following data: "
                + str(self._write_data_names)
            )
        else:
            self._logger.log_info_rank_zero(
                "No write data names provided. Micro manager will only read data from preCICE."
            )

        # Optional: read_data_names (default: None, write-only mode)
        self._read_data_names = _get_optional(
            coupling_params, "read_data_names", None, "coupling_params"
        )
        if self._read_data_names is not None:
            if not isinstance(self._read_data_names, list):
                raise ConfigError(
                    "coupling_params.read_data_names must be a list."
                )
            self._logger.log_info_rank_zero(
                "Micro Manager is reading the following data: "
                + str(self._read_data_names)
            )
        else:
            self._logger.log_info_rank_zero(
                "No read data names provided. Micro manager will only write data to preCICE."
            )

        # Mandatory: micro_dt
        self._micro_dt = _get_required(
            simulation_params, "micro_dt", "simulation_params"
        )

        # Optional: tasking
        tasking_config = _get_optional(self._data, "tasking", None)
        if tasking_config:
            backend = _get_required(tasking_config, "backend", "tasking")
            if backend not in ["mpi", "socket"]:
                raise ConfigError(
                    "tasking.backend must be either 'mpi' or 'socket'. "
                    f"Got: '{backend}'"
                )
            self._task_backend = backend
            self._task_is_slurm = _get_optional(tasking_config, "is_slurm", False)
            self._task_num_workers = _get_optional(
                tasking_config, "num_workers", self._task_num_workers
            )
            if self._task_is_slurm and backend == "mpi":
                raise ConfigError(
                    "MPI backend not supported on SLURM systems."
                )
            mpi_impl = _get_optional(tasking_config, "mpi_impl", self._task_mpi_impl)
            if mpi_impl not in ["open", "intel"]:
                raise ConfigError(
                    "tasking.mpi_impl must be either 'open' or 'intel'. "
                    f"Got: '{mpi_impl}'"
                )
            self._task_mpi_impl = mpi_impl
            self._task_pinning_hostfile = _get_optional(
                tasking_config, "hostfile", self._task_pinning_hostfile
            )
        else:
            self._logger.log_info_rank_zero(
                "No or incorrect tasking information provided. Micro manager will not create workers and instead solve micro simulations locally."
            )

    def read_json_micro_manager(self):
        """
        Reads Micro Manager relevant information from JSON configuration file
        and saves the data to the respective instance attributes.
        """
        self._read_json(self._config_file_name)  # Read base information

        coupling_params = self._data["coupling_params"]
        simulation_params = self._data["simulation_params"]

        # Mandatory: precice_config_file_name, macro_mesh_name
        precice_config_name = _get_required(
            coupling_params, "precice_config_file_name", "coupling_params"
        )
        self._precice_config_file_name = os.path.join(
            self._folder, precice_config_name
        )
        self._logger.log_info_rank_zero(
            "preCICE configuration file name: " + self._precice_config_file_name
        )

        self._macro_mesh_name = _get_required(
            coupling_params, "macro_mesh_name", "coupling_params"
        )
        self._logger.log_info_rank_zero("Macro mesh name: " + self._macro_mesh_name)

        # Mandatory: macro_domain_bounds
        self._macro_domain_bounds = _get_required(
            simulation_params, "macro_domain_bounds", "simulation_params"
        )
        self._logger.log_info_rank_zero(
            "Macro domain bounds: " + str(self._macro_domain_bounds)
        )

        # Optional: decomposition (default: None for serial)
        self._ranks_per_axis = _get_optional(
            simulation_params, "decomposition", None, "simulation_params"
        )
        if self._ranks_per_axis is not None:
            if not isinstance(self._ranks_per_axis, list):
                raise ConfigError(
                    "simulation_params.decomposition must be a list "
                    "(e.g. [1, 1, 1] for 3D serial)."
                )
            self._logger.log_info_rank_zero(
                "Axis-wise domain decomposition: " + str(self._ranks_per_axis)
            )
        else:
            self._logger.log_info_rank_zero(
                "Domain decomposition is not specified, so the Micro Manager will expect to be run in serial."
            )

        # Optional: adaptivity (default: False)
        adaptivity_enabled = _get_optional(
            simulation_params, "adaptivity", False
        )
        adaptivity_settings = _get_optional(
            simulation_params, "adaptivity_settings", None
        )
        if adaptivity_enabled:
            self._adaptivity = True
            self._logger.log_info_rank_zero(
                "Micro Manager will adaptively run micro simulations."
            )
            if not adaptivity_settings:
                raise ConfigError(
                    "adaptivity is true but simulation_params.adaptivity_settings "
                    "is missing or empty. Please provide adaptivity_settings."
                )
        else:
            self._adaptivity = False
            if adaptivity_settings:
                raise ConfigError(
                    "adaptivity_settings is provided but adaptivity is false. "
                    "Set simulation_params.adaptivity to true to use adaptivity."
                )
            self._logger.log_info_rank_zero(
                "Micro Manager will not adaptively run micro simulations, but instead will run all micro simulations."
            )

        if self._adaptivity:
            # adaptivity_settings is guaranteed non-None when _adaptivity is True
            adapt_type = _get_required(
                adaptivity_settings, "type", "adaptivity_settings"
            )
            if adapt_type == "local":
                self._adaptivity_type = "local"
            elif adapt_type == "global":
                self._adaptivity_type = "global"
            else:
                raise ConfigError(
                    "adaptivity_settings.type must be either 'local' or 'global'. "
                    f"Got: '{adapt_type}'"
                )

            self._logger.log_info_rank_zero("Adaptivity type: " + self._adaptivity_type)

            if _get_optional(
                adaptivity_settings, "lazy_initialization", False
            ):
                self._lazy_initialization = True

            self._logger.log_info_rank_zero(
                "Micro simulations will be created only when they are required to be active for the very first time."
            )

            self._data_for_adaptivity = _get_required(
                adaptivity_settings, "data", "adaptivity_settings"
            )

            self._logger.log_info_rank_zero(
                "Data used for adaptivity: " + str(self._data_for_adaptivity)
            )

            if self._data_for_adaptivity == self._write_data_names:
                self._logger.log_info_rank_zero(
                    "Only micro simulation data is used for similarity computation in adaptivity. This would lead to the"
                    " same set of active and inactive simulations for the entire simulation time. If this is not intended,"
                    " please include macro data as well."
                )

            self._adaptivity_n = _get_optional(
                adaptivity_settings,
                "adaptivity_every_n_time_windows",
                1,
                "adaptivity_settings",
            )
            self._logger.log_info_rank_zero(
                "Adaptivity will be computed every "
                + str(self._adaptivity_n)
                + " time windows."
            )

            self._adaptivity_output_type = _get_optional(
                adaptivity_settings, "output_type", "", "adaptivity_settings"
            )
            if self._adaptivity_output_type:
                if self._adaptivity_output_type not in ["all", "local", "global"]:
                    raise ConfigError(
                        "adaptivity_settings.output_type must be one of "
                        "'all', 'local', or 'global'. "
                        f"Got: '{self._adaptivity_output_type}'"
                    )
                self._logger.log_info_rank_zero(
                    "Adaptivity output type: " + self._adaptivity_output_type
                )
            else:
                self._logger.log_info_rank_zero(
                    "No adaptivity output type provided. No metrics will be output."
                )

            self._adaptivity_output_n = _get_optional(
                adaptivity_settings, "output_n", 1, "adaptivity_settings"
            )
            self._logger.log_info_rank_zero(
                "Adaptivity metrics will be output every "
                + str(self._adaptivity_output_n)
                + " time windows."
            )

            self._adaptivity_history_param = _get_required(
                adaptivity_settings, "history_param", "adaptivity_settings"
            )
            self._logger.log_info_rank_zero(
                "Adaptivity history parameter: " + str(self._adaptivity_history_param)
            )

            self._adaptivity_coarsening_constant = _get_required(
                adaptivity_settings, "coarsening_constant", "adaptivity_settings"
            )
            self._logger.log_info_rank_zero(
                "Adaptivity coarsening constant: "
                + str(self._adaptivity_coarsening_constant)
            )

            self._adaptivity_refining_constant = _get_required(
                adaptivity_settings, "refining_constant", "adaptivity_settings"
            )
            self._logger.log_info_rank_zero(
                "Adaptivity refining constant: "
                + str(self._adaptivity_refining_constant)
            )

            # Optional: similarity_measure (default: L2rel per docs)
            self._adaptivity_similarity_measure = _get_optional(
                adaptivity_settings,
                "similarity_measure",
                "L2rel",
                "adaptivity_settings",
            )
            self._logger.log_info_rank_zero(
                "Adaptivity similarity measure: "
                + str(self._adaptivity_similarity_measure)
            )

            # Optional: every_implicit_iteration (default: False)
            adaptivity_every_implicit_iteration = _get_optional(
                adaptivity_settings, "every_implicit_iteration", False
            )
            self._adaptivity_every_implicit_iteration = adaptivity_every_implicit_iteration
            if self._adaptivity_every_implicit_iteration:
                self._logger.log_info_rank_zero(
                    "Micro Manager will compute adaptivity in every implicit iteration, if implicit coupling is done."
                )
            else:
                self._logger.log_info_rank_zero(
                    "Micro Manager will compute adaptivity once at the start of every time window."
                )

            self._write_data_names.append("Active-State")
            self._write_data_names.append("Active-Steps")

        # Optional: load_balancing (default: False)
        self._adaptivity_is_load_balancing = _get_optional(
            simulation_params, "load_balancing", False
        )
        if self._adaptivity_is_load_balancing:
            self._logger.log_info_rank_zero(
                "Micro Manager will dynamically balance micro simulations based on the adaptivity computation."
            )
            self._write_data_names.append("rank_of_sim")
            if self._adaptivity_type != "global":
                raise ConfigError(
                    "load_balancing requires global adaptivity. "
                    "Set adaptivity_settings.type to 'global' when using load_balancing."
                )
        else:
            self._logger.log_info_rank_zero(
                "Micro Manager will not dynamically balance micro simulations based on the adaptivity computation."
            )

        if self._adaptivity_is_load_balancing:
            load_balancing_settings = _get_required(
                simulation_params,
                "load_balancing_settings",
                "simulation_params",
            )
            self._load_balancing_n = _get_required(
                load_balancing_settings,
                "every_n_time_windows",
                "load_balancing_settings",
            )
            self._logger.log_info_rank_zero(
                "Load balancing will be done every "
                + str(self._load_balancing_n)
                + " time windows."
            )

            self._load_balancing_threshold = _get_optional(
                load_balancing_settings, "balancing_threshold", 0
            )
            self._logger.log_info_rank_zero(
                "Load balancing threshold: " + str(self._load_balancing_threshold)
            )

            self._balance_inactive_sims = _get_optional(
                load_balancing_settings, "balance_inactive_sims", False
            )
            if self._balance_inactive_sims:
                self._logger.log_info_rank_zero(
                    "Micro Manager will redistribute inactive simulations in the load balancing."
                )
            else:
                self._logger.log_info_rank_zero(
                    "Micro Manager will not redistribute inactive simulations in the load balancing. Only active simulations will be redistributed. Note that this may significantly increase the communication cost of the adaptivity."
                )

        # Optional: model_adaptivity (default: False)
        model_adaptivity_enabled = _get_optional(
            simulation_params, "model_adaptivity", False
        )
        model_adaptivity_settings = _get_optional(
            simulation_params, "model_adaptivity_settings", None
        )
        if model_adaptivity_enabled:
            self._m_adap = True
            self._logger.log_info_rank_zero(
                "Micro Manager will use Model Adaptivity."
            )
            if not model_adaptivity_settings:
                raise ConfigError(
                    "model_adaptivity is true but simulation_params.model_adaptivity_settings "
                    "is missing. Please provide model_adaptivity_settings."
                )
        else:
            self._m_adap = False
            if model_adaptivity_settings:
                raise ConfigError(
                    "model_adaptivity_settings is provided but model_adaptivity is false. "
                    "Set simulation_params.model_adaptivity to true."
                )
            self._logger.log_info_rank_zero(
                "Micro Manager will not adaptively switch simulation models."
            )

        if self._m_adap:
            # model_adaptivity_settings is guaranteed non-None when _m_adap is True
            self._m_adap_micro_file_names = [
                name.replace("/", ".").replace("\\", ".").replace(".py", "")
                for name in _get_required(
                    model_adaptivity_settings,
                    "micro_file_names",
                    "model_adaptivity_settings",
                )
            ]

            if len(self._m_adap_micro_file_names) < 2:
                self._logger.log_info_rank_zero(
                    "Not enough Micro Models provided for Model Adaptivity. Need min 2."
                )
                self._logger.log_info_rank_zero("Disabling Model Adaptivity.")
                self._m_adap = False

            self._m_adap_switching_function = _get_required(
                model_adaptivity_settings,
                "switching_function",
                "model_adaptivity_settings",
            )

            if "micro_stateless" in model_adaptivity_settings:
                self._m_adap_micro_stateless = model_adaptivity_settings["micro_stateless"]
            else:
                self._m_adap_micro_stateless = [False] * len(
                    self._m_adap_micro_file_names
                )

            for i in range(len(self._m_adap_micro_file_names)):
                if self._m_adap_micro_stateless[i]:
                    self._logger.log_info_rank_zero(
                        f"Only creating one full instance of Micro Model {i}."
                    )
                else:
                    self._logger.log_info_rank_zero(
                        f"Creating full instance of Micro Model {i} per mesh vertex."
                    )

        # Optional: interpolate_crash (default: False)
        if _get_optional(
            simulation_params, "interpolate_crash", False
        ):
            self._interpolate_crash = True
            self._logger.log_info_rank_zero(
                "Micro Manager will interpolate output of crashed micro simulations from its neighbors."
            )

        # Optional: diagnostics section (default: empty dict)
        diagnostics = _get_optional(self._data, "diagnostics", {})

        diagnostics_data_names = _get_optional(
            diagnostics, "data_from_micro_sims", None
        )
        if diagnostics_data_names is not None:
            if not isinstance(diagnostics_data_names, list):
                raise ConfigError(
                    "diagnostics.data_from_micro_sims must be a list."
                )
            self._logger.log_info_rank_zero(
                "Diagnostics data: " + str(diagnostics_data_names)
            )
        else:
            self._logger.log_info_rank_zero(
                "No diagnostics data is defined. Micro Manager will not output any diagnostics data."
            )

        # Optional: micro_output_n (default: 1)
        self._micro_output_n = _get_optional(
            diagnostics, "micro_output_n", 1
        )
        self._logger.log_info_rank_zero(
            "Micro output will be called every "
            + str(self._micro_output_n)
            + " time windows."
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

        coupling_params = self._data["coupling_params"]

        # Mandatory: parameter_file_name (for snapshot mode)
        parameter_file_name = _get_required(
            coupling_params, "parameter_file_name", "coupling_params"
        )
        self._parameter_file_name = os.path.join(
            self._folder, parameter_file_name
        )
        self._logger.log_info_rank_zero(
            "Parameter file name: " + self._parameter_file_name
        )

        # Optional: snapshot_params section (default: empty dict)
        snapshot_params = _get_optional(
            self._data, "snapshot_params", {}
        )

        # Optional: output_file_name (default: "snapshot_data")
        self._output_file_name = _get_optional(
            snapshot_params, "output_file_name", "snapshot_data"
        )
        self._logger.log_info_rank_zero(
            "Output file name: " + self._output_file_name
        )

        # Optional: post_processing_file_name (default: None)
        post_proc_raw = _get_optional(
            snapshot_params, "post_processing_file_name", None
        )
        if post_proc_raw is not None:
            self._postprocessing_file_name = (
                post_proc_raw.replace("/", ".")
                .replace("\\", ".")
                .replace(".py", "")
            )
            self._logger.log_info_rank_zero(
                "Post-processing file name: " + self._postprocessing_file_name
            )
        else:
            self._postprocessing_file_name = None
            self._logger.log_info_rank_zero(
                "No post-processing file name provided. Snapshot computation will not perform any post-processing."
            )

        # Optional: diagnostics section (default: empty dict)
        diagnostics = _get_optional(self._data, "diagnostics", {})

        diagnostics_data_names = _get_optional(
            diagnostics, "data_from_micro_sims", None
        )
        if diagnostics_data_names is not None:
            if not isinstance(diagnostics_data_names, list):
                raise ConfigError(
                    "diagnostics.data_from_micro_sims must be a list."
                )
            self._logger.log_info_rank_zero(
                "Diagnostics data: " + str(diagnostics_data_names)
            )
        else:
            self._logger.log_info_rank_zero(
                "No diagnostics data is defined. Snapshot computation will not output any diagnostics data."
            )

        # Optional: initialize_once (default: False)
        self._initialize_once = _get_optional(
            snapshot_params, "initialize_once", False
        )
        if self._initialize_once:
            self._logger.log_info_rank_zero(
                "Micro Manager will initialize only one micro simulations object for snapshot computation."
            )
        else:
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

    def turn_on_micro_stateless(self):
        """
        Boolean stating whether micro model is stateless or not.

        Returns
        -------
        stateless : bool
            True if micro model is stateless, False otherwise.
        """
        return self._micro_stateless

    def get_micro_output_n(self):
        """
        Get the micro output frequency

        Returns
        -------
        micro_output_n : int
            Output frequency of micro simulations, so output every N timesteps
        """
        return self._micro_output_n

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

    def get_output_file_name(self):
        """
        Get the name of the output file.

        Returns
        -------
        output_file_name : string
            Name of the hdf5 file containing the snapshot data.
        """
        return self._output_file_name

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

    def turn_on_model_adaptivity(self):
        """
        Boolean stating whether adaptivity is ot or not.

        Returns
        -------
        adaptivity : bool
            True is model adaptivity settings are done, False otherwise.

        """
        return self._m_adap

    def get_model_adaptivity_file_names(self):
        """
        Get the paths to the Python scripts of the model-adaptive-micro-simulations.

        Returns
        -------
        micro_file_names : string
            String carrying the path to the Python script of the micro-simulation.
        """
        return self._m_adap_micro_file_names

    def get_model_adaptivity_micro_stateless(self):
        """
        List of boolean stating whether the respective micro model is stateless or not.

        Returns
        -------
        stateless : list
            True if micro model is stateless, False otherwise.
        """
        return self._m_adap_micro_stateless

    def get_model_adaptivity_switching_function(self):
        """
        Get path to switching function file

        Returns
        -------
        switching_function : str
            String containing the path to the switching function file
        """
        return self._m_adap_switching_function

    def get_tasking_num_workers(self):
        """
        Get number of workers

        Returns
        -------
        num_workers : int
            Number of workers
        """
        return self._task_num_workers

    def get_tasking_backend(self):
        """
        Get backend type

        Returns
        -------
        backend : str
            either socket or mpi
        """
        return self._task_backend

    def get_tasking_use_slurm(self):
        """
        Get flag whether slurm is used

        Returns
        -------
        use_slurm : bool
            use slurm or not
        """
        return self._task_is_slurm

    def get_tasking_hostfile(self):
        """
        Get hostfile path for workers

        Returns
        -------
        hostfile : str
            Hostfile path for workers
        """
        return self._task_pinning_hostfile

    def get_mpi_impl(self):
        """
        Get mpi implementation type

        Returns
        -------
        mpi_impl : str
            mpi implementation type
        """
        return self._task_mpi_impl
