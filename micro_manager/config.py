"""
Class Config provides functionality to read a JSON file and pass the values to the Micro Manager.
"""

import json
import logging
import os
import importlib.metadata
import string
from collections import defaultdict
from typing import Optional, Type, List, Dict, Any, Callable, Hashable
import inspect
from .tools.logging_wrapper import Logger


class ConfigDSL:
    """
    Provides a standardized context to read data from JSON.
    Data retrieval can be set up to yield optionals, with default values or raise Errors.
    """

    def __init__(self, data: Dict, log: Logger):
        """
        Constructs a context for the JSON data.

        Parameters
        ----------
        data : Dict
            JSON data
        log : Logger
            Logging object
        """
        self._data: Dict = data
        self._log: Logger = log
        self._access_queue: List[str] = []

    def __getitem__(self, path: str):
        """
        Adds an access request for the given path.
        Used for nested JSON objects as well as element access.

        Parameters
        ----------
        path : str
            Path

        Returns
        -------
        context : ConfigDSL
            same object, for chained requests
        """
        self._access_queue.append(path)
        return self

    def exists(self):
        """
        Checks if the requested path exists.

        Returns
        -------
        exists : bool
            True if exists, False otherwise
        """
        current_element = self._data
        try:
            for key in self._access_queue:
                current_element = current_element[key]
        except BaseException as e:
            return False
        return True

    def get_or_none(
        self,
        fmt_success: Optional[str] = None,
        fmt_error: Optional[str] = None,
        dtype: Optional[Type] = None,
        options: Optional[List] = None,
        **kwargs,
    ) -> Optional[Any]:
        """
        Resolves the value specified by the set path. Returns the result if available, otherwise None.
        Format strings can be provided for success or failure. {data} and {ex} are reserved keywords.
        {data} is the retrieved data, {ex} is the raised exception.
        In addition, a target data type and a list of possible options can be specified.

        Parameters
        ----------
        fmt_success : Optional[str]
            Format string containing message on success.
        fmt_error : Optional[str]
            Format string containing message on error.
        dtype: Optional[Type]
            Target data type
        options: Optional[List]
            Options of which the retrieved value must be an element.
        kwargs: ...
            Keyword arguments that should be passed to the format strings.

        Returns
        -------
        result: Optional[Any]
            None if path not available or other failure, else value of JSON.
        """
        return self.get_with_default(
            None, fmt_success, fmt_error, dtype, options, **kwargs
        )

    def get_with_default(
        self,
        default: Any,
        fmt_success: Optional[str] = None,
        fmt_error: Optional[str] = None,
        dtype: Optional[Type] = None,
        options: Optional[List] = None,
        **kwargs,
    ) -> Any:
        """
        Resolves the value specified by the set path. Returns the result if available, otherwise the specified default.
        Format strings can be provided for success or failure. {data} and {ex} are reserved keywords.
        {data} is the retrieved data, {ex} is the raised exception.
        In addition, a target data type and a list of possible options can be specified.

        Parameters
        ----------
        default : Any
            Default value.
        fmt_success : Optional[str]
            Format string containing message on success.
        fmt_error : Optional[str]
            Format string containing message on error.
        dtype: Optional[Type]
            Target data type
        options: Optional[List]
            Options of which the retrieved value must be an element.
        kwargs: ...
            Keyword arguments that should be passed to the format strings.

        Returns
        -------
        result: Any
            Default value if path not available or other failure, else value of JSON.
        """
        current_element = self._data
        try:
            for key in self._access_queue:
                current_element = current_element[key]

            if dtype is not None and type(current_element) != dtype:
                raise RuntimeError("Wrong data type")
            if options is not None and current_element not in options:
                raise RuntimeError("Wrong option")
        except BaseException as e:
            self.handle_fmt_error(fmt_error or "", e, kwargs)
            return default

        self.handle_fmt_success(fmt_success or "", current_element, kwargs)
        return current_element

    def get_or_raise(
        self,
        fmt_success: Optional[str] = None,
        fmt_error: Optional[str] = None,
        dtype: Optional[Type] = None,
        options: Optional[List] = None,
        **kwargs,
    ):
        """
        Resolves the value specified by the set path. Returns the result if available, otherwise throws.
        Format strings can be provided for success or failure. {data} and {ex} are reserved keywords.
        {data} is the retrieved data, {ex} is the raised exception.
        In addition, a target data type and a list of possible options can be specified.

        Parameters
        ----------
        fmt_success : Optional[str]
            Format string containing message on success.
        fmt_error : Optional[str]
            Format string containing message on error.
        dtype: Optional[Type]
            Target data type
        options: Optional[List]
            Options of which the retrieved value must be an element.
        kwargs: ...
            Keyword arguments that should be passed to the format strings.

        Returns
        -------
        result: Any
            Throws if path not available or other failure, else value of JSON.
        """
        current_element = self._data
        try:
            for key in self._access_queue:
                current_element = current_element[key]

            if dtype is not None and type(current_element) != dtype:
                raise RuntimeError("Wrong data type")
            if options is not None and current_element not in options:
                raise RuntimeError("Wrong option")
        except BaseException as e:
            self.handle_fmt_error(fmt_error or "", e, kwargs)
            raise e

        self.handle_fmt_success(fmt_success or "", current_element, kwargs)
        return current_element

    def handle_fmt(
        self,
        fmt: Optional[str],
        kwargs: Optional[Dict[str, Any]],
    ) -> None:
        """
        Processes the message and prints to the logger.

        Parameters
        ----------
        fmt : Optional[str]
            Format string.
        kwargs: Optional[Dict[str, Any]]
            Additional keyword arguments that should be passed to the format string.
        """
        if fmt is None:
            return
        if kwargs is None:
            kwargs = {}

        fmt_args = {}
        for key, value in kwargs.items():
            has_key = ConfigDSL.fmt_check_key(fmt, key)
            if has_key:
                fmt_args[key] = value

        msg = fmt.format(**fmt_args)
        self._log.log_info_rank_zero(msg)

    def handle_fmt_error(
        self, fmt_error: str, ex: BaseException, kwargs: Dict[str, Any]
    ) -> None:
        """
        Processes the error message and prints to the logger.

        Parameters
        ----------
        fmt_error : str
            Format string containing message on success.
        ex : BaseException
            Raised Exception.
        kwargs: Dict[str, Any]
            Additional keyword arguments that should be passed to the format string.
        """
        kwargs["ex"] = ex
        self.handle_fmt(fmt_error, kwargs)

    def handle_fmt_success(
        self,
        fmt_success: str,
        data: Any,
        kwargs: Dict[str, Any],
    ) -> None:
        """
        Processes the success message and prints to the logger.

        Parameters
        ----------
        fmt_success : str
            Format string containing message on success.
        data : Any
            Retrieved data.
        kwargs: Dict[str, Any]
            Additional keyword arguments that should be passed to the format string.
        """
        kwargs["data"] = data
        self.handle_fmt(fmt_success, kwargs)

    _formatter = string.Formatter()

    @staticmethod
    def fmt_check_key(fmt: str, key: str) -> bool:
        """
        Checks if the provided format string contains the specified key.

        Parameters
        ----------
        fmt : str
            Format string to check.
        key : str
            Key of question.

        Returns
        -------
        contains_key : bool
            True if fmt contains the key, else False.

        """
        for _, field_name, _, _ in ConfigDSL._formatter.parse(fmt):
            if field_name == key:
                return True
        return False


class ConfigEntryProxy:
    """
    Enables the usage of the following syntax:
        # DEFINITION
        @config_entry
        def member_name(self): ...

        # WRITE ACCESS
        self.member_name.set = value

        # KEY ACCESS
        self.member_name.key

    Hereby, config entries do not need to define individual backing fields and can instead use
    the dict self._fields within the config object. Data is stored using the respective keys.
    The additional write access should simplify writing to the dict.

    """

    def __init__(self, func: Callable):
        self._func: Callable = func
        self._instance: Optional[Config] = None

    def attach_instance(self, instance: "Config"):
        self._instance = instance

    @property
    def key(self) -> str:
        """
        Gets the key of the config entry.

        Returns
        -------
        key : str
            Key of the config entry.
        """
        return self._func.__name__

    def __call__(self) -> Any:
        """
        Gets the stored value for the config entry.

        Returns
        -------
        value : Any
            Value of the config entry.
        """
        return self._instance._fields[self.key]

    def __setattr__(self, key: str, value: Any) -> None:
        """
        Stores the provided value for the config entry, when attempting to write to the self.set member.

        Parameters
        ----------
        key : str
            Name of written to member. (provided by Python RT)
        value : Any
            Value of the config entry to be stored.
        """
        if key in ["_func", "_instance"]:
            super().__setattr__(key, value)
            return
        if key == "set":
            self._instance._fields[self.key] = value
        return


def config_entry(func: Callable) -> Callable:
    """
    Decorator for Config entries. See ConfigEntryProxy for details.

    Parameters
    ----------
    func : Callable
        config entry method

    Returns
    -------
    func : Callable
        config entry method, enriched by ConfigEntryProxy.
    """
    return ConfigEntryProxy(func)


class ConfigLogContext:
    """
    Provides a context in which the loggers of the config can be optionally switched.
    If one log uses a high log level, this can disable log-outputs for the subsequent code block.
    """

    def __init__(self, config: "Config", should_swap: bool):
        """
        Constructs a ConfigLogContext object.

        Parameters
        ----------
        config : Config
            Config object.
        should_swap : bool
            Should the loggers be swapped.
        """
        self._config: Config = config
        self._should_swap: bool = should_swap

    def swap(self):
        """
        Swaps the loggers used by the config object.
        """
        tmp = self._config._logger
        self._config._logger = self._config._logger_null
        self._config._logger_null = tmp

    def __enter__(self) -> "ConfigLogContext":
        """
        Called upon entering a with ... statement.

        Returns
        -------
        ctx : ConfigLogContext
            Logging context.
        """
        if self._should_swap:
            self.swap()
        return self

    def __exit__(self, *args) -> None:
        """
        Called upon exiting a with ... statement.
        """
        if self._should_swap:
            self.swap()


class Config:
    """
    Handles the reading of parameters in the JSON configuration file provided by the user. This class is based on
    the config class in https://github.com/precice/fenics-adapter/tree/develop/fenicsadapter
    """

    def __init__(self, config_file_name: str):
        """
        Constructor of the Config class.

        Parameters
        ----------
        config_file_name : string
            Path to the JSON configuration file
        """
        self._config_file_name: str = config_file_name
        self._base_dir: str = os.path.dirname(
            os.path.join(os.getcwd(), config_file_name)
        )
        self._logger_null: Logger = Logger(Config.__name__, level=logging.CRITICAL)
        self._logger: Logger = self._logger_null
        self._data: Optional[Dict[str, Any]] = None
        self._fields: Dict[str, Any] = defaultdict(lambda: None)

        # attach external backing field to all config entries
        # inspection triggers assertions: need to temporarily set to value other than None
        self._data = {}
        config_entries = inspect.getmembers(
            self, lambda f: hasattr(f, "attach_instance")
        )
        for _, e in config_entries:
            e.attach_instance(self)
        self._data = None

    def set_logger(self, logger: Logger):
        """
        Set the logger for the Config class.

        Parameters
        ----------
        logger : object of logging
            Logger defined from the standard package logging
        """
        self._logger = logger

    def show_log_if(self, cond: bool) -> ConfigLogContext:
        """
        Redirects log requests in the resulting context if cond is True.

        Usage:
        with self.show_log_if(cond):
            ... load fields ...

        Hereby default arguments can be loaded for inactive modules without emitting log output.

        Parameters
        ----------
        cond : bool
            Should redirect?

        Returns
        -------
        ctx : ConfigLogContext
            Logging context
        """
        return ConfigLogContext(self, not cond)

    @property
    def json(self) -> ConfigDSL:
        """
        Returns and Object to load JSON values dynamically.
        """
        assert self._data is not None
        return ConfigDSL(self._data, self._logger)

    def _read_json_base(self, config_file_name: str):
        """
        Reads JSON configuration file.

        Parameters
        ----------
        config_file_name : string
            Path to the JSON configuration file
        """
        self._logger.log_info_rank_zero(
            f"Micro Manager version: {importlib.metadata.version('micro-manager-precice')}"
        )
        path = os.path.join(self._base_dir, os.path.basename(config_file_name))
        with open(path, "r") as read_file:
            self._data = json.load(read_file)

        self._logger.log_info_rank_zero(f"Reading JSON configuration file: {path}")

        # ======================================================
        #                  Micro Manager Base
        # ======================================================

        # convert paths to python-importable paths
        file_names = self.json["micro_file_names"].get_or_raise(
            "Micro simulation file name: {data}",
            "'micro_file_name' must be specified!",
            list,
        )
        self.micro_file_names.set = [
            name.replace("/", ".").replace("\\", ".").replace(".py", "")
            for name in file_names
        ]
        if len(self.micro_file_names()) < 1:
            self._logger.log_info_rank_zero(
                "Must provide at least one micro simulation file."
            )
            raise RuntimeError("Missing Micro Simulation File")

        self.micro_stateless_flags.set = self.json[
            "micro_stateless_flags"
        ].get_with_default(
            [False] * len(self.micro_file_names()),
            "Only creating one full instance of Micro Model.",
            "Creating full instance of Micro Model per mesh vertex.",
            list,
        )

        for i in range(len(self.micro_file_names())):
            if self.micro_stateless_flags()[i]:
                self._logger.log_info_rank_zero(
                    f"Creating only one instance of Micro Model {i}."
                )
            else:
                self._logger.log_info_rank_zero(
                    f"Creating all instances of Micro Model {i} per mesh vertex."
                )

        self.output_dir.set = self.json["output_directory"].get_or_none(
            "Logging and metrics output directory: {data}",
            "No output directory provided. Output (including logging) will be saved in the current working directory.",
        )

        self.memory_usage_output_type.set = self.json[
            "memory_usage_output_type"
        ].get_with_default(
            "",
            "Memory usage output type: {data}",
            "Micro Manager will not output memory usage.",
            options=["all", "local", "global"],
        )

        self.memory_usage_output_n.set = self.json[
            "memory_usage_output_n"
        ].get_with_default(
            1,
            "Memory usage will be output every {data} time windows.",
            "No output interval for memory usage output provided. Memory usage will be output every time window.",
        )

        self.write_data_names.set = self.json["coupling_params"][
            "write_data_names"
        ].get_or_none(
            "Micro Manager is writing the following data: {data}",
            "No write data names provided. Micro manager will only read data from preCICE.",
            list,
        )

        self.read_data_names.set = self.json["coupling_params"][
            "read_data_names"
        ].get_or_none(
            "Micro Manager is reading the following data: {data}",
            "No read data names provided. Micro manager will only write data to preCICE.",
            list,
        )

        self.micro_dt.set = self.json["simulation_params"]["micro_dt"].get_or_raise()

        # ======================================================
        #                        Tasking
        # ======================================================
        with self.show_log_if(self.json["tasking"].exists()):
            self.tasking_backend.set = self.json["tasking"]["backend"].get_with_default(
                "socket",
                "Tasking backend: {data}",
                "No tasking backed defined. Falling back to sockets.",
                options=["mpi", "socket"],
            )
            self.enable_tasking_slurm.set = self.json["tasking"][
                "is_slurm"
            ].get_with_default(
                False,
                "Tasking using slurm: {data}",
                "No tasking slurm flag defined. Assuming non-slurm system.",
            )
            self.tasking_num_workers.set = self.json["tasking"][
                "num_workers"
            ].get_with_default(
                1,
                "Tasking will use {data} workers",
                "No tasking worker count defined. Using 1 worker per rank.",
            )
            if self.tasking_num_workers() < 1:
                raise RuntimeError("Invalid number of workers. Must be >= 1.")

            self.mpi_impl.set = self.json["tasking"]["mpi_impl"].get_with_default(
                "open",
                "Tasking using mpi implementation: {data}",
                "No tasking mpi implementation defined. Assuming open mpi.",
                options=["open", "mpi"],
            )
            self.tasking_hostfile.set = self.json["tasking"][
                "hostfile"
            ].get_with_default(
                "./hosts.micro",
                "Tasking will use nodes from hostlist file: {data}",
                "No hostfile for tasking defined. Using hosts.micro as default.",
            )

    def read_json_micro_manager(self):
        """
        Reads Micro Manager relevant information from JSON configuration file
        and saves the data to the respective instance attributes.
        """
        self._read_json_base(self._config_file_name)

        self.precice_config_file_name.set = os.path.join(
            self._base_dir, self._data["coupling_params"]["precice_config_file_name"]
        )
        self._logger.log_info_rank_zero(
            f"preCICE configuration file name: {self.precice_config_file_name()}"
        )

        # ======================================================
        #                 Mesh and Decomposition
        # ======================================================

        self.macro_mesh_name.set = self.json["coupling_params"][
            "macro_mesh_name"
        ].get_or_raise("Macro mesh name: {data}")

        self.macro_domain_bounds.set = self.json["simulation_params"][
            "macro_domain_bounds"
        ].get_or_raise("Macro domain bounds: {data}")

        self.ranks_per_axis.set = self.json["simulation_params"][
            "decomposition"
        ].get_with_default(
            [1, 1, 1],
            "Axis-wise domain decomposition: {data}",
            "Domain decomposition is not specified, so the Micro Manager will expect to be run in serial.",
            list,
        )

        self.decomposition_type.set = self.json["simulation_params"][
            "decomposition_type"
        ].get_with_default(
            "uniform",
            "Domain decomposition type: {data}",
            "Domain decomposition is not specified, so the Micro Manager will expect to be run in serial.",
            str,
            ["uniform", "nonuniform"],
        )
        self.minimum_access_region_size.set = []
        if self.decomposition_type() == "nonuniform":
            self.minimum_access_region_size.set = self.json["simulation_params"][
                "minimum_access_region_size"
            ].get_with_default(
                [],
                None,
                "Minimum access region size is not specified. Calculating it as 1 / (2^ranks_per_axis - 1) of the macro domain size in each axis.",
            )

        # ======================================================
        #                      Adaptivity
        # ======================================================

        self.enable_adaptivity.set = self.json["simulation_params"][
            "adaptivity"
        ].get_with_default(
            False,
            "Micro Manager will adaptively run micro simulations.",
            None,
            bool,
        )
        adaptivity_settings_avail = self.json["simulation_params"][
            "adaptivity_settings"
        ].exists()
        if self.enable_adaptivity() and not adaptivity_settings_avail:
            self.enable_adaptivity.set = False
            self._logger.log_info_rank_zero(
                "Adaptivity is turned on but no adaptivity settings are provided."
            )
            self._logger.log_info_rank_zero(
                "Micro Manager will not adaptively run micro simulations, but instead will run all micro simulations."
            )
        if not self.enable_adaptivity() and adaptivity_settings_avail:
            self._logger.log_info_rank_zero(
                "Adaptivity settings are provided but adaptivity is turned off."
            )

        if self.enable_adaptivity():
            self.adaptivity_type.set = self.json["simulation_params"][
                "adaptivity_settings"
            ]["type"].get_with_default(
                "local",
                "Adaptivity type: {data}",
                "Adaptivity type can be either local or global.",
                options=["local", "global"],
            )
            self.adaptivity_mapping_configs.set = self.json["simulation_params"][
                "adaptivity_settings"
            ]["mappings"].get_with_default(
                [],
                None,
                "Adaptivity will not interpolate outputs, only use representatives.",
            )
            self.enable_adaptivity_lazy_init.set = self.json["simulation_params"][
                "adaptivity_settings"
            ]["lazy_initialization"].get_with_default(False)
            if self.enable_adaptivity_lazy_init():
                self._logger.log_info_rank_zero(
                    "Micro simulations will be created only when they are required to be active for the very first time."
                )
            self.data_for_adaptivity.set = self.json["simulation_params"][
                "adaptivity_settings"
            ]["data"].get_or_raise(
                "Data used for adaptivity: {data}", "Adaptivity Data must be provided."
            )
            if self.data_for_adaptivity() == self.write_data_names():
                self._logger.log_info_rank_zero(
                    "Only micro simulation data is used for similarity computation in adaptivity. This would lead to the"
                    " same set of active and inactive simulations for the entire simulation time. If this is not intended,"
                    " please include macro data as well."
                )
            self.adaptivity_n.set = self.json["simulation_params"][
                "adaptivity_settings"
            ]["adaptivity_every_n_time_windows"].get_with_default(
                1,
                "Adaptivity will be computed every {data} time windows.",
                "No interval for adaptivity computation provided. Adaptivity will be computed in every time window.",
            )
            self.adaptivity_output_type.set = self.json["simulation_params"][
                "adaptivity_settings"
            ]["output_type"].get_with_default(
                "",
                "Adaptivity output type: {data}",
                "Adaptivity output type can be either 'all', 'local' or 'global'. No metrics will be output.",
                options=["all", "local", "global"],
            )
            self.adaptivity_output_n.set = self.json["simulation_params"][
                "adaptivity_settings"
            ]["output_n"].get_with_default(
                1,
                "Adaptivity will be computed every {data} time windows.",
                "No output interval for adaptivity provided. Adaptivity metrics will be output every time window.",
            )
            self.adaptivity_history_param.set = self.json["simulation_params"][
                "adaptivity_settings"
            ]["history_param"].get_or_raise("Adaptivity history parameter: {data}")
            self.adaptivity_coarsening_constant.set = self.json["simulation_params"][
                "adaptivity_settings"
            ]["coarsening_constant"].get_or_raise(
                "Adaptivity coarsening constant: {data}"
            )
            self.adaptivity_refining_constant.set = self.json["simulation_params"][
                "adaptivity_settings"
            ]["refining_constant"].get_or_raise("Adaptivity refining constant: {data}")
            self.adaptivity_similarity_measure.set = self.json["simulation_params"][
                "adaptivity_settings"
            ]["similarity_measure"].get_with_default(
                "L2rel",
                "Adaptivity similarity measure: {data}",
                "No similarity measure provided, using L1 norm as default.",
            )
            self.enable_adaptivity_each_implicit_iteration.set = self.json[
                "simulation_params"
            ]["adaptivity_settings"]["every_implicit_iteration"].get_with_default(
                False,
                "Micro Manager will compute adaptivity once at the start of every time window.",
            )
            if self.enable_adaptivity_each_implicit_iteration():
                self._logger.log_info_rank_zero(
                    "Micro Manager will compute adaptivity in every implicit iteration, if implicit coupling is done."
                )
            else:
                self._logger.log_info_rank_zero(
                    "Micro Manager will compute adaptivity once at the start of every time window."
                )

            self.write_data_names().append("Active-State")
            self.write_data_names().append("Active-Steps")

        # ======================================================
        #                   Load Balancing
        # ======================================================

        self.enable_load_balancing.set = self.json["simulation_params"][
            "load_balancing"
        ].get_with_default(False)
        if self.enable_load_balancing():
            self._logger.log_info_rank_zero(
                "Micro Manager will dynamically balance micro simulations based on compute times."
            )

            if self.enable_adaptivity() and not self.adaptivity_type() == "global":
                self.enable_load_balancing.set = False
                self._logger.log_info_rank_zero(
                    "Attempting to load balancing with adaptivity other than global adaptivity. "
                    "Disabling load balancing. To use load balancing either disable adaptivity or run with global adaptiviy."
                )
            else:
                self.write_data_names().append("rank_of_sim")

        with self.show_log_if(self.enable_load_balancing()):
            self.load_balancing_n.set = self.json["simulation_params"][
                "load_balancing_settings"
            ]["every_n_time_windows"].get_with_default(
                1,
                "Load balancing will be computed every {data} time windows.",
                "Load balancing will be computed in every time window.",
            )
            self.load_balancing_type.set = self.json["simulation_params"][
                "load_balancing_settings"
            ]["type"].get_with_default(
                "time",
                "Load balancing type: {data}",
                "Load balancing will use time based balancing.",
                options=["time", "active"],
            )
            if self.load_balancing_type() == "active":
                self.enable_load_balancing_inactive.set = self.json[
                    "simulation_params"
                ]["load_balancing_settings"][
                    "balance_inactive_simulations"
                ].get_with_default(
                    False,
                    "Load balancing enable inactive balancing: {data}",
                    "Load balancing will not balance inactive micro simulations.",
                    dtype=bool,
                )

                self.load_balancing_threshold.set = self.json["simulation_params"][
                    "load_balancing_settings"
                ]["threshold"].get_with_default(
                    0,
                    "Load balancing threshold: {data}",
                    "Load balancing will use 0 threshold.",
                )
            if self.load_balancing_type() == "time":
                self.load_balancing_partitioning.set = self.json["simulation_params"][
                    "load_balancing_settings"
                ]["partitioning"].get_with_default(
                    "lpt",
                    "Load balancing partitioning: {data}",
                    "No partitioning provided, using LPT as default.",
                    options=["lpt"],
                )

        # ======================================================
        #                   Model Adaptivity
        # ======================================================

        self.enable_model_adaptivity.set = self.json["simulation_params"][
            "model_adaptivity"
        ].get_with_default(False)
        if (
            self.enable_model_adaptivity()
            and not self.json["simulation_params"]["model_adaptivity_settings"].exists()
        ):
            self.enable_model_adaptivity.set = False
            self._logger.log_info_rank_zero(
                "Model Adaptivity is turned on but no model adaptivity settings are provided."
            )

        if self.enable_model_adaptivity():
            if len(self.micro_file_names()) < 2:
                self._logger.log_info_rank_zero(
                    "Not enough Micro Models provided for Model Adaptivity. Need min 2."
                )
                self._logger.log_info_rank_zero("Disabling Model Adaptivity.")
                self.enable_model_adaptivity.set = False
            else:
                self.write_data_names().append("model_resolution")

            self.model_adaptivity_switching_function.set = self.json[
                "simulation_params"
            ]["model_adaptivity_settings"]["switching_function"].get_or_raise()

        # ======================================================
        #              Interpolation and Diagnostics
        # ======================================================
        self.interpolation_configs.set = self.json["simulation_params"][
            "interpolation_configs"
        ].get_with_default(
            [],
            None,
            "Failed to load interpolation configs.",
        )

        self.enable_crashed_sim_interpolation.set = self.json["simulation_params"][
            "interpolate_crash"
        ].get_with_default(False)
        if self.enable_crashed_sim_interpolation():
            self._logger.log_info_rank_zero(
                "Micro Manager will interpolate output of crashed micro simulations from its neighbors."
            )
        if (
            self.enable_crashed_sim_interpolation()
            and not self.json["simulation_params"]["interpolate_crash_params"].exists()
        ):
            self.enable_crashed_sim_interpolation.set = False
            self._logger.log_info_rank_zero(
                "Crash Interpolation is turned on but no settings are provided."
            )
        with self.show_log_if(self.enable_crashed_sim_interpolation()):
            self.crashed_sim_interpolation_id.set = self.json["simulation_params"][
                "interpolate_crash_params"
            ]["interp_id"].get_with_default(
                None,
                "Crash Interpolation interpolates with config {data}.",
                "No interpolation config provided for Crash Interpolation.",
            )
            self.crashed_sim_interpolation_threshold.set = self.json[
                "simulation_params"
            ]["interpolate_crash_params"]["threshold"].get_with_default(
                0.2,
                "Crash Interpolation threshold: {data}.",
            )

        # TODO what? this is not being saved nor used
        diagnostics_data_names = self.json["diagnostics"][
            "data_from_micro_sims"
        ].get_or_none(
            None,
            "No diagnostics data is defined. Micro Manager will not output any diagnostics data.",
            list,
        )

        self.micro_output_n.set = self.json["diagnostics"][
            "micro_output_n"
        ].get_with_default(
            1,
            "Micro Manager will compute micro output every {data} time windows.",
            "Output interval of micro simulations not specified, if output is available then it will be called "
            "in every time window.",
        )

    def read_json_snapshot(self):
        """
        Reads Snapshot relevant information from JSON configuration file
        """
        self._read_json_base(self._config_file_name)  # Read base information
        self._logger.log_info_rank_zero(
            f"Reading JSON configuration file: {self._config_file_name}"
        )
        self._logger.log_info_rank_zero("Micro Manager is running in snapshot mode.")
        self.parameter_file_name.set = os.path.join(
            self._base_dir,
            self.json["coupling_params"]["parameter_file_name"].get_or_raise(),
        )
        self._logger.log_info_rank_zero(
            f"Parameter file name: {self.parameter_file_name()}"
        )

        self.output_file_name.set = self.json["snapshot_params"][
            "output_file_name"
        ].get_with_default(
            "snapshot_data",
            "Output file name: {data}",
            "No snapshot output file name provided. Defaulting to 'snapshot_data'.",
        )

        post_proc_file = self.json["snapshot_params"][
            "post_processing_file_name"
        ].get_or_none(
            "Post-processing file name {data}",
            "No post-processing file name provided. Snapshot computation will not perform any post-processing.",
        )
        if post_proc_file is not None:
            post_proc_file = (
                post_proc_file.replace("/", ".").replace("\\", ".").replace(".py", "")
            )
        self.postprocessing_file_name.set = post_proc_file

        # TODO what? this is not being saved nor used
        diagnostics_data_names = self.json["diagnostics"][
            "data_from_micro_sims"
        ].get_or_none(
            "Diagnostics data: {data}",
            "No diagnostics data is defined. Micro Manager will not output any diagnostics data.",
            list,
        )

        self.enable_single_sim_object.set = self.json["snapshot_params"][
            "initialize_once"
        ].get_with_default(
            False,
            "Micro Manager will initialize only one micro simulations object for snapshot computation.",
            "For each snapshot a new micro simulation object will be created.",
        )

    @config_entry
    def precice_config_file_name(self) -> str:
        """
        Get the name of the preCICE XML configuration file.

        Returns
        -------
        config_file_name : string
            Name of the preCICE XML configuration file.
        """
        pass

    @config_entry
    def macro_mesh_name(self) -> str:
        """
        Get the name of the macro mesh. This name is expected to be the same as the one defined in the preCICE
        configuration file.

        Returns
        -------
        macro_mesh_name : string
            Name of the macro mesh as stated in the JSON configuration file.

        """
        pass

    @config_entry
    def read_data_names(self) -> List[str]:
        """
        Get the user defined dictionary carrying information of the data to be read from preCICE.

        Returns
        -------
        read_data_names: List[str]
            A dictionary containing the names of the data to be read from preCICE as keys and information on whether
            the data are scalar or vector as values.
        """
        pass

    @config_entry
    def write_data_names(self) -> List[str]:
        """
        Get the user defined dictionary carrying information of the data to be written to preCICE.

        Returns
        -------
        write_data_names: List[str]
            A dictionary containing the names of the data to be written to preCICE as keys and information on whether
            the data are scalar or vector as values.
        """
        pass

    @config_entry
    def macro_domain_bounds(self) -> List[float]:
        """
        Get the upper and lower bounds of the macro domain.

        Returns
        -------
        macro_domain_bounds : List[List[float]]
            List containing upper and lower bounds of the macro domain.
            Format in 2D is [x_min, x_max, y_min, y_max]
            Format in 2D is [x_min, x_max, y_min, y_max, z_min, z_max]
        """
        pass

    @config_entry
    def ranks_per_axis(self) -> List[int]:
        """
        Get the ranks per axis for a parallel simulation

        Returns
        -------
        ranks_per_axis : List[int]
            List containing ranks in the x, y and z axis respectively.
        """
        pass

    @config_entry
    def decomposition_type(self) -> str:
        """
        Get the type of domain decomposition.

        Returns
        -------
        decomposition_type : str
            Type of domain decomposition, can be either "uniform" or "non-uniform".
        """
        pass

    @config_entry
    def minimum_access_region_size(self) -> List:
        """
        Get the minimum access region size for non-uniform domain decomposition.

        Returns
        -------
        minimum_access_region_size : List
            List containing the minimum access region size in each axis for non-uniform domain decomposition.
        """
        pass

    @config_entry
    def micro_file_names(self) -> List[str]:
        """
        Get the paths to the Python scripts of the micro-simulations.

        Returns
        -------
        micro_file_names : List[str]
            List of strings carrying the paths to the Python scripts of the micro-simulations.
        """
        pass

    @config_entry
    def micro_stateless_flags(self) -> List[bool]:
        """
        List of booleans stating whether micro models are stateless or not.

        Returns
        -------
        stateless_list : List[bool]
            Entry is True if micro model is stateless, False otherwise.
        """
        pass

    @config_entry
    def micro_output_n(self) -> int:
        """
        Get the micro output frequency

        Returns
        -------
        micro_output_n : int
            Output frequency of micro simulations, so output every N timesteps
        """
        pass

    @config_entry
    def enable_adaptivity(self) -> bool:
        """
        Boolean stating whether adaptivity is ot or not.

        Returns
        -------
        adaptivity : bool
            True is adaptivity settings are done, False otherwise.

        """
        pass

    @config_entry
    def adaptivity_type(self) -> str:
        """
        String stating type of adaptivity computation, either "local" or "global".

        Returns
        -------
        adaptivity_type : str
            Either "local" or "global" depending on the type of adaptivity computation
        """
        pass

    @config_entry
    def adaptivity_mapping_configs(self):
        """
        Get the mapping configurations for the adaptivity interpolation scheme.

        Returns
        -------
        adaptivity_mapping_configs : list
            List of adaptivity mapping configurations.
        """
        pass

    @config_entry
    def data_for_adaptivity(self) -> List[str]:
        """
        Get names of data to be used for similarity distance calculation in adaptivity

        Returns
        -------
        data_for_adaptivity : List[str]
            A list containing the names of the data to be used in adaptivity as keys and information on whether
            the data are scalar or vector as values.
        """
        pass

    @config_entry
    def adaptivity_n(self) -> int:
        """
        Get the frequency of adaptivity computation.

        Returns
        -------
        adaptivity_n : int
            Frequency of adaptivity computation, as a multiple of time windows.
        """
        pass

    @config_entry
    def adaptivity_output_type(self) -> str:
        """
        Get the type of adaptivity output.

        Returns
        -------
        adaptivity_output_type : str
            Type of adaptivity output, can be "all", "local" or "global".
        """
        pass

    @config_entry
    def adaptivity_output_n(self) -> int:
        """
        Get the output frequency of adaptivity metrics.

        Returns
        -------
        adaptivity_output_n : int
            Output frequency of adaptivity metrics, so output every N timesteps
        """
        pass

    @config_entry
    def adaptivity_history_param(self) -> float:
        """
        Get adaptivity history parameter.
        More details: https://precice.org/tooling-micro-manager-configuration.html#adaptivity

        Returns
        -------
        adaptivity_hist_param : float
            Adaptivity history parameter
        """
        pass

    @config_entry
    def adaptivity_coarsening_constant(self) -> float:
        """
        Get adaptivity coarsening constant.
        More details: https://precice.org/tooling-micro-manager-configuration.html#adaptivity

        Returns
        -------
        adaptivity_coarsening_constant : float
            Adaptivity coarsening constant
        """
        pass

    @config_entry
    def adaptivity_refining_constant(self) -> float:
        """
        Get adaptivity refining constant.
        More details: https://precice.org/tooling-micro-manager-configuration.html#adaptivity

        Returns
        -------
        adaptivity_refining_constant : float
            Adaptivity refining constant
        """
        pass

    @config_entry
    def adaptivity_similarity_measure(self) -> str:
        """
        Get measure to be used to calculate similarity between pairs of simulations.
        More details: https://precice.org/tooling-micro-manager-configuration.html#adaptivity

        Returns
        -------
        adaptivity_similarity_measure : str
            String of measure to be used in calculating similarity between pairs of simulations.
        """
        pass

    @config_entry
    def enable_adaptivity_each_implicit_iteration(self) -> bool:
        """
        Check if adaptivity needs to be calculated in every time iteration or every time window.

        Returns
        -------
        adaptivity_every_implicit_iteration : bool
            True if adaptivity needs to be calculated in every time iteration, False otherwise.
        """
        pass

    @config_entry
    def enable_load_balancing(self) -> bool:
        """
        Check if load balancing should be performed.

        Returns
        -------
        load_balancing : bool
            True if load balancing needs to be done, False otherwise.
        """
        pass

    @config_entry
    def load_balancing_type(self) -> str:
        """
        Get load balancing type.

        Returns
        -------
        type : str
            Load balancing type.
        """
        pass

    @config_entry
    def load_balancing_threshold(self) -> float:
        """
        Get load balancing threshold.

        Returns
        -------
        load_balancing_threshold : float
            Load balancing threshold
        """
        pass

    @config_entry
    def enable_load_balancing_inactive(self) -> bool:
        """
        Check if load balancing should be performed on inactive micro simulations.

        Returns
        -------
        balancing_inactive : bool
            True if load balancing should consider inactive simulations, False otherwise.
        """
        pass

    @config_entry
    def load_balancing_n(self) -> int:
        """
        Get the load balancing frequency.

        Returns
        -------
        load_balancing_n : int
            Load balancing frequency
        """
        pass

    @config_entry
    def load_balancing_partitioning(self) -> str:
        """
        Get the load balancing partitioning type

        Returns
        -------
        load_balancing_partitioning : str
            Load balancing partitioning type
        """
        pass

    @config_entry
    def enable_adaptivity_lazy_init(self) -> bool:
        """
        Check if simulations are to be created only when they are required to be active for the very first time.

        Returns
        -------
        enable_adaptivity_lazy_init : bool
            True if micro simulations are created only when needed, False otherwise.

        """
        pass

    @config_entry
    def micro_dt(self) -> float:
        """
        Get the size of the micro time window.

        Returns
        -------
        micro_time_window : float
            Size of the micro time window.
        """
        pass

    @config_entry
    def parameter_file_name(self) -> str:
        """
        Get the name of the parameter file.

        Returns
        -------
        parameter_file_name : str
            Name of the hdf5 file containing the macro parameters.
        """

        pass

    @config_entry
    def output_file_name(self) -> str:
        """
        Get the name of the output file.

        Returns
        -------
        output_file_name : str
            Name of the hdf5 file containing the snapshot data.
        """
        pass

    @config_entry
    def postprocessing_file_name(self) -> str:
        """
        Depending on user input, snapshot computation will perform post-processing for every micro simulation before writing output to a file.

        Returns
        -------
        postprocessing : str
            Name of post-processing script.
        """
        pass

    @config_entry
    def interpolation_configs(self) -> List[Dict[str, Any]]:
        """
        Gets the provided interpolation configurations.

        Returns
        interp_configs : List[Dict[str, Any]]
            Interpolation configurations.
        """
        pass

    @config_entry
    def enable_crashed_sim_interpolation(self) -> bool:
        """
        Check if user wants crashed micro simulations to be interpolated.

        Returns
        -------
        interpolate_crash : bool
            True if crashed micro simulations need to be interpolated, False otherwise.
        """
        pass

    @config_entry
    def crashed_sim_interpolation_id(self) -> Hashable:
        """
        Gets the associated interpolation config id used for crash interpolation.

        Returns
        -------
        interp_id : Hashable
            Interpolation config id.
        """
        pass

    @config_entry
    def crashed_sim_interpolation_threshold(self) -> float:
        """
        Gets the crash interpolation threshold.

        Returns
        -------
        threshold : float
            Threshold beyond which crashes are not recovered.
        """
        pass

    @config_entry
    def enable_single_sim_object(self) -> bool:
        """
        Check if multiple snapshots can be computed on a single micro simulation object.

        Returns
        -------
        initialize_once : bool
            True if initialization is done only once, False otherwise.
        """
        pass

    @config_entry
    def output_dir(self) -> str:
        """
        Get the name of the output directory.

        Returns
        -------
        output_dir : str
            Name of the output folder.
        """
        pass

    @config_entry
    def memory_usage_output_type(self) -> str:
        """
        Get the type of memory usage output.

        Returns
        -------
        mem_usage_output_type : str
            Type of adaptivity output, can be "all", "local" or "global".
        """
        pass

    @config_entry
    def memory_usage_output_n(self) -> int:
        """
        Get the output frequency of memory usage.

        Returns
        -------
        mem_usage_output_n : int
            Output frequency of memory usage, so output every N timesteps
        """
        pass

    @config_entry
    def enable_model_adaptivity(self) -> bool:
        """
        Boolean stating whether adaptivity is ot or not.

        Returns
        -------
        adaptivity : bool
            True is model adaptivity settings are done, False otherwise.

        """
        pass

    @config_entry
    def model_adaptivity_switching_function(self) -> str:
        """
        Get path to switching function file

        Returns
        -------
        switching_function : str
            String containing the path to the switching function file
        """
        pass

    @config_entry
    def tasking_num_workers(self) -> int:
        """
        Get number of workers

        Returns
        -------
        num_workers : int
            Number of workers
        """
        pass

    @config_entry
    def tasking_backend(self) -> str:
        """
        Get backend type

        Returns
        -------
        backend : str
            either socket or mpi
        """
        pass

    @config_entry
    def enable_tasking_slurm(self) -> bool:
        """
        Get flag whether slurm is used

        Returns
        -------
        use_slurm : bool
            use slurm or not
        """
        pass

    @config_entry
    def tasking_hostfile(self) -> str:
        """
        Get hostfile path for workers

        Returns
        -------
        hostfile : str
            Hostfile path for workers
        """
        pass

    @config_entry
    def mpi_impl(self) -> str:
        """
        Get mpi implementation type

        Returns
        -------
        mpi_impl : str
            mpi implementation type
        """
        pass
