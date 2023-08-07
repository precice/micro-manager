"""
Class Config provides functionality to read a JSON file and pass the values to the Micro Manager.
"""

import json
import os


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

        self._macro_domain_bounds = None
        self._ranks_per_axis = None
        self._micro_output_n = 1
        self._diagnostics_data_names = dict()

        self._output_micro_sim_time = False

        self._adaptivity = False
        self._adaptivity_type = "local"
        self._data_for_adaptivity = dict()
        self._adaptivity_history_param = 0.5
        self._adaptivity_coarsening_constant = 0.5
        self._adaptivity_refining_constant = 0.5
        self._adaptivity_every_implicit_iteration = False
        self._adaptivity_similarity_measure = "L1"

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
        self._micro_file_name = data["micro_file_name"].replace("/", ".").replace("\\", ".").replace(".py", "")

        self._config_file_name = os.path.join(folder, data["coupling_params"]["config_file_name"])
        self._macro_mesh_name = data["coupling_params"]["macro_mesh_name"]

        try:
            self._write_data_names = data["coupling_params"]["write_data_names"]
            assert isinstance(self._write_data_names, dict), "Write data entry is not a dictionary"
            for key, value in self._write_data_names.items():
                if value == "scalar":
                    self._write_data_names[key] = False
                elif value == "vector":
                    self._write_data_names[key] = True
                else:
                    raise Exception("Write data dictionary as a value other than 'scalar' or 'vector'")
        except BaseException:
            self._logger.info("No write data names provided. Micro manager will only read data from preCICE.")

        try:
            self._read_data_names = data["coupling_params"]["read_data_names"]
            assert isinstance(self._read_data_names, dict), "Read data entry is not a dictionary"
            for key, value in self._read_data_names.items():
                if value == "scalar":
                    self._read_data_names[key] = False
                elif value == "vector":
                    self._read_data_names[key] = True
                else:
                    raise Exception("Read data dictionary as a value other than 'scalar' or 'vector'")
        except BaseException:
            self._logger.info("No read data names provided. Micro manager will only write data to preCICE.")

        self._macro_domain_bounds = data["simulation_params"]["macro_domain_bounds"]

        try:
            self._ranks_per_axis = data["simulation_params"]["decomposition"]
        except BaseException:
            self._logger.info(
                "Domain decomposition is not specified, so the Micro Manager will expect to be run in serial.")

        try:
            if data["simulation_params"]["adaptivity"]:
                self._adaptivity = True
            else:
                self._adaptivity = False
        except BaseException:
            self._logger.info(
                "Micro Manager will not adaptively run micro simulations, but instead will run all micro simulations in all time steps.")

        if self._adaptivity:
            if data["simulation_params"]["adaptivity"]["type"] == "local":
                self._adaptivity_type = "local"
            elif data["simulation_params"]["adaptivity"]["type"] == "global":
                self._adaptivity_type = "global"
                self._logger.warning(
                    "Global adaptivity is still experimental. We recommend using it for small (<50 macro vertices) cases only.")
            else:
                raise Exception("Adaptivity type can be either local or global.")

            exchange_data = {**self._read_data_names, **self._write_data_names}
            for dname in data["simulation_params"]["adaptivity"]["data"]:
                self._data_for_adaptivity[dname] = exchange_data[dname]

            self._adaptivity_history_param = data["simulation_params"]["adaptivity"]["history_param"]
            self._adaptivity_coarsening_constant = data["simulation_params"]["adaptivity"]["coarsening_constant"]
            self._adaptivity_refining_constant = data["simulation_params"]["adaptivity"]["refining_constant"]

            if "similarity_measure" in data["simulation_params"]["adaptivity"]:
                self._adaptivity_similarity_measure = data["simulation_params"]["adaptivity"]["similarity_measure"]
            else:
                self._logger.info("No similarity measure provided, using L1 norm as default")
                self._adaptivity_similarity_measure = "L1"

            adaptivity_every_implicit_iteration = data["simulation_params"]["adaptivity"]["every_implicit_iteration"]

            if adaptivity_every_implicit_iteration == "True":
                self._adaptivity_every_implicit_iteration = True
            elif adaptivity_every_implicit_iteration == "False":
                self._adaptivity_every_implicit_iteration = False

            if not self._adaptivity_every_implicit_iteration:
                self._logger.info("Micro Manager will compute adaptivity once at the start of every time window")

            self._write_data_names["active_state"] = False
            self._write_data_names["active_steps"] = False

        try:
            diagnostics_data_names = data["diagnostics"]["data_from_micro_sims"]
            assert isinstance(diagnostics_data_names, dict), "Diagnostics data is not a dictionary"
            for key, value in diagnostics_data_names.items():
                if value == "scalar":
                    self._write_data_names[key] = False
                elif value == "vector":
                    self._write_data_names[key] = True
                else:
                    raise Exception("Diagnostics data dictionary as a value other than 'scalar' or 'vector'")
        except BaseException:
            self._logger.info("No diagnostics data is defined. Micro Manager will not output any diagnostics data.")

        try:
            self._micro_output_n = data["diagnostics"]["micro_output_n"]
        except BaseException:
            self._logger.info(
                "Output interval of micro simulations not specified, if output is available then it will be called "
                "in every time window.")

        try:
            if data["diagnostics"]["output_micro_sim_solve_time"]:
                self._output_micro_sim_time = True
                self._write_data_names["micro_sim_time"] = False
        except BaseException:
            self._logger.info(
                "Micro manager will not output time required to solve each micro simulation in each time step.")

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
