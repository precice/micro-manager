"""
Configuration module of the Micro Manager
"""

import json
import os
import sys


class Config:
    """
    Handles the reading of parameters in the JSON configuration file provided by the user. This class is based on
    the config class in https://github.com/precice/fenics-adapter/tree/develop/fenicsadapter
    """

    def __init__(self, config_filename):
        self._micro_file_name = None

        self._config_file_name = None
        self._macro_mesh_name = None
        self._read_data_names = None
        self._write_data_names = None

        self._macro_domain_bounds = None

        self.read_json(config_filename)

    def read_json(self, config_filename):
        """
        Reads JSON adapter configuration file and saves the data to the respective instance attributes.
        """
        folder = os.path.dirname(os.path.join(os.getcwd(), os.path.dirname(sys.argv[0]), config_filename))
        path = os.path.join(folder, os.path.basename(config_filename))
        read_file = open(path, "r")
        data = json.load(read_file)

        try:
            self._micro_file_name = data["micro_file_name"]
            i = 0
            micro_filename = list(self._micro_file_name)
            for c in micro_filename:
                if c == '/':
                    micro_filename[i] = '.'
                i += 1
            self._micro_file_name = ''.join(micro_filename)
        except BaseException:
            self._micro_file_name = "No micro file provided"

        self._config_file_name = os.path.join(folder, data["coupling_params"]["config_file_name"])
        self._macro_mesh_name = data["coupling_params"]["macro_mesh_name"]

        self._write_data_names = data["coupling_params"]["write_data_names"]
        assert isinstance(self._write_data_names, dict), "Entity write_data_name is not a dictionary"

        for key, value in self._write_data_names.items():
            if value == "scalar":
                self._write_data_names[key] = False
            elif value == "vector":
                self._write_data_names[key] = True
            else:
                raise Exception("Write data dictionary as a value other than 'scalar' or 'vector'")

        self._read_data_names = data["coupling_params"]["read_data_names"]
        assert isinstance(self._read_data_names, dict), "Entity read_data_name is not a dictionary"

        for key, value in self._read_data_names.items():
            if value == "scalar":
                self._read_data_names[key] = False
            elif value == "vector":
                self._read_data_names[key] = True
            else:
                raise Exception("Read data dictionary as a value other than 'scalar' or 'vector'")

        self._macro_domain_bounds = data["simulation_params"]["macro_domain_bounds"]

        read_file.close()

    def get_config_file_name(self):
        return self._config_file_name

    def get_macro_mesh_name(self):
        return self._macro_mesh_name

    def get_read_data_names(self):
        return self._read_data_names

    def get_write_data_names(self):
        return self._write_data_names

    def get_macro_domain_bounds(self):
        return self._macro_domain_bounds

    def get_micro_file_name(self):
        return self._micro_file_name
