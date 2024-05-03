import h5py
import numpy as np
from mpi4py import MPI
import os


class HDFParameterfile:
    def __init__(self, logger, rank, world) -> None:
        self._logger = logger
        self._rank = rank
        self._world = world

    def create_file(self, dir_name: str) -> str:
        # create path to file
        file_path = os.path.join(dir_name, "database_{}.hdf5".format(self._rank))
        # create hdf file file in subfolder.
        f = h5py.File(file_path, "w")
        f.close()
        # return path to file
        return file_path

    def collect_parameters_files(
        self, parameter_file: str, write_data_names: dict, read_data_names: dict
    ) -> None:
        """
        Iterate over all parameter files and write them to a single file.
        """
        # 1. read the base file in main mode
        # 2. iterate over all files in subdirectory (over world) and append their data to the base file

        pass

    def write_dict_to_hdf(self, file_name, input_data, mode="a") -> None:
        """
        Take a dictionary and write it to an hdf5 file.
        Data is a dict where each key is a parameter and each value is a numpy array or list containing the data.
        e.g. grain radius and average porosity: {grain_radius: np.array([1, 2, 3]), average_porosity: np.array([0.1, 0.2, 0.3])}
        or stress: {stress: np.array([1,0,0,0,0,0])}

        """
        parameter_file = h5py.File(file_name, mode)
        # Check if data sets corresponding to the keys in the dict. If not create them with max length none and the shape of their first entry
        # For all entries in the dict or all dicts in list of dict write the data to the corresponding data set
        if isinstance(input_data, list):
            for i in range(len(input_data)):
                for key in input_data[i].keys():
                    current_data = np.asarray(input_data[i][key])
                    if key not in parameter_file.keys():
                        parameter_file.create_dataset(
                            key,
                            data=current_data,
                            shape=(1, *current_data.shape),
                            maxshape=(None, *current_data.shape),
                        )
                        parameter_file[key]
                    else:
                        parameter_file[key].resize(
                            parameter_file[key].shape[0] + 1, axis=0
                        )
                        parameter_file[key][-1] = current_data
            parameter_file.close()
        else:
            for key in input_data.keys():
                current_data = np.asarray(input_data[key])
                if key not in parameter_file.keys():
                    parameter_file.create_dataset(
                        key,
                        data=current_data,
                        shape=(1, *current_data.shape),
                        maxshape=(None, *current_data.shape),
                    )
                    parameter_file[key]
                else:
                    parameter_file[key].resize(parameter_file[key].shape[0] + 1, axis=0)
                    parameter_file[key][-1] = current_data
            parameter_file.close()

    def write_sim_output_to_hdf(
        self, file_name: str, macro_data: dict, micro_data: dict
    ) -> None:
        parameter_file = h5py.File(file_name, "a")
        # Check if data sets corresponding to the keys in the dict. If not create them with max length none and the shape of their first entry
        # For all entries in the dict or all dicts in list of dict write the data to the corresponding data set
        input_data = [macro_data, micro_data]
        # Check if data sets corresponding to the keys in the dict. If not create them with max length none and the shape of their first entry
        # For all entries in the dict or all dicts in list of dict write the data to the corresponding data set
        for i in range(len(input_data)):
            for key in input_data[i].keys():
                current_data = np.asarray(input_data[i][key])
                if key not in parameter_file.keys():
                    parameter_file.create_dataset(
                        key,
                        data=current_data,
                        shape=(1, *current_data.shape),
                        maxshape=(None, *current_data.shape),
                    )
                    parameter_file[key]
                else:
                    parameter_file[key].resize(parameter_file[key].shape[0] + 1, axis=0)
                    parameter_file[key][-1] = current_data
        parameter_file.close()

    def read_parameter_hdf_to_dict(self, file_name: str, read_data_names: dict) -> list:
        """
        Read an hdf5 file and return a dictionary of the parameter (macro) data.
        """
        # 1. open file in read mode
        parameter_file = h5py.File(file_name, "r")
        # 2. create an empty list
        parameter_data = dict()
        output = []
        # Read all the macro data
        self._logger.info("Reading macro data {}".format(read_data_names))
        for key in read_data_names.keys():

            parameter_data[key] = np.asarray(parameter_file[key][:])
            my_key = key
        # 3. iterate over len of data. In each iteration write data from all macro data sets (using the read_data_names as keys) to a dictionary with the key being the name of the data set
        # and the value being the current value.
        for i in range(len(parameter_data[my_key])):
            current_data = dict()
            for key in read_data_names.keys():
                current_data[key] = parameter_data[key][i]
            output.append(current_data)
        # 4. return the dictionary list
        return output

    def read_micro_hdf_to_dict(self, file_name: str, write_data_names: dict):
        """
        Read an hdf5 file and return a dictionary of the micro data.
        """

        # 1. open file in read mode
        parameter_file = h5py.File(file_name, "r")
        # 2. create an empty list
        parameter_data = dict()
        output = []
        # Read all the macro data
        self._logger.info("Reading micro data {}".format(write_data_names))
        for key in write_data_names.keys():

            parameter_data[key] = np.asarray(parameter_file[key][:])
            my_key = key
        # 3. iterate over len of data. In each iteration write data from all macro data sets (using the read_data_names as keys) to a dictionary with the key being the name of the data set
        # and the value being the current value.
        for i in range(len(parameter_data[my_key])):
            current_data = dict()
            for key in write_data_names.keys():
                current_data[key] = parameter_data[key][i]
            output.append(current_data)
        # 4. return the dictionary list
        return output

    def read_all_hdf_to_dict(self, file_name: str, write_data_names, read_data_names):
        """
        Read an hdf5 file and return a dictionary of both parameter (macro) and micro data.
        """

        macro = self.read_parameter_hdf_to_dict(file_name, read_data_names)
        micro = self.read_micro_hdf_to_dict(file_name, write_data_names)

        return [macro, micro]
