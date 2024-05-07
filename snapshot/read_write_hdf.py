import h5py
import numpy as np
from mpi4py import MPI
import os


class ReadWriteHDF:
    def __init__(self, logger) -> None:
        self._logger = logger

    def create_file(self, dir_name: str, file_name: str) -> str:
        # create path to file
        file_path = os.path.join(dir_name, file_name)
        # create hdf file file in subfolder.
        f = h5py.File(file_path, "w")
        f.close()
        # return path to file
        return file_path

    def collect_output_files(self, dir_name: str, file_list: list) -> None:
        """
        Iterate over all given files in "output" directory and write them to a single file.
        """
        # 1. create a output file
        main_file = h5py.File(os.path.join(dir_name, "output.hdf5"), "w")
        for file in file_list:
            parameter_file = h5py.File(os.path.join(dir_name, file), "r")
            for key in parameter_file.keys():
                current_data = parameter_file[key][:]
                for i in range(len(current_data)):
                    if key not in main_file.keys():
                        main_file.create_dataset(
                            key,
                            data=current_data[i],
                            shape=(1, *current_data[i].shape),
                            maxshape=(None, *current_data[i].shape),
                        )
                    else:
                        main_file[key].resize(main_file[key].shape[0] + 1, axis=0)
                        main_file[key][-1] = current_data[i]
            parameter_file.close()
        main_file.close()

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

    def read_hdf_to_dict(self, file_name: str, data_names: dict) -> list:
        """
        Read an hdf5 file and return a dictionary of the given data.
        E.g. the if data_names corresponds to the macro data parameters a list of dictionaries with the macro data is returned.
        """
        # 1. open file in read mode
        parameter_file = h5py.File(file_name, "r")
        # 2. create an empty list
        parameter_data = dict()
        output = []
        # Read all the macro data
        for key in data_names.keys():

            parameter_data[key] = np.asarray(parameter_file[key][:])
            my_key = key
        # 3. iterate over len of data. In each iteration write data from all macro data sets (using the read_data_names as keys) to a dictionary with the key being the name of the data set
        # and the value being the current value.
        for i in range(len(parameter_data[my_key])):
            current_data = dict()
            for key in data_names.keys():
                current_data[key] = parameter_data[key][i]
            output.append(current_data)
        # 4. return the dictionary list
        return output
