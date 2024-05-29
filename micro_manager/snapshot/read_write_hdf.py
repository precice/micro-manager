import h5py
import numpy as np
from mpi4py import MPI
import os


class ReadWriteHDF:
    def __init__(self, logger) -> None:
        self._logger = logger

    def create_file(self, file_path: str) -> None:
        """
        Create a file with a given name in a given directory.

        Parameters
        ----------
        dir_name : str
            Directory in which the file should be created.
        file_name : str
            Name of the file to be created.

        """
        f = h5py.File(file_path, "w")
        f.close()

    def collect_output_files(self, dir_name: str, file_list: list) -> None:
        """
        Iterate over given files in a given directory and copy the content into a single file.

        Parameters
        ----------
        dir_name : str
            Path to directory containing the files.
        file_list : list
            List of files to be combined.
        """
        # Create a output file
        main_file = h5py.File(os.path.join(dir_name, "output.hdf5"), "w")
        # Loop over files
        for file in file_list:
            parameter_file = h5py.File(os.path.join(dir_name, file), "r")
            # Add all data sets to the main file.
            for key in parameter_file.keys():
                current_data = parameter_file[key][:]
                for i in range(len(current_data)):
                    # If data set does not exist create and add current data
                    if key not in main_file.keys():
                        main_file.create_dataset(
                            key,
                            data=current_data[i],
                            shape=(1, *current_data[i].shape),
                            maxshape=(None, *current_data[i].shape),
                        )

                    else:  # If dataset exists resize and add current data
                        main_file[key].resize(main_file[key].shape[0] + 1, axis=0)
                        main_file[key][-1] = current_data[i]
            parameter_file.close()
        main_file.close()

    def write_output_to_hdf(
        self, file_path: str, macro_data: dict, micro_data: dict
    ) -> None:
        """
        Write the output of a micro simulation to a hdf5 file.

        Parameters
        ----------
        file_path : str
            Path to file in which the data should be written.
        macro_data : dict
            Dict of macro simulation input.
        micro_data : dict
            Dict of micro simulation output.
        """
        parameter_file = h5py.File(file_path, "a")
        input_data = macro_data | micro_data
        # Iterate over macro and micro data sets

        for key in input_data.keys():
            current_data = np.asarray(input_data[key])
            # If dataset corresponding to dictionary key does not exist create and add current data
            if key not in parameter_file.keys():
                parameter_file.create_dataset(
                    key,
                    data=current_data,
                    shape=(1, *current_data.shape),
                    maxshape=(None, *current_data.shape),
                )
                parameter_file[key]
            # If dataset exists resize and add current data
            else:
                parameter_file[key].resize(parameter_file[key].shape[0] + 1, axis=0)
                parameter_file[key][-1] = current_data
        parameter_file.close()

    def read_hdf(self, file_path: str, data_names: dict) -> list:
        """
        Read data from a hdf5 file and return it as a list of dictionaries.

        Parameters
        ----------
        file_path : str
            Path of file to read data from.
        data_names : dict
            Names of parameters to read from the file.

        Returns
        -------
        list
            List of dicts where the keys are the names of the parameters and the values the corresponding data.
        """

        parameter_file = h5py.File(file_path, "r")
        parameter_data = dict()
        output = []
        # Read data by iterating over the relevant datasets
        for key in data_names.keys():
            parameter_data[key] = np.asarray(parameter_file[key][:])
            my_key = (
                key  # Save one key to be able to iterate over the length of the data
            )
        # Iterate over len of data. In each iteration write data from all macro data sets
        # to a dictionary and append it to the output list of dicts.
        for i in range(len(parameter_data[my_key])):
            current_data = dict()
            for key in data_names.keys():
                current_data[key] = parameter_data[key][i]
            output.append(current_data)
        return output
