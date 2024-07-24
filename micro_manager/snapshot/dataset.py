from importlib import metadata
import os
from datetime import datetime

import numpy as np

try:
    import h5py
except ImportError:
    raise ImportError(
        "The Micro Manager snapshot computation requires the h5py package."
    )


class ReadWriteHDF:
    def __init__(self, logger) -> None:
        self._logger = logger
        self._has_datasets = False

    def create_file(self, file_path: str) -> None:
        """
        Create an HDF5 file for a given file name and path.

        Parameters
        ----------
        file_path : str
            File name added to the path to the file.

        """
        f = h5py.File(file_path, "w")
        f.attrs["status"] = "writing"
        f.attrs["MicroManager_version"] = str(metadata.version("micro-manager-precice"))
        f.attrs["date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.close()

    def collect_output_files(
        self, dir_name: str, file_list: list, database_length: int
    ) -> None:
        """
        Iterate over a list of HDF5 files in a given directory and copy the content into a single file.
        The files are deleted after the content is copied.

        Parameters
        ----------
        dir_name : str
            Path to directory containing the files.
        file_list : list
            List of files to be combined.
        dataset_length : int
            Global number of snapshots.
        """
        # Create a output file
        main_file = h5py.File(os.path.join(dir_name, "snapshot_data.hdf5"), "w")
        main_file.attrs["status"] = "writing"
        main_file.attrs["MicroManager_version"] = str(
            metadata.version("micro-manager-precice")
        )
        main_file.attrs["date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Create datasets in output file
        with h5py.File(os.path.join(dir_name, file_list[0]), "r") as parameter_file:
            for key in parameter_file.keys():
                if not key == "crashed_snapshots":
                    current_data = parameter_file[key][0]
                    main_file.create_dataset(
                        key,
                        shape=(database_length, *current_data.shape),
                        chunks=(1, *current_data.shape),
                        fillvalue=np.nan,
                    )
        # Loop over files
        crashed_snapshots = []
        outer_position = 0
        for file in file_list:
            parameter_file = h5py.File(os.path.join(dir_name, file), "r")
            # Add all data sets to the main file.
            for key in parameter_file.keys():
                inner_position = outer_position
                for chunk in parameter_file[key].iter_chunks():
                    current_data = parameter_file[key][chunk]
                    # If the key is "crashed_snapshots" add the indices to the list of crashed snapshots
                    # Otherwise write the data to the main file
                    if key == "crashed_snapshots":
                        crashed_snapshots.extend(
                            inner_position + parameter_file[key][:]
                        )
                    else:
                        main_file[key][inner_position] = current_data
                    inner_position += 1
            outer_position = inner_position
            parameter_file.close()
            os.remove(os.path.join(dir_name, file))

        # Write the indices of crashed snapshots to the main file
        if len(crashed_snapshots) > 0:
            main_file.create_dataset(
                "crashed_snapshots", data=crashed_snapshots, dtype=int
            )
        main_file.attrs["status"] = "finished"
        main_file.close()

    def write_output_to_hdf(
        self,
        file_path: str,
        macro_data: dict,
        micro_data: dict | None,
        idx: int,
        length: int,
    ) -> None:
        """
        Write the output of a micro simulation to a HDF5 file.

        Parameters
        ----------
        file_path : str
            Path to file in which the data should be written.
        macro_data : dict
            Dict of macro simulation input.
        micro_data : dict | None
            Dict of micro simulation output. If None, only the macro data is written.
        idx: int
            Local index of the current snapshot.
        length : int
            Local number of snapshots.
        """
        parameter_file = h5py.File(file_path, "a")
        if micro_data is None:
            input_data = macro_data
        else:
            input_data = macro_data | micro_data

        #  If the datasets are not created yet, create them
        if not self._has_datasets:
            for key in input_data.keys():
                current_data = np.asarray(input_data[key])
                parameter_file.create_dataset(
                    key,
                    shape=(length, *current_data.shape),
                    chunks=(1, *current_data.shape),
                    fillvalue=np.nan,
                )
            self._has_datasets = True

        # Iterate over macro and micro data sets and write current simulation data to the file
        for key in input_data.keys():
            current_data = np.asarray(input_data[key])
            parameter_file[key][idx] = current_data

        parameter_file.close()

    def read_hdf(self, file_path: str, data_names: dict, start: int, end: int) -> list:
        """
        Read data from an HDF5 file and return it as a list of dictionaries.

        Parameters
        ----------
        file_path : str
            Path of file to read data from.
        data_names : dict
            Names of parameters to read from the file.
        start: int
            Index of the first snapshot to read on process.
        end: int
            Index of the last snapshot to read on process.

        Returns
        -------
        output: list
            List of dicts where the keys are the names of the parameters and the values the corresponding data.
        """

        parameter_file = h5py.File(file_path, "r")
        parameter_data = dict()
        output = []
        # Read data by iterating over the relevant datasets
        for key in data_names.keys():
            parameter_data[key] = np.asarray(parameter_file[key][start:end])
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

    def get_parameter_space_size(self, file_path: str) -> int:
        """
        Get the length of the parameter space from the HDF5 file.

        Parameters
        ----------
        file_path : str
            Path of file to read data from.

        Returns
        -------
        int
            Size of Parameter Space
        """
        with h5py.File(file_path, "r") as file:
            return file[list(file.keys())[0]].len()

    def write_crashed_snapshots(self, file_path: str, crashed_input: list):
        """
        Write indices of crashed snapshots to the HDF5 database.

        Parameters
        ----------
        file_path : str
            Path of file to read data from.
        crashed_indices : list
            list of indices of crashed simulations.
        """
        with h5py.File(file_path, "a") as file:
            file.create_dataset("crashed_snapshots", data=crashed_input, dtype=int)

    def set_status(self, file_path: str, status: str):
        """
        Set the status of the file to "finished" to indicate that it is no longer accessed.

        Parameters
        ----------
        file_path : str
            Path of file to read data from.
        """
        with h5py.File(file_path, "a") as file:
            file.attrs["status"] = status
