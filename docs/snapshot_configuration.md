---
title: Snapshot Computation
permalink: tooling-micro-manager-snapshot-configuration.html
aliases:
  - /tooling-micro-manager-snapshot-configuration.html
keywords: tooling, macro-micro, two-scale, snapshot
summary: Set up the Micro Manager snapshot computation.
---

## Installation

To use the Micro Manager for snapshot computation, the dependency `h5py` is necessary. To install `micro-manager-precice` with `h5py`, run

```bash
pip install micro-manager-precice[snapshot]
```

If you have already installed `micro-manager-precice`, you can install `h5py` separately by running

```bash
pip install h5py
```

## Preparation

Prepare your micro simulation for the Micro Manager snapshot computation by following the instructions in the [preparation guide](tooling-micro-manager-prepare-micro-simulation.html).

Note: The `initialize()` method is not used for the snapshot computation.

## Configuration

Configure the snapshot computation functionality with a JSON file. An example configuration file is

```json
{
    "micro_manager": {
        "micro_file_names": ["python-dummy/micro_dummy"],
        "output_directory": "output",
        "coupling_params": {
            "parameter_file_name": "parameter.hdf5",
            "read_data_names": ["macro-scalar-data", "macro-vector-data"],
            "write_data_names": ["micro-scalar-data", "micro-vector-data"],
        },
        "simulation_params": {
            "micro_dt": 1.0,
        },
        "snapshot_params": {
            "post_processing_file_name": "snapshot_postprocessing",
            "initialize_once": true,
            "output_file_name": "snapshot_data"
        }
    }
}
```

Unlike the coupled Micro Manager, the snapshot computation does not use a preCICE participant, so it does not require `participant_name`, `precice_config_file_path`, or `interfaces` at the top level, and the configuration is not validated against the [preCICE adapter configuration schema](https://github.com/precice/adapter-schema). Instead, everything is nested under the top-level `micro_manager` key.

## Micro Manager Configuration

Parameter | Description | Default
--- | --- | ---
`micro_file_names` | Paths to the files containing the Python importable micro simulation classes. If the files are not in the working directory, give the relative paths from the directory where the Micro Manager is executed. | -
`output_directory` | Path to output directory for logging and performance metrics. Directory is created if not existing already. | `.`

Apart from the base settings, there are three main sections in the `micro_manager` section of the configuration file, [coupling parameters](#coupling-parameters), [simulation parameters](#simulation-parameters), [snapshot parameters](#snapshot-parameters), and [diagnostics](#diagnostics).

The paths to the files containing the Python importable micro simulation classes are specified in the `micro_file_names` parameter. If the files are not in the working directory, give the relative paths.

There are four main sections in the `micro_manager` section of the configuration file, the `coupling_params`, the `simulations_params`, the `snapshot_params` and the optional `diagnostics`.

## Coupling Parameters

Parameter | Description
--- | ---
`parameter_file_name` | Path to the HDF5 file containing the parameter space from the current working directory. Each macro parameter must be given as a dataset. Macro data for the same micro simulation should have the same index in the first dimension. The name must correspond to the names given in the config file.
`read_data_names` | A Python list with the names of the data to be read from preCICE.
`write_data_names` | A Python list with the names of the data to be written to the database.

## Simulation Parameters

Parameter | Description
--- | ---
`micro_dt` | Initial time window size (dt) of the micro simulation. Must be set even if the micro simulation is time-independent.

## Snapshot Parameters

Parameter | Description
--- | ---
`post_processing_file_name` | Path to the post-processing Python script from the current working directory. Providing a post-processing script is optional. The script must contain a class `PostProcessing` with a method `postprocessing(sim_output)` that takes the simulation output as an argument. The method can be used to post-process the simulation output before writing it to the database.
`initialize_once` | If `true`, only one micro simulation is initialized and solved for all macro inputs per rank. If `false` a new micro simulation is initialized and solved for each macro input in the parameter space. Default is `false`. It is recommended to set the parameter to `true` if the micro simulation is not history-dependent and the same setup is shared across all micro simulations.
`output_file_name` | Name of the HDF5 file which stores the simulation inputs and output data snapshots. Default is `"snapshot_data"`.

## Diagnostics

Parameter | Description | Default
--- | --- | ---
`output_micro_sim_solve_time` | If `true`, the Micro Manager writes the wall clock time of the `solve()` function of each micro simulation to the database. | `false`

## Running

Run the snapshot computation directly from the terminal by adding the `--snapshot` argument to the Micro Manager executable, and by providing the path to the configuration file as an input argument in the following way

```bash
micro-manager-precice --snapshot snapshot-config.json
```

Run the snapshot computation in parallel by

```bash
mpiexec -n <number-of-processes> micro-manager-precice --snapshot snapshot-config.json
```

where `<number-of-processes>` is the number of processes used.

### Results

The results of the snapshot computation are written into `output/` in HDF5-format. Each parameter is stored in a separate dataset. The dataset names correspond to the names specified in the configuration file. The first dimension of the datasets corresponds to the macro parameter index.

### What happens when a micro simulation crashes during snapshot computation?

If the computation of a snapshot fails, the snapshot is skipped.
