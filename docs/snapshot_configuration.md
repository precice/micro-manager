---
title: Snapshot Computation
permalink: tooling-micro-manager-snapshot-configuration.html
keywords: tooling, macro-micro, two-scale, snapshot
summary: Set up the Micro Manager snapshot computation.
---

## Installation

To perform snapshot computations, the in the coupled case optional dependency `h5py` becomes mandatory. To install `micro-manager-precice` with `h5py`, run

```bash
pip install --user micro-manager-precice[snapshot]
```

If you have already installed `micro-manager-precice`, you can install `h5py` separately by running

```bash
pip install --user h5py
```

## Preparation

To prepare a micro simulation for the Micro Manager, follow the instructions in the [Micro Manager preparation guide](tooling-micro-manager-preparation.html).

Note: The `initialize()` method is not supported for the snapshot computation.

## Configuration

The Micro Manager snapshot computation is configured with a JSON file. An example configuration file is

```json
{
    "micro_file_name": "python-dummy/micro_dummy",
    "coupling_params": {
        "parameter_file_name": "parameter.hdf5",
        "read_data_names": {"macro-scalar-data": "scalar", "macro-vector-data": "vector"},
        "write_data_names": {"micro-scalar-data": "scalar", "micro-vector-data": "vector"},
    },
    "simulation_params": {
        "micro_dt": 1.0,
    },
    "snapshot_params": {
        "post_processing_file_name": "snapshot_postprocessing"
    },
    "diagnostics": {
        "output_micro_sim_solve_time": "True"
    }
}
```

This example configuration file is in [`examples/snapshot-config.json`](https://github.com/precice/micro-manager/tree/develop/examples/snapshot-config.json).

The path to the file containing the Python importable micro simulation class is specified in the `micro_file_name` parameter. If the file is not in the working directory, give the relative path.

There are four main sections in the configuration file, the `coupling_params`, the `simulations_params`, the `snapshot_params` and the optional `diagnostics`.

## Coupling Parameters

Parameter | Description
--- | ---
`parameter_file_name` | Path to the HDF5 file containing the parameter space from the current working directory. Each macro parameter must be given as a dataset. Macro data for the same micro simulation should have the same index in the first dimension. The name must correspond to the names given in the config file.
`read_data_names` | A Python dictionary with the names of the data to be read from preCICE as keys and `"scalar"` or `"vector"` as values depending on the nature of the data.
`write_data_names` | A Python dictionary with the names of the data to be written to the database as keys and `"scalar"` or `"vector"` as values depending on the nature of the data.

## Simulation Parameters

Parameter | Description
--- | ---
`micro_dt` | Initial time window size (dt) of the micro simulation. Must be set even if the micro simulation is time-independent.

## Snapshot Parameters

Parameter | Description
--- | ---
`post_processing_file_name`| Path to the post-processing Python script from the current working directory. Providing a post-processing script is optional. The script must contain a class `PostProcessing` with a method `postprocessing(sim_output)` that takes the simulation output as an argument. The method can be used to post-process the simulation output before writing it to the database.

## Diagnostics

Parameter | Description
--- | ---
`output_micro_sim_solve_time` | If `True`, the Micro Manager writes the wall clock time of the `solve()` function of each micro simulation to the database.

## Running

The Micro Manager snapshot computation is run directly from the terminal by adding the `--snapshot` argument and by providing the path to the configuration file as an input argument in the following way

```bash
micro-manager-precice --snapshot snapshot-config.json
```

The Micro Manager snapshot computation can also be run in parallel

```bash
mpiexec -n <number-of-processes> micro-manager-precice --snapshot snapshot-config.json
```

where `<number-of-processes>` must be replaced with the number of processes to be used.

### Results

The results of the snapshot computation are written into `output/` in HDF5-format. Each parameter is stored in a separate dataset. The dataset names correspond to the names specified in the configuration file. The first dimension of the datasets corresponds to the macro parameter index.

### What happens when a micro simulation crashes during snapshot computation?

If the computation of a snapshot fails, the snapshot is skipped.
