---
title: Configure the Snapshot Computation
permalink: tooling-micro-manager-snapshot-configuration.html
keywords: tooling, macro-micro, two-scale, snapshot
summary: Provide a JSON file to configure the Micro Manager snapshot computation.
---

> Note: To install the Micro Manager for the snapshot computation, follow the instructions in the [Micro Manager installation guide](tooling-micro-manager-installation.html).
> To prepare a micro simulation for the Micro Manager, follow the instructions in the [Micro Manager preparation guide](tooling-micro-manager-preparation.html).
> Currently, the Micro Manager Snapshot tool can not be used with an `initialize()` method.

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
`write_data_names` | A Python dictionary with the names of the data to be written to preCICE as keys and `"scalar"` or `"vector"` as values depending on the nature of the data.

## Simulation Parameters

Parameter | Description
--- | ---
`micro_dt` | Initial time window size (dt) of the micro simulation. Must be set even if the micro simulation is time-independent.

## Snapshot Parameters

Parameter | Description
--- | ---
`post_processing_file_name`| Path to the post-processing script from the current working directory.

## Diagnostics

Parameter | Description
--- | ---
`output_micro_sim_solve_time` | If `True`, the Micro Manager writes the wall clock time of the `solve()` function of each micro simulation.

## Next step

After creating a configuration file you are ready to [run the Micro Manager snapshot computation](tooling-micro-manager-running.html/#snapshot-computation).
