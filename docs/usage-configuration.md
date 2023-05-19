---
title: Configuration of the Micro Manager
permalink: tooling-micro-manager-usage-configuration.html
keywords: tooling, macro-micro, two-scale
summary: The Micro Manager uses a JSON file to configure the coupling. The coupled data has to be specified in the preCICE configuration file.
---

The Micro Manager is configured at runtime using a JSON file `micro-manager-config.json`. An example configuration file is:

```json
{
    "micro_file_name": "micro_dummy",
    "coupling_params": {
        "config_file_name": "./precice-config.xml",
        "macro_mesh_name": "macro-mesh",
        "read_data_names": {"macro-scalar-data": "scalar", "macro-vector-data": "vector"},
        "write_data_names": {"micro-scalar-data": "scalar", "micro-vector-data": "vector"}
    },
    "simulation_params": {
        "macro_domain_bounds": [0.0, 25.0, 0.0, 25.0, 0.0, 25.0],
    },
    "diagnostics": {
      "output_micro_sim_solve_time": "True"
    }
}
```

There are three main sections in the configuration file, the `coupling_params`, the `simulation_params` and the optional `diagnostics`.
The file containing the python importable micro simulation class is specified in the `micro_file_name` parameter.

## Coupling Parameters

Parameter | Description
--- | ---
`config_file_name` |  Path to the preCICE XML configuration file.
`macro_mesh_name` |  Name of the macro mesh as stated in the preCICE configuration.
`read_data_names` |  A Python dictionary with the names of the data to be read from preCICE as keys and `"scalar"` or `"vector"`  as values.
`write_data_names` |  A Python dictionary with the names of the data to be written to preCICE as keys and `"scalar"` or `"vector"`  as values.

## Simulation Parameters

Parameter | Description
--- | ---
`macro_domain_bounds`| Minimum and maximum limits of the macro-domain, having the format `[xmin, xmax, ymin, ymax, zmin, zmax]`
*optional:* `micro_output_n`|  Frequency of calling the output functionality of the micro simulation in terms of number of time steps. If not given, `micro_sim.output()` is called every time step
*optional:* Adaptivity parameters | See section on [Adaptivity](#adaptivity). By default, adaptivity is disabled.

## *Optional*: Diagnostics

Parameter | Description
--- | ---
`data_from_micro_sims` | A Python dictionary with the names of the data from the micro simulation to be written to VTK files as keys and `"scalar"` or `"vector"` as values. This relies on the [export functionality](configuration-export.html#enabling-exporters) of preCICE and requires the corresponding export tag to be set in the preCICE XML configuration script.
`output_micro_sim_solve_time` | If `True`, the Manager writes the wall clock time of the `solve()` function of each micro simulation to the VTK output.

An example configuration file can be found in [`examples/micro-manager-config.json`](https://github.com/precice/micro-manager/tree/main/examples/micro-manager-config.json).

## Adaptivity

The Micro Manager can adaptively initialize micro simulations. The following adaptivity strategies are implemented:

1. Redeker, Magnus & Eck, Christof. (2013). A fast and accurate adaptive solution strategy for two-scale models with continuous inter-scale dependencies. Journal of Computational Physics. 240. 268-283. [10.1016/j.jcp.2012.12.025](https://doi.org/10.1016/j.jcp.2012.12.025).

2. Bastidas, Manuela & Bringedal, Carina & Pop, Iuliu. (2021). A two-scale iterative scheme for a phase-field model for precipitation and dissolution in porous media. Applied Mathematics and Computation. 396. 125933. [10.1016/j.amc.2020.125933](https://doi.org/10.1016/j.amc.2020.125933).

To turn on adaptivity, the following options need to be set in `simulation_params`:

Parameter | Description
--- | ---
`adaptivity` | Set as `True` to turn on adaptivity.
`adaptivity_data` | List of names of data which are to be used to calculate if two micro-simulations are similar or not. For example `["macro-scalar-data", "macro-vector-data"]`
`adaptivity_history_param` | History parameter $$ \Lambda $$, set as $$ \Lambda >= 0 $$.
`adaptivity_coarsening_constant` | Coarsening constant $$ C_c $$, set as $$ C_c < 1 $$.
`adaptivity_refining_constant` | Refining constant $$ C_r $$, set as $$ C_r >= 0 $$.
`adaptivity_every_implicit_iteration` | If True, adaptivity is calculated in every implicit iteration. <br> If False, adaptivity is calculated once at the start of the time window and then reused in every implicit time iteration.

All variables names are chosen to be same as the [second publication](https://doi.org/10.1016/j.amc.2020.125933) mentioned above.

If adaptivity is turned on, the Micro Manager will attempt to write a scalar data set `active_state` to preCICE. Add this data set to the preCICE configuration file. In the mesh and the micro participant add the following lines:

```xml
    <data:scalar name="active_state"/>
    <mesh name="macro-mesh">
       <use-data name="active_state"/>
    </mesh>
    <participant name="micro-mesh">
       <write-data name="active_state" mesh="macro-mesh"/>
    </participant>
```

TODO: what about active_steps? from the examples?

## Next Steps

After creating a configuration file you are ready to [run the Micro Manager](tooling-micro-manager-usage-running.html).
