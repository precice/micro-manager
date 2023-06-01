---
title: Configure the Micro Manager
permalink: tooling-micro-manager-configuration.html
keywords: tooling, macro-micro, two-scale
summary: The Micro Manager uses a JSON file to configure the coupling. The coupled data has to be specified in the preCICE configuration file.
---

The Micro Manager is configured at runtime using a JSON file. An example configuration file is

```json
{
    "micro_file_name": "micro_solver",
    "coupling_params": {
        "config_file_name": "precice-config.xml",
        "macro_mesh_name": "macro-mesh",
        "read_data_names": {"temperature": "scalar", "heat-flux": "vector"},
        "write_data_names": {"porosity": "scalar", "conductivity": "vector"}
    },
    "simulation_params": {
        "macro_domain_bounds": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    },
    "diagnostics": {
      "output_micro_sim_solve_time": "True"
    }
}
```

There are three main sections in the configuration file, the `coupling_params`, the `simulation_params` and the optional `diagnostics`.

The path to the file containing the Python importable micro simulation class is specified in the `micro_file_name` parameter. If the file is not in the working directory, give the relative path.

## Coupling Parameters

Parameter | Description
--- | ---
`config_file_name` |  Path to the preCICE XML configuration file from the current working directory.
`macro_mesh_name` |  Name of the macro mesh as stated in the preCICE configuration.
`read_data_names` |  A Python dictionary with the names of the data to be read from preCICE as keys and `"scalar"` or `"vector"`  as values depending on the nature of the data.
`write_data_names` |  A Python dictionary with the names of the data to be written to preCICE as keys and `"scalar"` or `"vector"`  as values depending on the nature of the data.

## Simulation Parameters

Parameter | Description
--- | ---
`macro_domain_bounds`| Minimum and maximum bounds of the macro-domain, having the format `[xmin, xmax, ymin, ymax, zmin, zmax]` in 3D and `[xmin, xmax, ymin, ymax]` in 2D.
*optional:* `adaptivity_similarity_measure`| Similarity measure to be used for adaptivity. Can be either `L1`, `L2`, `L1rel` or `L2rel`. By default, `L1` is used. The `rel` variants calculate the respective relative norms.
*optional:* Domain decomposition parameters | See section on [Domain decomposition](#domain-decomposition). But default, the Micro Manager assumes that it will be run in serial.
*optional:* Adaptivity parameters | See section on [Adaptivity](#adaptivity). By default, adaptivity is disabled.

## *Optional*: Diagnostics

Parameter | Description
--- | ---
`data_from_micro_sims` | A Python dictionary with the names of the data from the micro simulation to be written to VTK files as keys and `"scalar"` or `"vector"` as values. This relies on the [export functionality](configuration-export.html#enabling-exporters) of preCICE and requires the corresponding export tag to be set in the preCICE XML configuration script.
`output_micro_sim_solve_time` | If `True`, the Manager writes the wall clock time of the `solve()` function of each micro simulation to the VTK output.
`micro_output_n`|  Frequency of calling the optional output functionality of the micro simulation in terms of number of time steps. If not given, `micro_sim.output()` is called every time step.

An example configuration file can be found in [`examples/micro-manager-config.json`](https://github.com/precice/micro-manager/tree/main/examples/micro-manager-config.json).

## Domain decomposition

The Micro Manager can be run in parallel. For a parallel run, set the desired number of paritions in each axis by setting the `axiswise_ranks` variable. For example, if the domain is 3D and the decomposition needs to be two paritions in x, one partition in y, and sixteen partitions in z, the setting is

```json
"simulation_params": {
    "macro_domain_bounds": [0, 1, 0, 1, 0, 1],
	"axiswise_ranks": [2, 1, 16]
}
```

For a 2D domain, only two values need to be set `axiswise_ranks`.

## Adaptivity

{% note %} This feature is optional. {% endnote %}

The Micro Manager can adaptively control micro simulations. The adaptivity strategy is taken from

1. Redeker, Magnus & Eck, Christof. (2013). A fast and accurate adaptive solution strategy for two-scale models with continuous inter-scale dependencies. Journal of Computational Physics. 240. 268-283. [10.1016/j.jcp.2012.12.025](https://doi.org/10.1016/j.jcp.2012.12.025).

2. Bastidas, Manuela & Bringedal, Carina & Pop, Iuliu. (2021). A two-scale iterative scheme for a phase-field model for precipitation and dissolution in porous media. Applied Mathematics and Computation. 396. 125933. [10.1016/j.amc.2020.125933](https://doi.org/10.1016/j.amc.2020.125933).

To turn on adaptivity, the following options need to be set in `simulation_params` under the sub-heading `adaptivity`:

Parameter | Description
--- | ---
`type` | Set to either `local` or `global`. The type of adaptivity matters when the Micro Manager is run in parallel. `local` means comparing micro simulations within a local partitioned domain for similarity. `global` means comparing micro simulations from all partitions, so over the entire domain.
`data` | List of names of data which are to be used to calculate if micro-simulations are similar or not. For example `["temperature", "porosity"]`.
`history_param` | History parameter $$ \Lambda $$, set as $$ \Lambda >= 0 $$.
`coarsening_constant` | Coarsening constant $$ C_c $$, set as $$ C_c < 1 $$.
`refining_constant` | Refining constant $$ C_r $$, set as $$ C_r >= 0 $$.
`<every_implicit_iteration` | If True, adaptivity is calculated in every implicit iteration. <br> If False, adaptivity is calculated once at the start of the time window and then reused in every implicit time iteration.

All variables are chosen from the [second publication](https://doi.org/10.1016/j.amc.2020.125933) mentioned above.

Example of adaptivity configuration

```json
"simulation_params": {
    "macro_domain_bounds": [0, 1, 0, 1, 0, 1],
    "adaptivity" {
        "type": "local",
        "data": ["temperature", "porosity"],
        "history_param": 0.5,
        "coarsening_constant": 0.3,
        "refining_constant": 0.4,
        "every_implicit_iteration": "True"
    }
}
```

If adaptivity is turned on, the Micro Manager will attempt to write two scalar data per micro simulation to preCICE, called `active_state` and `active_steps`.

Parameter | Description
--- | ---
`active_state` | `1` if the micro simulation is active in the time window, and `0` if inactive.
`active_steps` | Summation of `active_state` up to the current time window.

The Micro Manager uses the output functionality of preCICE, hence these data sets to the preCICE configuration file. In the mesh and the micro participant add the following lines:

```xml
<data:scalar name="active_state"/>
<data:scalar name="active_steps"/>

<mesh name="macro-mesh">
    <use-data name="active_state"/>
    <use-data name="active_steps"/>
</mesh>

<participant name="micro-mesh">
    <write-data name="active_state" mesh="macro-mesh"/>
    <write-data name="active_steps" mesh="macro-mesh"/>
</participant>
```

## Next Steps

After creating a configuration file you are ready to [run the Micro Manager](tooling-micro-manager-usage-running.html).
