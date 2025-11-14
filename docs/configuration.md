---
title: Configure the Micro Manager
permalink: tooling-micro-manager-configuration.html
keywords: tooling, macro-micro, two-scale
summary: Provide a JSON file to configure the Micro Manager.
---

{% note %} In the preCICE XML configuration the Micro Manager is a participant with the name `Micro-Manager`. {% endnote %}

The Micro Manager is configured with a JSON file. An example configuration file looks like

```json
{
    "micro_file_name": "micro_solver",
    "coupling_params": {
        "precice_config_file_name": "precice-config.xml",
        "macro_mesh_name": "macro-mesh",
        "read_data_names": ["temperature", "heat-flux"],
        "write_data_names": ["porosity", "conductivity"]
    },
    "simulation_params": {
        "macro_domain_bounds": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        "micro_dt": 1.0
    }
}
```

This example configuration file is in [`examples/micro-manager-config.json`](https://github.com/precice/micro-manager/tree/develop/examples/micro-manager-config.json).

## Micro Manager Configuration

Parameter | Description
--- | ---
`micro_file_name` | Path to the file containing the Python importable micro simulation class. If the file is not in the working directory, give the relative path from the directory where the Micro Manager is executed.
`output_directory` | Path to output directory for logging and performance metrics. Directory is created if not existing already.
`memory_usage_output_type` | Set to either `local`, `global`, or `all`. `local` outputs rank-wise peak memory usage. `global` outputs global averaged peak memory usage. `all` outputs both local and global levels. All output is to a CSV file with the peak memory usage (RSS) in every time window, in MBs.
`memory_usage_output_n` | Frequency of output of memory usage (integer which is number of time windows).

Apart from the base settings, there are three main sections in the configuration file, [coupling parameters](#coupling-parameters), [simulation parameters](#simulation-parameters), and [diagnostics](#diagnostics).

## Coupling Parameters

Parameter | Description
--- | ---
`precice_config_file_name` |  Path to the preCICE XML configuration file from the current working directory.
`macro_mesh_name` |  Name of the macro mesh as stated in the preCICE configuration.
`read_data_names` |  A list with the names of the data to be read from preCICE.
`write_data_names` |  A list with the names of the data to be written to preCICE.

## Simulation Parameters

Parameter | Description
--- | ---
`macro_domain_bounds`| Minimum and maximum bounds of the macro-domain, having the format `[xmin, xmax, ymin, ymax, zmin, zmax]` in 3D and `[xmin, xmax, ymin, ymax]` in 2D.
Domain decomposition parameters | See section on [domain decomposition](#domain-decomposition). But default, the Micro Manager assumes serial execution.
Adaptivity parameters | See section on [adaptivity](#adaptivity). By default, adaptivity is disabled.
Load balancing parameters | See section on [load balancing](#load-balancing). By default, load balancing is disabled.
`micro_dt` | Initial time window size (dt) of the micro simulation.

## Diagnostics

Parameter | Description
--- | ---
`data_from_micro_sims` | A Python dictionary with the names of the data from the micro simulation to be written to VTK files as keys and `"scalar"` or `"vector"` as values.
`micro_output_n`|  Frequency of calling the optional output functionality of the micro simulation in terms of number of time steps. If not given, `micro_sim.output()` is called every time step.

### Adding diagnostics in the preCICE XML configuration

If the parameter `data_from_micro_sims` is set, the data to be output needs to be written to preCICE, and an export tag needs to be added for the participant `Micro-Manager`. For example, let us consider the case that the data `porosity`, which is a scalar, needs to be exported. Unless already defined, define the data, and then write it to preCICE. Also, add an export tag. The resulting entries in the XML configuration file look like:

```xml
<data:scalar name="porosity"/>

<participant name="Micro-Manager">
  ...
  <write-data name="porosity" mesh="macro-mesh"/>
  <export:vtu directory="Micro-Manager-output" every-n-time-windows="5"/>
</participant>
```

## Domain decomposition

The Micro Manager can be run in parallel. For a parallel run, set the desired partitions in each axis by setting the `decomposition` parameter. For example, if the domain is 3D and the decomposition needs to be two partitions in x, one partition in y, and sixteen partitions in for z, the setting is

```json
"simulation_params": {
    "macro_domain_bounds": [0, 1, 0, 1, 0, 1],
    "decomposition": [2, 1, 16]
}
```

For a 2D domain, only two values need to be set for `decomposition`. The total number of partitions provided in the `decomposition` should be the same as the number of processors provided in the `mpirun`/`mpiexec` command.

## Adaptivity

See the [adaptivity](tooling-micro-manager-adaptivity.html) documentation for a detailed explanation about the algorithm and variants.

To turn on adaptivity, set `"adaptivity": true` in `simulation_params`. Then under `adaptivity_settings` set the following variables:

Parameter | Description
--- | ---
`type` | Set to either `local` or `global`. The type of adaptivity matters when the Micro Manager is run in parallel. `local` means comparing micro simulations within a local partitioned domain for similarity. `global` means comparing micro simulations from all partitions, so over the entire domain.
`data` | List of names of data which are to be used to calculate if micro-simulations are similar or not. For example `["temperature", "porosity"]`.
`adaptivity_every_n_time_windows` | Frequency of adaptivity computation (integer which is number of time windows).
`output_type` | Set to either `local`, `global`, or `all`. `local` outputs rank-wise adaptivity metrics. `global` outputs global averaged metrics. `all` outputs both local and global metrics.
`output_n` | Frequency of output of adaptivity metrics (integer which is number of time windows).
`history_param` | History parameter $$ \Lambda $$, set as $$ \Lambda >= 0 $$.
`coarsening_constant` | Coarsening constant $$ C_c $$, set as $$ 0 =< C_c < 1 $$.
`refining_constant` | Refining constant $$ C_r $$, set as $$ 0 =< C_r < 1 $$.
`every_implicit_iteration` | If `true`, adaptivity is calculated in every implicit iteration. <br> If False, adaptivity is calculated once at the start of the time window and then reused in every implicit time iteration. Default `false`.
`similarity_measure`| Similarity measure to be used for adaptivity. Can be either `L1`, `L2`, `L1rel` or `L2rel`. By default, `L1` is used. The `rel` variants calculate the respective relative norms. This parameter is *optional*. Default `L1`.
`lazy_initialization` | Set to `true` to lazily create and initialize micro simulations. If selected, micro simulation objects are created only when the micro simulation is activated for the first time. Default: `false`.
`load_balancing` | Set to `true` to dynamically balance simulations for parallel runs. Default `false`. See [load balancing settings](#load-balancing) below.

Example of adaptivity configuration is

```json
"simulation_params": {
    "macro_domain_bounds": [0, 1, 0, 1, 0, 1],
    "adaptivity": true,
    "adaptivity_settings" {
        "type": "local",
        "data": ["temperature", "porosity"],
        "history_param": 0.5,
        "coarsening_constant": 0.3,
        "refining_constant": 0.4,
        "every_implicit_iteration": true,
        "lazy_initialization": true
    }
}
```

## Model Adaptivity

See the [model adaptivity](tooling-micro-manager-model-adaptivity.html) documentation for a detailed explanation about the interface.

To turn on model adaptivity, set `"model_adaptivity": true` in `simulation_params`. Then under `model_adaptivity_settings` set the following variables:

Parameter | Description
--- | ---
`micro_file_names` | List of paths to the files containing the Python importable micro simulation classes. If the files are not in the working directory, give the relative path from the directory where the Micro Manager is executed. Requires a minimum of 2 files.
`switching_function` | Path to the file containing the Python importable switching function. If the file is not in the working directory, give the relative path from the directory where the Micro Manager is executed.

Example of model adaptivity configuration is

```json
"simulation_params": {
    "micro_dt": 1.0,
    "macro_domain_bounds": [0.0, 25.0, 0.0, 25.0, 0.0, 25.0],
    "model_adaptivity": true,
    "model_adaptivity_settings": {
        "micro_file_names": ["python-dummy/micro_dummy", "python-dummy/micro_dummy", "python-dummy/micro_dummy"],
        "switching_function": "mada_switcher"
    }
}
```

### Adding adaptivity in the preCICE XML configuration

If adaptivity is used, the Micro Manager will attempt to write two scalar data per micro simulation to preCICE, called `active_state` and `active_steps`.

Parameter | Description
--- | ---
`active_state` | `1` if the micro simulation is active in the time window, and `0` if inactive.
`active_steps` | Summation of `active_state` up to the current time window.

The Micro Manager uses the output functionality of preCICE, hence these data sets need to be manually added to the preCICE configuration file. In the mesh and the participant Micro-Manager add the following lines:

```xml
<data:scalar name="active_state"/>
<data:scalar name="active_steps"/>

<mesh name="macro-mesh">
    <use-data name="active_state"/>
    <use-data name="active_steps"/>
</mesh>

<participant name="Micro-Manager">
    <write-data name="active_state" mesh="macro-mesh"/>
    <write-data name="active_steps" mesh="macro-mesh"/>
</participant>
```

## Load balancing

Under `load_balancing_settings`, the following parameters can be set

Parameter | Description
--- | ---
`every_n_time_windows` | Frequency of balancing the simulations. Default `1` (every time window).
`balancing_threshold` | Integer threshold value. Default `0`.
`balance_inactive_sims` | If `true`, inactive simulations associated to redistributed active simulations are moved to the new ranks of the active simulations. See [balance inactive simulations](tooling-micro-manager-adaptivity.html#balance-inactive-simulations). Default `false`.

## Interpolate a crashed micro simulation

If the optional dependency `sklearn` is installed, the Micro Manager will derive the output of a crashed micro simulation by interpolating outputs from similar simulations. To enable this, set
`"interpolate_crash": true` in the `simulation_params` section of the configuration file.

For more details on the interpolation see the [crash handling documentation](tooling-micro-manager-running.html#what-happens-when-a-micro-simulation-crashes).

## Next step

After creating a configuration file you are ready to [run the Micro Manager](tooling-micro-manager-running.html).
