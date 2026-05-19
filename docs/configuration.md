---
title: Configure the Micro Manager
permalink: tooling-micro-manager-configuration.html
keywords: tooling, macro-micro, two-scale
summary: Provide a JSON file to configure the Micro Manager.
---

{% note %} In the preCICE XML configuration the Micro Manager is a participant with the name `Micro-Manager`. {% endnote %}

The Micro Manager is configured with a JSON file. Several parameters can be set.

## Micro Manager Configuration

| Parameter                  | Description                                                                                                                                                                                           | Default       |
|----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|
| `micro_file_name`          | Path to the file containing the Python importable micro simulation class. If the file is not in the working directory, give the relative path from the directory where the Micro Manager is executed. | -             |
| `micro_stateless`          | Boolean if micro simulation is stateless allowing model instancing.                                                                                                                                   | False         |
| `output_directory`         | Path to output directory for logging and performance metrics. Directory is created if not existing already.                                                                                           | `.`           |
| `memory_usage_output_type` | Set to either `local`, `global`, or `all`. `local` outputs rank-wise peak memory usage. `global` outputs global averaged peak memory usage. `all` outputs both local and global levels.               | Empty string. |
| `memory_usage_output_n`    | Interval of output.                                                                                                                                                                                   | 1             |

All output is to a CSV file with the peak memory usage (RSS) in every time window, in MBs.

Apart from the base settings, there are three main sections in the configuration file, [coupling parameters](#coupling-parameters), [simulation parameters](#simulation-parameters), and [diagnostics](#diagnostics).

## Coupling Parameters

| Parameter                  | Description                                                                    |
|----------------------------|--------------------------------------------------------------------------------|
| `precice_config_file_name` | Path to the preCICE XML configuration file from the current working directory. |
| `macro_mesh_name`          | Name of the macro mesh as stated in the preCICE configuration.                 |
| `read_data_names`          | List with the names of the data to be read from preCICE.                       |
| `write_data_names`         | List with the names of the data to be written to preCICE.                      |

## Simulation Parameters

| Parameter | Description | Default |
| --- | --- | --- |
| `macro_domain_bounds` | Minimum and maximum bounds of the macro-domain, having the format `[xmin, xmax, ymin, ymax, zmin, zmax]` in 3D and `[xmin, xmax, ymin, ymax]` in 2D. | - |
| `decomposition` | List of number of ranks in each axis with format `[xranks, yranks, zranks]` in 3D and `[xranks, yranks]` in 2D. | `[1, 1, 1]` or `[1, 1]` |
| `decomposition_type` | Type of domain decomposition. Either `uniform` or `nonuniform`. | `uniform` |
| `minimum_access_region_size` | If `nonuniform` decomposition, optionally set a minimum domain width in each axis. Format `[xmin, ymin, zmin]` | - |
| `micro_dt` | Initial time window size (dt) of the micro simulation. | - |
| `adaptivity` | Set `true` for simulations with adaptivity. See section on [adaptivity](#adaptivity). | `false` |
| `load_balancing` | Set `true` for load balancing. See section on [load balancing](#load-balancing). | `false` |

The total number of partitions ranks in the `decomposition` list should be the same as the number of ranks in the `mpirun` or `mpiexec` command.

Non-uniform domain decomposition is based on a geometric progression.

## Diagnostics

| Parameter              | Description                                                                                                                                | Default |
|------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|---------|
| `data_from_micro_sims` | Dictionary with the names of the data from the micro simulation to be written to VTK files as keys and `"scalar"` or `"vector"` as values. | -       |
| `micro_output_n`       | Frequency of calling the optional output functionality of the micro simulation in terms of number of time steps.                           | 1       |

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

## Adaptivity

See the [adaptivity](tooling-micro-manager-adaptivity.html) documentation for a detailed explanation about the algorithm and variants.

To turn on adaptivity, set `"adaptivity": true` in `simulation_params`. Then under `adaptivity_settings` set the following variables:

| Parameter                         | Description                                                                                                                                                                                                                                                                                         | Default       |
|-----------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|
| `type`                            | Set to either `local` or `global`. The type of adaptivity matters when the Micro Manager is run in parallel. `local` means comparing micro simulations within a local partitioned domain for similarity. `global` means comparing micro simulations from all partitions, so over the entire domain. | None          |
| `data`                            | List of names of data which are to be used to calculate if micro-simulations are similar or not. For example `["temperature", "porosity"]`.                                                                                                                                                         | -             |
| `adaptivity_every_n_time_windows` | Interval of adaptivity computation.                                                                                                                                                                                                                                                                 | 1             |
| `output_type`                     | Set to either `local`, `global`, or `all`. `local` outputs rank-wise adaptivity metrics. `global` outputs global averaged metrics. `all` outputs both local and global metrics.                                                                                                                     | Empty string. |
| `output_n`                        | Frequency of output of adaptivity metrics.                                                                                                                                                                                                                                                          | 1             |
| `history_param`                   | History parameter $$ \Lambda $$, set as $$ \Lambda >= 0 $$.                                                                                                                                                                                                                                         | 0.5           |
| `coarsening_constant`             | Coarsening constant $$ C_c $$, set as $$ 0 =< C_c < 1 $$.                                                                                                                                                                                                                                           | 0.5           |
| `refining_constant`               | Refining constant $$ C_r $$, set as $$ 0 =< C_r < 1 $$.                                                                                                                                                                                                                                             | 0.5           |
| `every_implicit_iteration`        | If `true`, adaptivity is calculated in every implicit iteration. <br> If False, adaptivity is calculated once at the start of the time window and then reused in every implicit time iteration.                                                                                                     | `false`       |
| `similarity_measure`              | Similarity measure to be used for adaptivity. Can be either `L1`, `L2`, `L1rel` or `L2rel`. By default, `L1` is used. The `rel` variants calculate the respective relative norms. This parameter is *optional*.                                                                                     | `L2rel`       |
| `lazy_initialization`             | Set to `true` to lazily create and initialize micro simulations. If selected, micro simulation objects are created only when the micro simulation is activated for the first time.                                                                                                                  | `false`       |
| `load_balancing`                  | Set to `true` to dynamically balance simulations for parallel runs. See [load balancing settings](#load-balancing) below.                                                                                                                                                                           | `false`       |

Example of adaptivity configuration is

```json
"simulation_params": {
    "adaptivity_settings" {
        "type": "local",
        "data": ["temperature", "porosity"],
        "adaptivity_every_n_time_windows": 5,
        "output_type": "all",
        "output_n": 5,
        "history_param": 0.5,
        "coarsening_constant": 0.3,
        "refining_constant": 0.4,
        "every_implicit_iteration": false,
        "lazy_initialization": true
    }
}
```

## Model Adaptivity

See the [model adaptivity](tooling-micro-manager-model-adaptivity.html) documentation for a detailed explanation about the interface.

To turn on model adaptivity, set `"model_adaptivity": true` in `simulation_params`. Then under `model_adaptivity_settings` set the following variables:

| Parameter            | Description                                                                                                                                                                                                                                        |
|----------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `micro_file_names`   | List of paths to the files containing the Python importable micro simulation classes, in order of decreasing model fidelity. If the files are not in the working directory, give the relative path from the directory where the Micro Manager is executed. Requires a minimum of 2 files. |
| `switching_function` | Path to the file containing the Python importable switching function. If the file is not in the working directory, give the relative path from the directory where the Micro Manager is executed.                                                  |
| `micro_stateless`    | List of boolean values, whether the respective micro simulation model is stateless and can use model instancing.                                                                                                                                   |

Example of model adaptivity configuration is

```json
"simulation_params": {
    "model_adaptivity": true,
    "model_adaptivity_settings": {
        "micro_file_names": ["python-dummy/micro_dummy", "python-dummy/micro_dummy", "python-dummy/micro_dummy"],
        "switching_function": "mada_switcher",
        "micro_stateless": [False, True, True]
    }
}
```

### Adding adaptivity in the preCICE XML configuration

If adaptivity is used, the Micro Manager will attempt to write two scalar data per micro simulation to preCICE, called `Active-State` and `Active-Steps`.

- `Active-State` is `1` if the micro simulation is active in the time window, and `0` if inactive.
- `Active-Steps` is summation of `Active-State` up to the current time window.

The Micro Manager uses the output functionality of preCICE, hence these data sets need to be manually added to the preCICE configuration file. In the mesh and the participant Micro-Manager add the following lines:

```xml
<data:scalar name="Active-State"/>
<data:scalar name="Active-Steps"/>

<mesh name="macro-mesh">
    <use-data name="Active-State"/>
    <use-data name="Active-Steps"/>
</mesh>

<participant name="Micro-Manager">
    <write-data name="Active-State" mesh="macro-mesh"/>
    <write-data name="Active-Steps" mesh="macro-mesh"/>
</participant>
```

## Load balancing

Load balancing can be activated by setting `load_balancing` to true.
It balances based on either the elapsed time required to solve the prior iteration `type="time""` or the number of active simulations `type=active`.
One Initial load balancing step is performed, prior to any computation (assuming equal workload for time based load balancing or the current active counts for `active` load balancing.).
Subsequently, in the following iteration another load balancing step is performed based. (This is mainly for the time based balancing to use the just acquired timings.)
Afterwards balancing is performed `every_n_time_windows`.
Upon activation, further configuration must be provided in `load_balancing_settings`.
The following parameters can be set

| Parameter               | Description                                      | Default  |
|-------------------------|--------------------------------------------------|----------|
| `every_n_time_windows`  | Frequency of balancing the simulations.          | `1`      |
| `partitioning`          | Partitioning Algorithm. Options: ["lpt"].        | `"lpt"`  |
| `type`                  | Load balancing type. Options: ["time", "active"] | `"time"` |
| `threshold`             | Threshold parameter                              | `0`      |
| `balance_inactive_sims` | Balance inactive simulations                     | `False`  |

```json
"simulation_params": {
    "load_balancing": true,
    "load_balancing_settings": {
        "every_n_time_windows": 5,
        "partitioning": "lpt",
        "type": "active",
        "threshold": 2,
        "balance_inactive_sims": true
    }
}
```

## Interpolate a crashed micro simulation

If the optional dependency `sklearn` is installed, the Micro Manager will derive the output of a crashed micro simulation by interpolating outputs from similar simulations. To enable this, set
`"interpolate_crash": true` in the `simulation_params` section of the configuration file.

For more details on the interpolation see the [crash handling documentation](tooling-micro-manager-running.html#what-happens-when-a-micro-simulation-crashes).

## Next step

After creating a configuration file you are ready to [run the Micro Manager](tooling-micro-manager-running.html).
