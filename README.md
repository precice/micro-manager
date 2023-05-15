---
title: The Micro Manager
permalink: tooling-micro-manager.html
redirect_from: adapter-openfoam.html
keywords: tooling, micro-manager, two-scale, macro-micro
summary: The Micro Manager is a tool to facilitate solving two-scale (macro-micro) coupled problems using the coupling library preCICE.
---


# The Micro Manager

The Micro Manager is a tool to facilitate solving two-scale (macro-micro) coupled problems using the coupling library [preCICE](https://www.precice.org/).

## Installation

The Micro Manager is a Python package that can be installed using `pip` or manually. Make sure [preCICE](installation-overview.html) is installed before installing the Micro Manager. The Micro Manager is tested with preCICE version [2.5.0](https://github.com/precice/precice/releases/tag/v2.5.0).

### Option 1: Using pip

It is recommended to install [micro-manager-precice from PyPI](https://pypi.org/project/micro-manager-precice/) by running

```bash
pip install --user micro-manager-precice
```

Installing preCICE is mandatory, other dependencies will be installed by `pip` if they are not already installed. If you encounter problems in the direct installation, see the [dependencies section](#required-dependencies) below.

### Option 2: Clone this repository and install manually

#### Required dependencies

Ensure that the following dependencies are installed:

* Python 3 or higher
* [preCICE](installation-overview.html) Version [2.5.0](https://github.com/precice/precice/releases/tag/v2.5.0)
* [pyprecice: Python language bindings for preCICE](installation-bindings-python.html)
* [numpy](https://numpy.org/install/)
* [mpi4py](https://mpi4py.readthedocs.io/en/stable/install.html)

#### Build and install the Manager using pip

After cloning this repository, go to the project directory `micro-manager` and run

```bash
pip install --user .
```

#### Build and install the Manager using Python

After cloning this repository, go to the project directory `micro-manager` and run

```bash
python setup.py install --user
```

## Using the Micro Manager

The Micro Manager facilitates two-scale coupling between one macro-scale and many micro-scale simulations. It creates instances of several micro simulations and couples them to one macro simulation, using preCICE.

An existing micro simulation code written in Python needs to be converted into a library with a specific class name and specific function names. The next section describes the required library structure of the micro simulation code. On the other hand, the micro-problem is coupled to preCICE directly. The section [couple your code](couple-your-code-overview.html) of the preCICE documentation gives more details on coupling existing codes.

### Steps to convert micro simulation code to a callable library

The Micro Manager requires a specific structure of the micro simulation code. Create a class that can be called from Python with the following structure:

```python
class MicroSimulation: # Name is fixed

    def __init__(self): # No input arguments
        # Initialize class member variables

    def initialize(self) -> dict:
        # *Optional*
        # Compute initial state of the micro simulation and return initial values.
        # Return values have to be a dictionary of shape {"data-name":<value>,...}

    def solve(self, macro_data, dt) -> dict:
        # Solve one time step of the micro simulation or for steady-state problems: solve until steady state is reached
        # `macro_data` is a dictionary with macro quantity names as keys and data as values
        # Return values need to be communicated to the macro simulation in a analogously shaped dictionary

    def save_checkpoint(self):
        # *Required for implicit coupling*
        # Save current state of the micro simulation in an internal variable

    def reload_checkpoint(self):
        # *Required for implicit coupling*
        # Revert to the saved state of the micro simulation

    def output(self):
        # *Optional*
        # Write micro simulation output, e.g. export to vtk
        # Will be called with frequency set by configuration option `simulation_params: micro_output_n`
```

Examples of MicroSimulation classes can be found in the `examples/` directory. Currently the following [examples](https://github.com/precice/micro-manager/tree/main/examples/) are available:

* `examples/python-dummy/`: Dummy micro simulation written in Python
* `examples/cpp-dummy/`: Dummy micro simulation written in C++ and compiled to a Python library using [pybind11](https://pybind11.readthedocs.io/en/stable/)

### Configuring the Micro Manager

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

#### Coupling Parameters

Parameter | Description
--- | ---
`config_file_name` |  Path to the preCICE XML configuration file.
`macro_mesh_name` |  Name of the macro mesh as stated in the preCICE configuration.
`read_data_names` |  A Python dictionary with the names of the data to be read from preCICE as keys and `"scalar"` or `"vector"`  as values.
`write_data_names` |  A Python dictionary with the names of the data to be written to preCICE as keys and `"scalar"` or `"vector"`  as values.

#### Simulation Parameters

Parameter | Description
--- | ---
`macro_domain_bounds`| Minimum and maximum limits of the macro-domain, having the format `[xmin, xmax, ymin, ymax, zmin, zmax]`
*optional:* `micro_output_n`|  Frequency of calling the output functionality of the micro simulation in terms of number of time steps. If not given, `micro_sim.output()` is called every time step
*optional:* Adaptivity parameters | See section on [Adaptivity](#adaptivity). By default, adaptivity is disabled.

#### *Optional*: Diagnostics

Parameter | Description
--- | ---
`data_from_micro_sims` | A Python dictionary with the names of the data from the micro simulation to be written to VTK files as keys and `"scalar"` or `"vector"` as values. This relies on the [export functionality](configuration-export.html#enabling-exporters) of preCICE and requires the corresponding export tag to be set in the preCICE XML configuration script.
`output_micro_sim_solve_time` | If `True`, the Manager writes the wall clock time of the `solve()` function of each micro simulation to the VTK output.

An example configuration file can be found in [`examples/micro-manager-config.json`](https://github.com/precice/micro-manager/tree/main/examples/micro-manager-config.json).

#### Adaptivity

The Micro Manager can adaptively initialize micro simulations. The following adaptivity strategies are implemented:

1. Redeker, Magnus & Eck, Christof. (2013). A fast and accurate adaptive solution strategy for two-scale models with continuous inter-scale dependencies. Journal of Computational Physics. 240. 268-283. [10.1016/j.jcp.2012.12.025](https://doi.org/10.1016/j.jcp.2012.12.025).

2. Bastidas, Manuela & Bringedal, Carina & Pop, Iuliu. (2021). A two-scale iterative scheme for a phase-field model for precipitation and dissolution in porous media. Applied Mathematics and Computation. 396. 125933. [10.1016/j.amc.2020.125933](https://doi.org/10.1016/j.amc.2020.125933).

To turn on adaptivity, the following options need to be set in `simulation_params`:

Parameter | Description
--- | ---
`adaptivity` | Set as `True` to turn on adaptivity.
`adaptivity_data` | List of names of data which are to be used to calculate if two micro-simulations are similar or not. For example `["macro-scalar-data", "macro-vector-data"]`
`adaptivity_history_param` | History parameter \$\Lambda\$, set as \$\Lambda >= 0\$.
`adaptivity_coarsening_constant` | Coarsening constant \$C_c\$, set as \$C_c < 1\$.
`adaptivity_refining_constant` | Refining constant \$C_r\$, set as \$C_r >= 0\$.
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

### Running the Micro Manager

The Micro Manager is run directly from the terminal by providing the configuration file as an input argument in the following way:

```bash
micro_manager micro-manager-config.json
```

Alternatively the Manager can also be run by creating a Python script which imports the Micro Manager package and calls its run function. For example a run script `run-micro-manager.py` would look like:

```python
from micro_manager import MicroManager

manager = MicroManager("micro-manager-config.json")

manager.initialize()

manager.solve()
```

The Micro Manager can also be run in parallel, using the same script as stated above:

```bash
mpirun -n <number-of-procs> python3 run-micro-manager.py
```

### Create your own micro simulation in C++

A C++ dummy micro simulation is provided in [`examples/cpp-dummy/`](github.com/precice/micro-manager/tree/main/examples/cpp-dummy).
It uses [pybind11](https://pybind11.readthedocs.io/en/stable/) to compile a C++ library which can be imported in Python. To install pybind11, follow the instructions [here](https://pybind11.readthedocs.io/en/stable/installing.html).

Creating a new micro simulation in C++ requires the following steps.

1. Create a C++ class which implements the functions given [above](#steps-to-convert-micro-simulation-code-to-a-callable-library).
The `solve()` function should have the following signature:

    ```cpp
    py::dict MicroSimulation::solve(py::dict macro_data, double dt)
    ```

    [`py::dict`](https://pybind11.readthedocs.io/en/stable/advanced/pycpp/object.html?#instantiating-compound-python-types-from-c) is a Python dictionary which can be used to pass data between Python and C++. You need to cast the data to the correct type before using it in C++ and vice versa. An example is given in the dummy micro simulation.
2. Export the C++ class to Python using pybind11. Follow the instructions to exporting classes in the [pybind11 documentation](https://pybind11.readthedocs.io/en/stable/classes.html) or read their [first steps](https://pybind11.readthedocs.io/en/stable/basics.html) to get started.
3. Compile the C++ library including pybind11. For the solverdummy, run

    ```bash
    c++ -O3 -Wall -shared -std=c++11 -fPIC \$(python3 -m pybind11 --includes) micro_cpp_dummy.cpp -o micro_dummy\$(python3-config --extension-suffix)
    ```

    This will create a shared library `micro_dummy.so` which can be directly imported in Python.
    For more information on compiling C++ libraries, see the [pybind11 documentation](https://pybind11.readthedocs.io/en/stable/compiling.html).
