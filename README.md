# Micro Manager

A Manager tool to facilitate solving two-scale (macro-micro) coupled problems using the coupling library [preCICE](https://github.com/precice/precice).

## Installing the Micro Manager

### Option 1: Using pip

It is recommended to install [micro-manager-precice from PyPI](https://pypi.org/project/micro-manager-precice/) by running

```bash
pip install --user micro-manager-precice
```

If the dependencies are not installed, then `pip` will attempt to install them for you. If you encounter problems in the direct installation, see the [dependencies section](https://github.com/precice/micro-manager#required-dependencies) below for links to installation procedures of all dependencies.

### Option 2: Clone this repository and install manually

#### Required dependencies

Ensure that the following dependencies are installed:

* Python 3 or higher
* [preCICE](https://github.com/precice/precice/wiki)
* [pyprecice: Python language bindings for preCICE](https://github.com/precice/python-bindings)
* [numpy](https://numpy.org/install/)
* [mpi4py](https://mpi4py.readthedocs.io/en/stable/install.html)

#### Build and install the Manager using pip

After cloning this repository, go to the project directory `micro-manager` and run `pip3 install --user .`.

#### Build and install the Manager using Python

After cloning this repository, go to the project directory `micro-manager` and run `python setup.py install --user`.

## Using the Micro Manager

The Micro Manager facilitates two-scale coupling between one macro-scale simulation and many micro-scale simulations. It creates instances of several micro simulations and couples them to one macro simulation, using preCICE. An existing micro simulation code written in Python needs to be converted into a library with a specific class name and specific function names. The next section describes the required library structure of the micro simulation code. On the other hand, the micro-problem is coupled to preCICE directly. The section [couple your code](https://precice.org/couple-your-code-overview.html) of the preCICE documentation gives more details on coupling existing codes.

### Steps to convert micro simulation code to a callable library

* Create a class called `MicroSimulation`. It is good practice to define class member variables in the class constructor `__init__`. This constructor does not get any input.
* **Optional**: Define a function `initialize` which computes the initial state of the micro simulation and returns initial values, which need to be transferred to the macro simulation. The return value needs to be a Python dictionary with the names of the quantities as keys and the values of the quantities as the dictionary values.
* Create a function `solve`, which consists of all solving steps of one time step of a micro simulation or, if the micro problem is a steady-state simulation, all solving steps until the steady state is reached. `solve` should take a Python dictionary as an input, which has the name of the input data as keys and the corresponding data values as values. The `solve` function should return the quantities that need to be communicated to the macro-side. The return entity needs to again be a Python dictionary with the names of the quantities as keys and the values of the quantities as values.
* If implicit coupling is required between the macro and all micro problems, then you can additionally define two functions `save_checkpoint` and `revert_to_checkpoint`.
  * `save_checkpoint` should save the current state of the micro problem.
  * `revert_to_checkpoint` should revert to the saved state (required if the coupling loop does not convergence after a time step).
* **Optional**: Define a function `output` which writes the micro simulation output. The micro Manager will call this function with the frequency set by the configuration option `simulation_params: micro_output_n`.

An example of a MicroSimulation class as used in `/examples/macro-micro-dummy`:

```python
class MicroSimulation:

    def __init__(self):
        """
        Constructor of MicroSimulation class.
        """
        self._dims = 3
        self._micro_scalar_data = None
        self._micro_vector_data = None
        self._checkpoint = None

    def initialize(self):
        self._micro_scalar_data = 0
        self._micro_vector_data = []
        self._checkpoint = 0

    def solve(self, macro_data, dt):
        assert dt != 0
        self._micro_vector_data = []
        self._micro_scalar_data = macro_data["macro-scalar-data"]
        for d in range(self._dims):
            self._micro_vector_data.append(macro_data["macro-vector-data"][d])

        return {"micro-scalar-data": self._micro_scalar_data.copy(),
                "micro-vector-data": self._micro_vector_data.copy()}

    def save_checkpoint(self):
        print("Saving state of micro problem")
        self._checkpoint = self._micro_scalar_data

    def reload_checkpoint(self):
        print("Reverting to old state of micro problem")
        self._micro_scalar_data = self._checkpoint

    def output(self):
        print("Writing VTK output of micro problem")
        self._write_vtk()
```

### Configuring the Micro Manager

The Micro Manager is configured at runtime using a JSON file `micro-manager-config.json`. An example configuration file is:

```json
{
    "micro_file_name": "micro_heat",
    "coupling_params": {
            "participant_name": "Micro-Manager",
            "config_file_name": "precice-config.xml",
            "macro_mesh_name": "macro-mesh",
            "write_data_names": {"k_00": "scalar", "k_11": "scalar", "porosity": "scalar"},
            "read_data_names": {"concentration": "scalar"}
    },
    "simulation_params": {
      "macro_domain_bounds": [0.0, 1.0, 0.0, 0.5],
    },
    "diagnostics": {
      "micro_output_n": 5,
      "data_from_micro_sims": {"grain_size": "scalar"},
      "output_micro_sim_solve_time": "True"
    }
}
```

The following quantities need to be configured:

* `micro_file`: Path to the micro-simulation script. **Do not add the file extension** `.py`.
* `coupling_params`:
  * `precice_config`: Path to the preCICE XML configuration file.
  * `macro_mesh`: Name of the macro mesh as stated in the preCICE configuration.
  * `read_data_names`: A Python dictionary with the names of the data to be read from preCICE as keys and `"scalar"` or `"vector"`  as values.
  * `write_data_names`: A Python dictionary with the names of the data to be written to preCICE as keys and `"scalar"` or `"vector"`  as values.
* `simulation_params`:
  * `macro_domain_bounds`: Minimum and maximum limits of the macro-domain, having the format `[xmin, xmax, ymin, ymax, zmin, zmax]`.

The Micro Manager is capable of generating diagnostics type output of the micro simulations, which is critical in the development phase of two-scale simulations. The following configuration options are available:

* `diagnostics`:
  * `micro_output_n`: Frequency of calling the output functionality of the micro simulation in terms of number of time steps. If this quantity is configured the Micro Manager will attempt to call the `output()` function of the micro simulation.
  * `data_from_micro_sims`: A Python dictionary with the names of the data from the micro simulation to be written to VTK files as keys and `"scalar"` or `"vector"`  as values.
  * `output_micro_sim_solve_time`: When `True`, the Manager writes the wall clock time of the `solve()` function of each micro simulation to the VTK output.

The Micro Manager can adaptively initialize micro simulations. The adaptivity strategy is taken from two publications:

1. Redeker, Magnus & Eck, Christof. (2013). A fast and accurate adaptive solution strategy for two-scale models with continuous inter-scale dependencies. Journal of Computational Physics. 240. 268-283. 10.1016/j.jcp.2012.12.025.

2. Bastidas, Manuela & Bringedal, Carina & Pop, Iuliu. (2021). A two-scale iterative scheme for a phase-field model for precipitation and dissolution in porous media. Applied Mathematics and Computation. 396. 125933. 10.1016/j.amc.2020.125933.

To turn on adaptivity, the following options need to be set in `simulation_params`:

* `adaptivity`: Set as `True`.
* `adaptivity_data`: List of names of data which are to be used to calculate if two micro-simulations are similar or not. For example `["macro-scalar-data", "macro-vector-data"]`
* `adaptivity_history_param`: History parameter $\Lambda$, set as $\Lambda >= 0$.
* `adaptivity_coarsening_constant`: Coarsening constant $C_c$, set as $C_c < 1$.
* `adaptivity_refining_constant`: Refining constant $C_r$, set as $C_r >= 0$.
* `adaptivity_every_implicit_iteration`: Set as `True` if adaptivity calculation is to be done in every implicit iteration. Setting `False` would lead to adaptivity being calculated once at the start of the time window and then reused in every implicit time iteration.

All variables names are chosen to be same as the second publication mentioned above.

#### Changes to preCICE configuration file

The Micro Manager relies on the [export functionality](https://precice.org/configuration-export.html#enabling-exporters) of preCICE to write diagnostics data output.

* If the option `diagnotics: data_from_micro_sims` is configured, the corresponding export tag also needs to be set in the preCICE XML configuration script.
* If adaptivity is turned on, the Micro Manager will attempt to write a scalar data set `active_state` to preCICE. Add this data set to the preCICE configuration file.

### Running the Micro Manager

The Micro Manager is run directly from the terminal by providing the configuration file as an input argument in the following way:

```bash
micro_manager micro-manager-config.json
```

Alternatively the Manager can also be run by creating a Python script which imports the Micro Manager package and calls its run function. For example a run script `run-micro-manager.py` would look like:

```python
from micro_manager import MicroManager

manager = MicroManager("micro-manager-config.json")

manager.run()
```

The script is then run:

```bash
python run-micro-manager.py
```

The Micro Manager can also be run in parallel, using the same script as stated above:

```bash
mpirun -n <number-of-procs> python3 run-micro-manager.py
```

### Advanced configuration options

In addition to the above mentioned configuration options, the Manager offers more options for diagnostics output.

If the user wants to output the clock time required to solve each micro simulation, They can add the following keyword to the configuration:

```json
"diagnostics": {
  "output_micro_sim_solve_time": "True"
}
```

Additionally if the micro simulation code has a function called `output`, the Manager will try to call it in order to generate output of all micro simulations. In this situation, the Manager can be configured to output at a particular interval. This configuration is done as follows:

```json
"diagnostics": {
  "micro_output_n": 10
}
```

Here, the Manager will write output of micro simulations every 10 time steps. If the entity `micro_output_n` is not defined, then the Manager will output the micro simulation output in every time step.

### Creating a preCICE configuration file for a macro-micro problem

In addition to configuring the Micro Manager, preCICE itself also needs to be configured via a [XML configuration file](https://precice.org/configuration-overview.html).
The user is expected to configure preCICE with the correct names of the data being exchanged between the macro and micro side. An example of such a macro-micro configuration for preCICE can be found in [this two-scale heat conduction example](https://github.com/IshaanDesai/coupled-heat-conduction).
