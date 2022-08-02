# Micro Manager

A manager tool to facilitate solving two-scale (macro-micro) coupled problems using the coupling library [preCICE](https://github.com/precice/precice).

## Installing the manager

### Option 1: Using pip

It is recommended to install [micro-manager from PyPI]() by running

```bash
pip install --user micro-manager
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

#### Build and install the manager using pip

After cloning this repository, go to the project directory `micro-manager` and run `pip3 install --user .`.

#### Build and install the manager using Python

After cloning this repository, go to the project directory `micro-manager` and run `python setup.py install --user`.

## Using the micro-manager

The micro-manager facilitates two-scale coupling between one macro-scale simulation and many micro-scale simulations. The manager creates and controls a set of micro problems and couples them to one macro simulation, using preCICE. An existing micro-simulation code written in Python needs to be converted into a library with a specific class name and specific function names. The next section describes the required library structure of the micro-simulation code. On the other hand, the micro-problem is coupled to preCICE directly. The section [couple your code](https://precice.org/couple-your-code-overview.html) of the preCICE documentation gives more details on coupling existing codes.

### Steps to convert micro-simulation code to a callable library

* Create a class called `MicroSimulation`. It is good practice to define class member variables in the class constructor `__init__`.
* **Optional**: Define a function `initialize` which computes the initial state of the micro simulation and returns initial values, which need to be transferred to the macro simulation. The return entity needs to be a Python dictionary with the names of the quantities as keys and the values of the quantities as values.
* Create a function `solve`, which consists of all solving steps of one time step of a micro simulation or, if the micro problem is a steady-state simulation, all solving steps until the steady state is reached. `solve` should take a Python dictionary as an input, which has the name of the input data as keys and the corresponding data values as values. The `solve` function should return the quantities that need to be communicated to the macro-side. The return entity needs to again be a Python dictionary with the names of the quantities as keys and the values of the quantities as values.
* If implicit coupling is required between the macro and all micro problems, then you can additionally define two functions `save_checkpoint` and `revert_to_checkpoint`.
  * `save_checkpoint` should save the current state of the micro problem.
  * `revert_to_checkpoint` should revert to the saved state (required if the coupling loop does not convergence after a time step).

An example of an adapted micro problem as used in `/examples/macro-micro-dummy`:

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
```

### Configuring the micro manager

The micro manager is configured at runtime using a JSON file `micro-manager-config.json`. The configuration file for example in `/examples/macro-micro-dummy`:

```json
{
    "micro_file": "micro_problem",
    "coupling_params": {
        "precice_config": "precice-config.xml",
        "macro_mesh": "Macro-Mesh",
        "read_data_names": {"Macro-Scalar-Data": "scalar", "Macro-Vector-Data": "vector"},
        "write_data_names": {"Micro-Scalar-Data": "scalar", "Micro-Vector-Data": "vector"}
    },
    "simulation_params": {
        "macro_domain_bounds": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
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
  
### Running the micro manager

The micro manager is run directly from the terminal by providing the configuration file as an input argument in the following way:

```bash
micro_manager micro-manager-config.json
```

Alternatively the manager can also be run by creating a Python script which imports the micro manager package and calls its run function. For example a run script `run-micro-manager.py` would look like:

```python
from micro_manager import MicroManager

manager = MicroManager("micro-manager-config.json")

manager.run()
```

The script is then run:

```bash
python run-micro-manager.py
```

The micro manager can also be run in parallel, using the same script as stated above:

```bash
mpirun -n <number-of-procs> python3 run-micro-manager.py
```

### Advanced configuration options

In addition to the above mentioned configuration options, the manager offers more options for diagnostics output.

If the user wants to output the clock time required to solve each micro simulation, They can add the following keyword to the configuration:

```json
"diagnostics": {
  "output_micro_sim_solve_time": "True"
}
```

Additionally if the micro simulation code has a function called `output`, the manager will try to call it in order to generate output of all micro simulations. In this situation, the manager can be configured to output at a particular interval. This configuration is done as follows:

```json
"simulation_params": {
  "micro_output_n": 10
}
```

Here, the manager will write output of micro simulations every 10 time steps. If the entity `micro_output_n` is not defined, then the manager will output the micro simulation output in every time step.

### Creating a preCICE configuration file for a macro-micro problem

In addition to configuring the Micro Manager, preCICE itself also needs to be configured via a [XML configuration file](https://precice.org/configuration-overview.html).
The user is expected to configure preCICE with the correct names of the data being exchanged between the macro and micro side. An example of such a macro-micro configuration for preCICE can be found in [this two-scale heat conduction example](https://github.com/IshaanDesai/coupled-heat-conduction).