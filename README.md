# Micro Manager

A manager tool to enable two-scale macro-micro coupling using the coupling library preCICE.

## Installing the package

### Option 1: Using pip

It is recommended to install [micro-manager from PyPI]() by running

```bash
pip3 install --user micro-manager
```

If the dependencies are not installed, then `pip` will attempt to install them for you. If you encounter problems in the direct installation, see the [dependencies section](https://github.com/precice/micro-manager#required-dependencies) below for links to installation procedures of all dependencies.

### Option 2: Clone this repository and use pip

#### Required dependencies

Ensure that the following dependencies are installed:

* [preCICE](https://github.com/precice/precice/wiki)
* python3 (this adapter **only supports python3**)
* [python language bindings for preCICE](https://github.com/precice/python-bindings)
* [numpy](https://numpy.org/install/)
* [mpi4py](https://mpi4py.readthedocs.io/en/stable/install.html)

#### Build and install the manager

After cloning this repository, step into the project directory `micro-manager` and run `pip3 install --user .`.

## Using the micro-manager

The micro-manager facilitates two-scale coupling between a macro-scale simulation and many micro-scale simulations. The manager creates and controls a set of micro problems and couples them to one macro simulation using preCICE. 

To this end, an existing micro-simulation script needs to be converted into a Python library with a specific structure so that the micro manager can create and steer the micro simulations. Such a micro-simulation library needs to have a central class to represent one micro simulation. The micro manager then creates many instances of this class, one for each micro simulation, and steers them till the end of the coupled simulation.

The macro-problem script, on the other hand, is coupled to preCICE directly (just as it would be volume-coupled via preCICE to any other code). The section [couple your code](https://precice.org/couple-your-code-overview.html) of the preCICE documentation gives more details.

### Steps to convert micro-simulation code to a callable library

* Create a class called `MicroSimulation` that consists of all functions of the micro simulation. It is good practice to define class member variables in the class constructor `__init__`.
* **Optional**: Define a function `initialize` which computes the initial state of the micro simulation and returns initial values, which need to be transferred to the macro simulation. The return entity needs to be a Python dictionary with the names of the quantities as keys and the values of the quantities as values.
* Create a function `solve`, which consists of all solving steps of one time step of a micro simulation or, if the micro problem is a steady-state simulation, all solving steps until the steady state is reached. The `solve` function should return the quantities that need to be communicated to the macro-side. The return entity needs to be again a Python dictionary with the names of the quantities as keys and the values of the quantities as values.
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



### Configuration

The micro manager is configured at runtime using a JSON file `micro-manager-config.json`. The configuration file for example in `/examples/macro-micro-dummy`:

```json
{
    "micro_file": "micro_problem",
    "coupling_params": {
        "participant": "Micro-Manager",
        "precice_config": "precice-config.xml",
        "macro_mesh": "Macro-Mesh",
        "read_data": {"Macro-Scalar-Data": "scalar", "Macro-Vector-Data": "vector"},
        "write_data": {"Micro-Scalar-Data": "scalar", "Micro-Vector-Data": "vector"}
    },
    "simulation_params": {
        "macro_domain_bounds": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        "total_time": 10.0,
        "timestep": 1.0,
        "output_interval": 1.0
    }
}
```

The following quantities need to be configured:

* `micro_file`: Path to the micro-simulation script without ending `.py`.
* `coupling_params`:
  * `participant`: Name of the micro manager as stated in the preCICE configuration.
  * `precice_config`: Path to the preCICE XML configuration file.
  * `macro_mesh`: Name of the macro mesh as stated in the preCICE configuration.
  * `read_data`: A Python dictionary with the names of the data to be read from preCICE as keys and `"scalar"` or `"vector"`  as values.
  * `write_data`: A Python dictionary with the names of the data to be written to preCICE as keys and `"scalar"` or `"vector"`  as values.
* `simulation_params`:
  * `macro_domain_bounds`: Minimum and maximum limits of the macro-domain, having the format `[xmin, xmax, ymin, ymax, zmin, zmax]`.
  * `total_time`: Total simulation time.
  * `timestep`: Initial timestep of the simulation.
  * `output_interval`: Time interval at which the micro-manager outputs data.
  
### How to run

TODO
