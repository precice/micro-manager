---
title: Convert Your Micro Simulation to Library
permalink: micro-manager-code-changes.html
keywords: tooling, macro-micro, two-scale
summary: You need to create an Python-importable class from your micro simulation code.
---

## Steps to convert micro simulation code to a callable library

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
    c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) micro_cpp_dummy.cpp -o micro_dummy$(python3-config --extension-suffix)
    ```

    This will create a shared library `micro_dummy.so` which can be directly imported in Python.
    For more information on compiling C++ libraries, see the [pybind11 documentation](https://pybind11.readthedocs.io/en/stable/compiling.html).
