---
title: Convert Your Micro Simulation to Library
permalink: tooling-micro-manager-micro-simulation-callable-library.html
keywords: tooling, macro-micro, two-scale
summary: You need to create an Python-importable class from your micro simulation code.
---

## Steps to convert micro simulation code to a callable library

The Micro Manager requires a specific structure of the micro simulation code. Create a class that can be called from Python with the structure given below. The docstring of each function gives information on what it should do and what its input and output should be.

```python
class MicroSimulation: # Name is fixed

    def __init__(self):
        """
        Constructor of class MicroSimulation. Initialize all class member variables here.
        """

    def initialize(self) -> dict:
        """
        Initialize the micro simulation. This function is *optional*.

        Returns
        -------
        data : dict
            Python dictionary with keys as names of micro data and values as the data at the initial condition
        """

    def solve(self, macro_data, dt) -> dict:
        """
        Solve one time step of the micro simulation or for steady-state problems: solve until steady state is reached.

        Parameters
        ----------
        macro_data : dict
            Dictionary with keys as names of macro data and values as the data
        dt : float
            Time step size

        Returns
        -------
        micro_data : dict
            Dictionary with keys as names of micro data and values as the updated micro data
        """

    def save_checkpoint(self):
        """
        Save the state of the micro simulation. *Required for implicit coupling*.
        Save the state internally.
        """

    def reload_checkpoint(self):
        """
        Revert the micro simulation to a previously saved state. *Required for implicit coupling*.
        """

    def output(self):
        """
        This function writes output of the micro simulation in some form.
        It will be called with frequency set by configuration option `simulation_params: micro_output_n`
        This function is *optional*.
        """
```

Skeleton dummy code of a sample MicroSimulation class can be found in the [examples/](https://github.com/precice/micro-manager/tree/main/examples/) directory. There are two variants

* `examples/python-dummy/`: Dummy micro simulation written in Python
* `examples/cpp-dummy/`: Dummy micro simulation written in C++ and compiled to a Python library using [pybind11](https://pybind11.readthedocs.io/en/stable/)

### Convert your micro simulation written in C++ to a callable library

A C++ dummy micro simulation is provided in [`examples/cpp-dummy/`](github.com/precice/micro-manager/tree/main/examples/cpp-dummy).
It uses [pybind11](https://pybind11.readthedocs.io/en/stable/) to compile the C++ code into a library which can be imported in Python. If the micro simulation in C++, [install pybind11](https://pybind11.readthedocs.io/en/stable/installing.html).

Creating a new micro simulation in C++ has the following steps

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

## Next Steps

With your code converted to a library, you can now [create a coupling configuration](tooling-micro-manager-usage-configuration.html).
