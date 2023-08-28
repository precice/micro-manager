---
title: Prepare micro simulation
permalink: tooling-micro-manager-prepare-micro-simulation.html
keywords: tooling, macro-micro, two-scale
toc: off
summary: Create an Python-importable class from your micro simulation code.
---

The Micro Manager requires that the micro simulation code be in a predefined class structure. As the Micro Manager is written in Python, micro simulation codes written in Python are the easiest to prepare. For micro simulation codes not written in Python, look at the [C++ micro simulation section](#create-an-python-importable-class-from-your-micro-simulation-code-written-in-c) below.

{% note %} The Micro Manager [solver dummy examples](https://github.com/precice/micro-manager/tree/develop/examples) are minimal code examples with the predefined class structure. We recommend copying the appropriate example and modifying it with your micro simulation code to create a Python-importable class. {% endnote %}

Restructure your micro simulation code into a Python class with the structure given below. The docstring of each function gives information on what it should do and what its input and output should be.

```python
class MicroSimulation: # Name is fixed
    def __init__(self):
        """
        Constructor of class MicroSimulation.
        """

    def initialize(self) -> dict:
        """
        Initialize the micro simulation. This function is *optional*.

        Returns
        -------
        data : dict
            Python dictionary with names of micro data as keys and the data as values at the initial condition
        """

    def solve(self, macro_data: dict, dt: float) -> dict:
        """
        Solve one time step of the micro simulation for transient problems or solve until steady state for steady-state problems.

        Parameters
        ----------
        macro_data : dict
            Dictionary with names of macro data as keys and the data as values.
        dt : float
            Current time step size.

        Returns
        -------
        micro_data : dict
            Dictionary with names of micro data as keys and the updated micro data a values.
        """

    def set_state(self, state):
        """
        Set the state of the micro simulation.
        """

    def get_state(self):
        """
        Return the state of the micro simulation.
        """

    def output(self):
        """
        This function writes output of the micro simulation in some form.
        It will be called with frequency set by configuration option `simulation_params: micro_output_n`
        This function is *optional*.
        """
```

A dummy code of a sample MicroSimulation class can be found in the [examples/python-dummy/micro_dummy.py](https://github.com/precice/micro-manager/blob/develop/examples/python-dummy/micro_dummy.py) directory.

## Create an Python-importable class from your micro simulation code written in C++

A dummy C++ dummy micro simulation code having a Python-importable class structure is provided in [`examples/cpp-dummy/micro_cpp_dummy.cpp`](https://github.com/precice/micro-manager/blob/develop/examples/cpp-dummy/micro_cpp_dummy.cpp). It uses [pybind11](https://pybind11.readthedocs.io/en/stable/) to enable control and use from Python. Restructuring a C++ micro simulation code has the following steps

1. Create a C++ class which implements the functions given in the code snippet above.
The `solve()` function should have the following signature:

    ```cpp
    py::dict MicroSimulation::solve(py::dict macro_data, double dt)
    ```

    [`py::dict`](https://pybind11.readthedocs.io/en/stable/advanced/pycpp/object.html?#instantiating-compound-python-types-from-c) is a Python dictionary which can be used to pass data between Python and C++. Cast the data to the correct type before using it in C++ and vice versa.

2. Export the C++ class to Python using pybind11. Follow the instructions to exporting classes in the [pybind11 documentation](https://pybind11.readthedocs.io/en/stable/classes.html) or read their [first steps](https://pybind11.readthedocs.io/en/stable/basics.html) to get started.

3. Compile the C++ library including pybind11. For example, for the solverdummy, the command is

    ```bash
    c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) micro_cpp_dummy.cpp -o micro_dummy$(python3-config --extension-suffix)
    ```

    This will create a shared library `micro_dummy.so` which can be directly imported in Python.
    For more information on compiling C++ libraries, see the [pybind11 documentation](https://pybind11.readthedocs.io/en/stable/compiling.html).

## Next step

After restructuring your micro simulation code into a Python-importable class structure, [configure the Micro Manager](tooling-micro-manager-configuration.html).
