# Solverdummies

The solverdummies are minimum working examples to show how the Micro Manager works with Python and C++ based micro problems.
The Micro Manager couples to `macro_dummy.py` via preCICE. The `macro_dummy.py` needs the input `adaptivity` or `no_adaptivity` depending on which type of dummy problem is being solved.

## Python

To run the Python solverdummies, run the commands given below in the `examples/` directory in **two different terminals**.

First terminal:

```bash
python macro_dummy.py no_adaptivity
```

Second terminal:

```bash
micro-manager-precice micro-manager-python-config.json
```

To run the Python solverdummies with adaptivity, run the commands given below in the `examples/` directory in **two different terminals**.

First terminal:

```bash
python macro_dummy.py adaptivity
```

Second terminal:

```bash
micro-manager-precice micro-manager-python-adaptivity-config.json
```

## C++

The C++ solverdummies have to be compiled first using [pybind11](https://pybind11.readthedocs.io/en/stable/index.html).
Run the following commands in the `cpp-dummy` directory:

```bash
c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) micro_cpp_dummy.cpp -o micro_dummy$(python3-config --extension-suffix)
```

<details>
<summary>Explanation</summary>

The command above compiles the C++ solverdummy and creates a shared library that can be imported from python using `pybind11`.

- The `$(python3 -m pybind11 --includes)` part is necessary to include the correct header files for `pybind11`.
- The `$(python3-config --extension-suffix)` part is necessary to create the correct file extension for the shared library. For more information, see the [pybind11 documentation](<https://pybind11.readthedocs.io/en/stable/compiling.html#building-manually>).
- If you have multiple versions of Python installed, you might have to replace `python3-config` with `python3.8-config` or similar.

</details>

To run the C++ micro problem case with adaptivity, run the following commands in the `examples/` directory in **two different terminals**:

First terminal:

```bash
python macro_dummy.py adaptivity
```

Second terminal:

```bash
micro-manager-precice micro-manager-cpp-adaptivity-config.json
```
