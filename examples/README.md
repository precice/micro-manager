# Solverdummies

The `solverdummies` are minimal working examples for using the preCICE Micro Manager with different languages. At the moment, there are examples for Python, and C++. They can be coupled with any other solver, for example the `macro_dummy.py` in this directory.

## Python

To run the Python solverdummies, run the following commands in the `examples/` directory in **two different terminals**:

```bash
python macro_dummy.py
python python-dummy/run_micro_manager.py --config micro-manager-config.json
```

Note that running `micro_manager micro-manager-config.json` from the terminal will not work, as the path in the configuration file is relative to the current working directory. See [#36](https://github.com/precice/micro-manager/issues/36) for more information.

To run the Python solverdummies with adaptivity run the following commands in the `examples/` directory in **two different terminals**:

```bash
python macro_dummy.py
python python-dummy/run_micro_manager.py --config micro-manager-adaptivity-config.json
```

## C++

The C++ solverdummies have to be compiled first using [`pybind11`](https://pybind11.readthedocs.io/en/stable/index.html). To do so, install `pybind11` using `pip`:

```bash
pip install pybind11
```

Then, run the following commands in the `cpp-dummy` directory:

```bash
c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) micro_cpp_dummy.cpp -o micro_dummy$(python3-config --extension-suffix)
```

<details>
<summary>Explanation</summary>

The command above compiles the C++ solverdummy and creates a shared library that can be imported from python using `pybind11`.

- The `$(python3 -m pybind11 --includes)` part is necessary to include the correct header files for `pybind11`.
- The `$(python3-config --extension-suffix)` part is necessary to create the correct file extension for the shared library. For more information, see the [pybind11 documentation](https://pybind11.readthedocs.io/en/stable/compiling.html#building-manually).
- If you have multiple versions of Python installed, you might have to replace `python3-config` with `python3.8-config` or similar.

</details>

Then, run the following commands in the `examples/` directory, in **two different terminals**:

```bash
python macro_dummy.py
python cpp-dummy/run_micro_manager.py --config micro-manager-config.json
```

To run the C++ solverdummies with adaptivity run the following commands in the `examples/` directory in **two different terminals**:

```bash
python macro_dummy.py
python cpp-dummy/run_micro_manager.py --config micro-manager-adaptivity-config.json
```

When changing the C++ solverdummy to your own solver, make sure to change the `PYBIND11_MODULE` in `micro_cpp_dummy.cpp` to the name that you want to compile to.
For example, if you want to import the module as `my_solver`, change the line to `PYBIND11_MODULE(my_solver, m) {`. Then, change the `micro_file_name` in `micro-manager-config.json` to `my_solver`.
