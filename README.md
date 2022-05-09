# Micro Manager

A manager tool which facilitates two-scale macro-micro coupling using the coupling library preCICE.

# Installing the package

## Using pip3

It is recommended to install [micro-manager from PyPI]() by running

```bash
pip3 install --user micro-manager
```

This should work directly if all the dependencies are installed correctly. If you encounter problems in this, see the dependencies section below.

## Clone this repository and use pip3

### Required dependencies

Ensure that the following dependencies are installed:

* [preCICE](https://github.com/precice/precice/wiki)
* python3 (this adapter **only supports python3**)
* [the python language bindings for preCICE](https://github.com/precice/python-bindings)
* [numpy]() and [mpi4py]()

### Build and install the manager

After cloning this repository, go to the project directory `micro-manager` and run `pip3 install --user .`.

## Using the micro-manager

### How to configure a micro-simulation script to be coupled via the micro-manager

The micro-simulation script needs to be converted into a library having a class structure which would be callable from the micro-manager.
The micro-manager creates objects of this class for each micro-simulation and controls them till the end of the coupled simulation.
The Micro Manager script is intended to be used *as is*, and to facilitate that, certain conventions need to be followed.

#### Steps to convert micro-simulation code to a callable library

* Create a class called `MicroSimulation` which consists of all the functions of the micro-simulation.
* Apart from the class constructor, define a function `initialize` which should consist of all steps to fully define the initial state of the micro-simulation
* Create a function named `solve` which should consist of all the solving steps for one time step of a micro-simulation or is the micro-problem is steady-state then solving until the steady-state is achieved.  The `solve` function will have all the steps which need to be done repeatedly.
* If implicit coupling is required between the macro- and micro- problems, then you can additionally define two functions `save_checkpoint` and `revert_to_checkpoint`.
  * `save_checkpoint` saves the state of the micro-problem such that if there is no convergence then the micro-problem can be reversed to this state.
  * `revert_to_checkpoint` reverts to the state which was saved earlier.

### Configuring the Micro Manager

The Micro Manager is configured using a JSON file. For the example above, the configuration file is [micro-manager-config.json](https://github.com/IshaanDesai/coupled-heat-conduction/blob/main/micro-manager-config.json).
Most of the configuration quantities are self explanatory, some of the important ones are:
* `micro_file_name` is the path to the micro-simulation script. The `.py` of the micro-simulation script is not necessary here.
* The entities `write_data_name` and `read_data_name` need to be lists which carry names of the data entities as strings.
* `macro_domain_bounds` has the lower and upper [min and max] limits of the macro-domain. The entires are of the form [xmin, xmax, ymin, ymax]. Currently only 2D simulations are supported by the Micro Manager.

