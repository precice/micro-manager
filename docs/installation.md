---
title: Installing the Micro Manager
permalink: tooling-micro-manager-installation.html
keywords: tooling, macro-micro, two-scale
summary: Install the Micro Manager by running `pip install --user micro-manager-precice`.
---

## Installation

The Micro Manager is a Python package that can be installed using `pip`. Make sure [preCICE](installation-overview.html) is installed before installing the Micro Manager. The Micro Manager is compatible with preCICE version [2.5.0](https://github.com/precice/precice/releases/tag/v2.5.0).

### Option 1: Using pip

It is recommended to install [micro-manager-precice from PyPI](https://pypi.org/project/micro-manager-precice/) by running

```bash
pip install --user micro-manager-precice
```

Unless already installed, the dependencies will be installed by `pip` during the installation procedure. preCICE itself needs to be installed separately. If you encounter problems in the direct installation, see the [dependencies section](#required-dependencies) below.

### Option 2: Clone this repository and install manually

#### Required dependencies

Ensure that the following dependencies are installed:

* Python 3 or higher
* [preCICE](installation-overview.html) [v2.5.0](https://github.com/precice/precice/releases/tag/v2.5.0)
* [pyprecice: Python language bindings for preCICE](installation-bindings-python.html)
* [numpy](https://numpy.org/install/)
* [mpi4py](https://mpi4py.readthedocs.io/en/stable/install.html)

#### Build and install the Manager using pip

After cloning this repository, go to the directory `micro-manager/` and run

```bash
pip install --user .
```

#### Build and install the Manager using Python

After cloning this repository, go to the project directory `micro-manager/` and run

```bash
python setup.py install --user
```
