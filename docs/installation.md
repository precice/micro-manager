---
title: Get the Micro Manager
permalink: tooling-micro-manager-installation.html
keywords: tooling, macro-micro, two-scale
summary: Install the Micro Manager by running `pip install --user micro-manager-precice`.
---

## Get the latest Micro Manager release

The Micro Manager can be installed using `pip`. Make sure [preCICE](installation-overview.html) is installed before installing the Micro Manager. The Micro Manager is compatible with preCICE version [2.3.0](https://github.com/precice/precice/releases/tag/v2.3.0) and higher.

### Option 1: Install from PyPI

The Micro Manager package has the name [micro-manager-precice](https://pypi.org/project/micro-manager-precice/) on PyPI. To install `micro-manager-precice`, run

```bash
pip install --user micro-manager-precice
```

Unless already installed, the dependencies will be installed by `pip` during the installation procedure. preCICE itself needs to be installed separately. If you encounter problems in the direct installation, see the [dependencies section](#required-dependencies) below.

### Option 2: Install manually

#### Required dependencies

Ensure that the following dependencies are installed:

* Python 3
* [preCICE](installation-overview.html) [v2.3.0](https://github.com/precice/precice/releases/tag/v2.3.0) or higher
* [pyprecice: Python language bindings for preCICE](installation-bindings-python.html)
* [numpy](https://numpy.org/install/)
* [mpi4py](https://mpi4py.readthedocs.io/en/stable/install.html)

#### Clone this repository

```bash
git clone https://github.com/precice/micro-manager.git
```

#### Build manually using pip

Go to the directory `micro-manager/` and run

```bash
pip install --user .
```

#### Build manually using Python

Go to the project directory `micro-manager/` and run

```bash
python setup.py install --user
```

## Get the latest development version

If you want to use the latest development version of the Micro Manager, clone the develop[https://github.com/precice/micro-manager/tree/develop] branch and then [build manually using pip](#build-manually-using-pip).
