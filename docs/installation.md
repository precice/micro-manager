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

Unless already installed, the dependencies will be installed by `pip` during the installation procedure. To enable [crash handling by interpolation](tooling-micro-manager-running.html/#what-happens-when-a-micro-simulation-crashes), the optional dependency `sklearn` is required. To install `micro-manager-precice` with `sklearn`, run

```bash
pip install --user micro-manager-precice[sklearn]
```

To perform snapshot computations, the optional dependency `h5py` is required. To install `micro-manager-precice` with `h5py`, run

```bash
pip install --user micro-manager-precice[snapshot]
```

preCICE itself needs to be installed separately. If you encounter problems in the direct installation, see the [dependencies section](#required-dependencies) and [optional dependency section](#optional-dependencies) below.

### Option 2: Install manually

#### Required dependencies

Ensure that the following dependencies are installed:

* Python 3
* [preCICE](installation-overview.html) [v2.3.0](https://github.com/precice/precice/releases/tag/v2.3.0) or higher
* [pyprecice: Python language bindings for preCICE](installation-bindings-python.html)
* [numpy](https://numpy.org/install/)
* [mpi4py](https://mpi4py.readthedocs.io/en/stable/install.html)

#### Optional dependencies

* [sklearn](https://scikit-learn.org/stable/index.html)
* [h5py](https://www.h5py.org/) (required for snapshot computations)

#### Clone the Micro Manager

```bash
git clone https://github.com/precice/micro-manager.git
```

#### Install manually

To install using `pip`, go to the directory `micro-manager/` and run

```bash
pip install --user .
```

To install using Python, go to the project directory `micro-manager/` and run

```bash
python setup.py install --user
```

## Get the latest development version

If you want to use the latest development version of the Micro Manager, clone the [develop](https://github.com/precice/micro-manager/tree/develop) branch and then [build manually using pip](#install-manually).

## Next step

After successfully installing the Micro Manager, proceed to [preparing your micro simulation for the coupling](tooling-micro-manager-prepare-micro-simulation.html).
