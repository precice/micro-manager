---
title: Get the Micro Manager
permalink: tooling-micro-manager-installation.html
keywords: tooling, macro-micro, two-scale
summary: Install the Micro Manager by running `pip install --user micro-manager-precice`.
---

## Get the latest Micro Manager release

### Option 1: Install using pip

The Micro Manager package on PyPI is [micro-manager-precice](https://pypi.org/project/micro-manager-precice/). To install, run

```bash
pip install micro-manager-precice
```

To enable [crash handling by interpolation](tooling-micro-manager-running.html/#what-happens-when-a-micro-simulation-crashes), the optional dependency sklearn is required. To install with sklearn, run

```bash
pip install micro-manager-precice[sklearn]
```

To use the Micro Manager for [snapshot computation](tooling-micro-manager-snapshot-configuration.html), the optional dependency h5py is required. To install with h5py, run

```bash
pip install micro-manager-precice[snapshot]
```

### Option 2: Install manually

#### Required dependencies

Ensure that the following dependencies are installed:

* Python 3
* [pyprecice: Python language bindings for preCICE](installation-bindings-python.html)
* [numpy](https://numpy.org/install/)
* [mpi4py](https://mpi4py.readthedocs.io/en/stable/install.html)
* [psutil](https://psutil.readthedocs.io/en/latest/)

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
pip install .
```

Adding optional dependencies works as above by adding them after the dot, e.g. `.[sklearn]`.

## Get the latest development version

If you want to use the latest development version of the Micro Manager, clone the [develop](https://github.com/precice/micro-manager/tree/develop) branch and then [build manually using pip](#install-manually).

## Next step

After successfully installing the Micro Manager, proceed to [preparing your micro simulation for the coupling](tooling-micro-manager-prepare-micro-simulation.html).
