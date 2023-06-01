---
title: The Micro Manager
permalink: tooling-micro-manager-overview.html
keywords: tooling, macro-micro, two-scale
summary: A tool to manage many micro simulations and couple them to a macro simulation via preCICE.
---

## What is this?

The Micro Manager is a tool for solving coupled problems where the coupling is across scales (for example macro-micro).

![Micro Manager strategy schematic](images/ManagerSolution.pdf)

## What can it do?

The Micro Manager couples many micro simulations with one macro simulation. This includes

- transferring scalar and vector data to and from a large number of micro simulations.
- running micro simulations in parallel using MPI.
- adaptively activating and deactivating micro simulations based on whether their similar exist.

## Documentation

The Micro Manager creates instances of several micro simulations and couples them to one macro simulation, using preCICE.

An existing micro simulation code needs to be converted into a library with a specific class name which has functions with specific names. For a macro-micro coupled problem, the macro simulation code is coupled to preCICE directly. The section [couple your code](couple-your-code-overview.html) of the preCICE documentation gives more details on coupling existing codes. To setup a macro-micro coupled simulation using the Micro Manager, follow the steps

- [Installation](tooling-micro-manager-installation.html)
- [Micro simulation as callable library](tooling-micro-manager-micro-simulation-callable-library.html)
- [Configuration](tooling-micro-manager-configuration.html)
- [Running](tooling-micro-manager-running.html)
