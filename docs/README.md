---
title: The Micro Manager
permalink: tooling-micro-manager-overview.html
keywords: tooling, macro-micro, two-scale
summary: The Micro Manager is a tool for solving two-scale (macro-micro) coupled problems using the coupling library preCICE.
---

## What is this?

The Micro Manager is a tool for solving coupled problems where the coupling is across scales (for example macro-micro). It is developed as a library extension to the coupling library [preCICE](https://www.precice.org/).

## What can it do?

The Micro Manager couples many micro simulations with one macro simulation. This includes

- transferring scalar and vector data to and from a large number of micro simulations.
- running micro simulations in parallel using MPI.
- adaptively activating and deactivating micro simulations based on whether their similar exist.

## Documentation

For a more detailed look, the documentation is split into the following sections

- [Installation](tooling-micro-manager-installation.html)
- [Micro simulation as callable library](tooling-micro-manager-micro-simulation-callable-library.html)
- [Configuration](tooling-micro-manager-configuration.html)
- [Running](tooling-micro-manager-running.html)
