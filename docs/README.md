---
title: The Micro Manager
permalink: tooling-micro-manager-overview.html
keywords: tooling, macro-micro, two-scale
summary: The Micro Manager is a tool to facilitate solving two-scale (macro-micro) coupled problems using the coupling library preCICE.
---

## What is this?

The Micro Manager is a tool to facilitate solving two-scale (macro-micro) coupled problems using the coupling library [preCICE](https://www.precice.org/).

## What can it do?

The Micro Manager is able to couple many micro simulations with one macro simulation. This includes:

- Passing data between micro and macro simulations
- Running micro simulations in parallel using MPI
- Adaptively turning micro simulations on and off based on similar micro simulations globally and locally

## Documentation

For a more detailed look, the documentation is split into the following sections:

- [Installation](tooling-micro-manager-installation.html)
- [Usage](tooling-micro-manager-usage.html)
  - [Code Changes](tooling-micro-manager-code-changes.html)
  - [Configuration](tooling-micro-manager-configuration.html)
  - [Running](tooling-micro-manager-running.html)
