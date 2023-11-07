---
title: 'Micro Manager: a Python package for adaptive and flexible two-scale coupling'
tags:
  - Python
authors:
  - name: Ishaan Desai
    orcid: 0000-0002-2552-7509
    affiliation: 1
  - name: Erik Scheurer
    orcid: 0009-0000-9396-2994
    affiliation: 1
  - name: Carina Bringedal
    orcid: 0000-0003-0495-2634
    affiliation: 2
  - name: Benjamin Uekermann
    orcid: 0000-0002-1314-9969
    affiliation: 1
affiliations:
 - name: Institute for Parallel and Distributed Systems, University of Stuttgart, Germany
   index: 1
 - name: Department of Computer Science, Electrical Engineering and Mathematical Sciences, Western Norway University of Applied Sciences, Norway
   index: 2
date: 31 August 2023
bibliography: paper.bib
---

# Summary

The Micro Manager facilitates coupling between simulation software packages, which solve problems at different physical scales. Broadly speaking, simulation-based analysis is an effective tool to gain insights in real-world scenarios without incurring the high cost of prototyping and testing.
Complex simulations are oftentimes broken down into simpler components, which are resolved by tailor-made software.
Such complex simulations can be of multiphysics nature, meaning different physics are solved in different parts of the domain.
To do such multiphysics simulations, we can couple different software packages together. preCICE [@preCICE_v2] is an open-source coupling library for partitioned multiphysics simulations. However, sometimes the coupling is not just between different physics but also different physical scales, which is frequently referred to as multiscale coupling. To understand how the scales affect each other, both scales need to be solved simultaneously while being coupled in a bi-directional manner.
The Micro Manager, together with preCICE, is capable of handling a multiphysics and multiscale coupling.
We refer to the coarse scale as the macro scale, and the fine scale as the micro scale. The name *Micro Manager* is derived from its core functionality of controlling a set of micro-scale simulations and coupling them to one macro-scale simulation via preCICE.

# Statement of need

Two-scale coupled simulations have already been done in several application areas, such as porous media  [e.g., @Bastidas_two_scale; @Gaerttner_two_scale], computational mechanics [e.g., @Fritzen_adaptivity] and biomechanics [e.g., @Lambers_liver_multiscale].
For each of these publications, the coupling software is implemented from scratch. Such implementations typically involve communication between the scales, coupling schemes, and other case-specific technical solutions.
For coupled problems on a single scale, preCICE handles these coupling aspects.
The Micro Manager is a thin layer on top of preCICE, which enables preCICE to couple problems across two scales.
Compared to existing multiscale coupling software such as [MUSCLE3](https://github.com/multiscale/muscle3) [@MUSCLE3], [MUI](https://github.com/MxUI/MUI) [@MUI], and [AMUSE](https://github.com/amusecode/amuse) [@Amuse], the Micro Manager is an add-on package to the general coupling library preCICE and not a stand-alone coupling solution.

For single-scale simulations, many widely-used solvers such as OpenFOAM, FEniCS, deal.II, and more are already coupled using preCICE. Additionally, preCICE has a steadily growing community [@preCICE_v2]. Using the Micro Manager, we expose these advantages of using preCICE for multiscale scenarios.
According to @Alowayyed_multiscale_exascale, our solution falls into the heterogeneous multiscale computing pattern. For this pattern, high-performance computing (HPC) software is still rare [@Alowayyed_multiscale_exascale]. Application-tailored software packages for multiscale simulations with massively parallel capabilities, such as the one by @Natale_RAS_cancer, exist, but they do not propose a general software solution.
preCICE scales on tens of thousands of MPI ranks [@preCICE_HPC] and the Micro Manager is capable of adaptively (\autoref{fig:ManagerSolution}) running micro simulations in parallel. We are proposing a software solution that could potentially solve large two-scale coupled problems efficiently, while building on existing single-scale know-how.

![Macro simulation with an averaged view of the materials (illustrated by stripes) is coupled via preCICE to a set of micro simulations controlled by the Micro Manager. The enlarged micro simulation shows a representative micro structure with the different materials. Micro simulations are run adaptively: highlighted ones are active, rest are inactive.\label{fig:ManagerSolution}](ManagerSolution.png)

# Functionality & Use

In @Desai2022micromanager, we use the Micro Manager to solve a two-scale heat conduction problem, where both the macro and micro scales are solved using the finite element library Nutils [@Nutils7]. @Kschidock2023DuMuxMacroMicro shows the flexibility of preCICE and the Micro Manager by solving the same problem using DuMu^x^ [@Koch2021DuMux].
The micro-scale simulation needs to be converted to a callable library so that the Micro Manager can control it. In the [documentation](https://precice.org/tooling-micro-manager-prepare-micro-simulation.html), we demonstrate how to convert a Python or a C++ program into a callable library. The macro-scale simulation is coupled directly to preCICE. This coupling is black-box; hence, the macro-scale simulation has no knowledge of the macro-micro coupling or the Micro Manager.

The Micro Manager is configured via a [JSON](https://www.json.org/json-en.html) file. It can run micro simulations in parallel using MPI [@mpi4py]. For realistic multiscale scenarios, the number of micro simulations can be very high, and each micro simulation can be computationally expensive. The Micro Manager is able to adaptively activate and deactivate micro simulations depending on whether their similar counterparts exist [@Redeker_adaptivity].

In addition to the two-scale heat conduction problem in @Desai2022micromanager, the Micro Manager has already been used in multiscale models of the human liver in which a lobule-scale continuum-biomechanical model is coupled to many cell-scale models [@Otlinghaus2022Liver].

The Micro Manager is written in Python and hosted on [GitHub](https://github.com/precice/micro-manager). New versions are released and packaged for [PyPI](https://pypi.org/project/micro-manager-precice/). We recommend installing the Micro Manager via pip and running it directly on the command line or by calling its public methods. It is designed to work on all major Linux distributions that have Python 3.x support.

# Acknowledgements

The authors are funded by Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany's Excellence Strategy - EXC 2075 – 390740016. We acknowledge the support of the Stuttgart Center for Simulation Science (SimTech).

# References
