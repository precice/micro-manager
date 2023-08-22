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

The Micro Manager facilitates coupling between simulation softwares which solve problems at different physical scales. Broadly speaking, simulation-based analysis is an effective tool to gain insights in real-world scenarios without incurring the high cost of prototyping and testing.
Complex simulations are oftentimes broken down into simpler components which are resolved by tailor-made software.
Such complex simulations can be of multiphysics nature, meaning different physics are solved in different parts of the domain.
To do such multiphysics simulations, we can couple different softwares together. preCICE [@preCICE_v2] is an open-source coupling library for partitioned multiphysics simulations. Sometimes the coupling is not just between different physics, but also different physical scales, which is oftentimes referred to as multiscale coupling.
The Micro Manager, together with preCICE, is capable of handling a multiphysics and multiscale coupling.
In two-scale coupled scenarios, the coarse scale can be referred to as the macro scale, and the fine scale as the micro scale. The name *Micro Manager* is derived from its core functionality of controlling a set of micro-scale simulations and coupling them to one macro-scale simulation via preCICE.

# Statement of need

For multiscale simulations, the physics on different scales are essentially separate problems, which are evaluated using dedicated software. To understand how the scales affect each other, both scales need to be solved simultaneously while being coupled in a bi-directional manner.
Two-scale coupled simulations have already been done in several application areas, such as, porous media  [@Bringedal_precipitation_dissolution; @Bringedal_reactive_porous_media], computational mechanics [@Fritzen_adaptivity] and biomechanics [@Lambers_liver_multiscale].
For each of these applications, the coupling software is implemented from scratch. Such software implementations typically involve communication between the scales, coupling schemes, and other case-specific technical solutions.
@Groen_multiscale_survey states that the field of generic multiscale coupling software is still maturing. Already-existing multiscale coupling software such as [MUSCLE3](https://github.com/multiscale/muscle3) [@MUSCLE3], [MUI](https://github.com/MxUI/MUI) [@MUI], and [AMUSE](https://github.com/amusecode/amuse) [@Amuse] are all tailored to particular multiscale computing patterns.
The Micro Manager reuses functionality from preCICE to make generic two-scale coupling possible. preCICE is already coupled to many widely-used solvers, like OpenFOAM, FEniCS, deal.II and more, which allows for additionally flexibility in setting up two-scale coupled problems.
According to @Alowayyed_multiscale_exascale, our solution falls into the heterogeneous multiscale computing pattern. For this pattern, high-performance computing (HPC) software is still rare [@Alowayyed_multiscale_exascale]. Application-tailored softwares for multiscale simulations with massively parallel capabilities such as @Natale_RAS_cancer exist, but they do not propose a general software solution.
preCICE is already able to couple simulations on the same scale, and together with the Micro Manager, we are able to couple one macro-scale simulation with several micro-scale simulations (\autoref{fig:ManagerSolution}).

![Macro simulation with two materials coupled via preCICE to a set of micro simulations controlled by the Micro Manager. Lower left micro simulation shows a representative micro structure with two materials. Micro simulations are run adaptively: highlighted ones are active, rest are inactive.\label{fig:ManagerSolution}](ManagerSolution.png)

# Software

In @Desai2022micromanager, we use the Micro Manager to solve a two-scale heat conduction problem, where both the macro and micro scales are solved using the finite element library Nutils [@Nutils7].
The micro-scale simulation needs to be converted to a callable library so that the Micro Manager can control it. In the [documentation](https://precice.org/tooling-micro-manager-prepare-micro-simulation.html), we demonstrate how to convert a Python or a C++ program into a callable library. The macro-scale simulation is coupled directly to preCICE.

The Micro Manager is configured via a [JSON](https://www.json.org/json-en.html) file. It can run micro simulations in parallel using MPI [@mpi4py]. For realistic multiscale scenarios, the number of micro simulations can be very high and each micro simulation can be computationally expensive. The Micro Manager is able to adaptively activate and deactivate micro simulations depending on whether their similar counterparts exist [@Redeker_adaptivity and @Bastidas_two_scale]. In the configuration, the user can choose between a *local* and *global* adaptivity scheme, both of which are described in the [documentation](https://precice.org/tooling-micro-manager-configuration.html#adaptivity).

The user is able to set up a two-scale coupled simulation by modifying an existing micro simulation software into a callable library and configuring the Micro Manager. preCICE and the Micro Manager handle the coupling and the high-performance computing aspects of the set up so that domain experts can concentrate on the macro and micro-scale models.

# Availability & Use

The Micro Manager is written in Python and hosted on [GitHub](https://github.com/precice/micro-manager). New versions are released and packaged for [PyPI](https://pypi.org/project/micro-manager-precice/). We recommend installing the Micro Manager via pip and running it directly on the command line or by calling its public methods. It is designed to work on all major Linux distributions which have Python 3.x support.

# Acknowledgements

The authors are funded by Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany's Excellence Strategy - EXC 2075 – 390740016. We acknowledge the support by the Stuttgart Center for Simulation Science (SimTech).

# References
