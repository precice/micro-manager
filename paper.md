---
title: 'Micro Manager: a Python package for adaptive and flexible two-scale coupling'
tags:
  - Python
authors:
  - name: Ishaan Desai
    orcid: 0000-0002-2552-7509
    affiliation: 1
  - name: Erik Scheurer
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
Such complex simulations can be of multi-physics nature, meaning different physics solved in different parts of the domain.
To do such multi-physics simulations, we can couple different softwares together in a black-box manner. In some scenarios, the coupling is not just between different physics, but also different physical scales.
The Micro Manager, together with the coupling library preCICE [@preCICE_v2], enables such a multiscale, multi-physics coupling.
In two-scale coupled scenarios, the coarse scale can be referred to as a macro scale, and the fine scale can be the micro scale. The name *Micro Manager* is derived from its core functionality of controlling a set of micro scale simulations and coupling them to one macro scale simulation via preCICE.

# Statement of need

For multiscale simulations, the physics on different scales are essentially separate problems which are oftentimes evaluated using dedicated software. To understand how the scales affect each other, both the scales need to be solved simultaneously while being coupled in a bi-directional manner.
Two-scale coupled simulations have already been done in several application areas, such as, porous media  [@Bringedal_precipitation_dissolution, @Bringedal_reactive_porous_media], computational mechanics [@Fritzen_adaptivity] and biomechanics [@Lambers_liver_multiscale].
For each of these applications, the coupling software is implemented from scratch. Such software implementations typically involve communication between the scales, coupling schemes, and other case-specific technical solutions.
[@Groen_multiscale_survey] states that the field of generic multiscale coupling software is still maturing. Already-existing multiscale coupling software like [@MUSCLE3], [@MUI], and [@Amuse] are all tailored to particular multiscale computing patterns.
Compared to these softwares and according to [@Alowayyed_multiscale_exascale], our solution falls into the heterogeneous multiscale computing pattern. For this pattern, high-performance computing (HPC) software is still rare [@Alowayyed_multiscale_exascale]. Application-tailored softwares for multiscale simulations with massively parallel capabilities such as [@Natale_RAS_cancer] exist, but they do not propose a general software solution.
We present preCICE and the Micro Manager to couple simulations across scales in an application-agnostic way. preCICE is already able to couple simulations on one scale, and together with the Micro Manager, we are able to couple one macro-scale simulation with several micro-scale simulations.

# Overview

We introduce the concept of the Micro Manager in [@Desai2022micromanager]. The Micro Manager is written in Python and hosted on GitHub. In [@Desai2022micromanager], we describe the version 0.2.1, and in this paper we present version 0.3.0. The Micro Manager is essentially an executable which can be installed via pip and run directly on the command line. It is designed to work on all major Linux distributions which have Python 3.x support.

The micro simulation software needs to be converted to a callable library so that the Micro Manager can control it. The way to do this is described in the documentation [@MicroManager_documentation]. In the documentation, we demonstrate how to convert a Python or a C++ software into a callable library. Additionally, we show a working tutorial of a two-scale heat conduction problem where both the macro and micro scales are solved using the finite element library Nutils [@Nutils7].

The Micro Manager is configured via a JSON file and it can be run directly from the command line or its public methods can be called in a user-written Python file. It can run micro simulations in parallel using MPI [@mpi4py]. For real scenarios, the number of micro simulations can be very high and each micro simulation can be computationally expensive. The Micro Manager is able to adaptively activate and deactivate micro simulations depending on whether their similar counterparts exist. The adaptivity strategy is from [@Bastidas_two_scale, @Redeker_adaptivity]. The user can choose between a *local* and *global* adaptivity scheme \autoref{fig:ManagerSolution}, both of which are described in detail in the documentation [@MicroManager_documentation].

![Macro simulation with two materials coupled via preCICE to a set of micro simulations controlled by the Micro Manager. Micro simulations are run adaptively: highlighted ones are active, rest are inactive.\label{fig:ManagerSolution}](ManagerSolution_material.png)

The user is able to set up a two-scale coupled simulation by modifying an existing micro simulation software into a callable library and configuring the Micro Manager. preCICE and the Micro Manager handle the coupling and the high-performance computing aspects of the set up so that domain experts can concentrate on the macro and micro-scale models.

# Acknowledgements

The authors are funded by Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany's Excellence Strategy - EXC 2075 – 390740016. We acknowledge the support by the Stuttgart Center for Simulation Science (SimTech).

# References
