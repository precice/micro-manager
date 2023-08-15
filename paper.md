---
title: 'Micro Manager: a Python package for adaptive and flexible two scale macro-micro coupling'
tags:
  - Python
authors:
  - name: Ishaan Desai
    orcid: 0000-0002-2552-7509
    equal-contrib: true
    affiliation: 1
  - name: Erik Scheurer
    affiliation: 1
  - name: Carina Bringedal
    orcid: 0000-0003-0495-2634
    equal-contrib: true
    affiliation: 2
  - name: Benjamin Uekermann
    orcid: 0000-0002-1314-9969
    equal-contrib: true
    affiliation: 1
affiliations:
 - name: Institute for Parallel and Distributed Systems, University of Stuttgart, Germany
   index: 1
 - name: Department of Computer science, Electrical engineering and Mathematical sciences, Western Norway University of Applied Sciences, Norway
   index: 2
date: 31 August 2023
bibliography: paper.bib
---

# Summary

The Micro Manager facilitates coupling between simulation softwares which solve problems at different scales. Broadly speaking,physics-based simulation analysis is an effective tool to gain insights in real-world scenarios without incurring the high cost of prototyping and testing.
Complex simulations are oftentimes broken down into simpler components which are resolved by specific software tailored to solve the set of problems.
Such complex simulations can be of multi-physics nature, in which different physics is solved in different parts of the domain.
To perform such multi-physics simulations, we can couple different softwares together in a black-box manner. In some scenarios, coupling at just one physical scale may not be enough, and models needs to be solved on multiple scales.
The Micro Manager, together with the coupling library preCICE [@preCICE-v2], enables such a multiscale, multi-physics coupling between two softwares.
In two-scale scenarios, the coarse scale can be referred to as a macro scale, and the fine scale can be the micro scale. The name *Micro Manager* is derived from its core functionality of controlling a set of micro scale simulations and coupling them to one macro scale simulation via preCICE.

# Statement of need

For multiscale simulations, the physics on different scales are essentially separate problems which are oftentimes evaluated using dedicated software. To understand how the scales affect each other, the software evaluating the scales needs to be coupled in a bi-directional manner.
Two-scale coupled simulations have already been done in several application areas, such as, porous media  [Bringedal_precipitation_dissolution, Bringedal_reactive_porous_media], computational mechanics [@Fritzen_adaptivity] and biomechanics [@Lambers_liver_multiscale].
For each of these applications, the coupling software is implemented from scratch. Such software implementations typically involve communication between the scales, coupling schemes, and other case-specific technical solutions.
[@Groen_multiscale_survey] states that the field of generic multiscale coupling software is still maturing. Already-existing multiscale coupling software like [@MUSCLE3], [@MUI], and [@Amuse] are all tailored to particular multiscale computing patterns.
Compared to these softwares and according to [@Alowayyed_multiscale_exascale], our solution falls into the heterogeneous multiscale computing pattern. For this pattern, high-performance computing (HPC) software is still rare [@Alowayyed_multiscale_exascale]. Application-tailored softwares for multiscale simulations with massively parallel capabilities such as [@Natale_RAS_cancer] exist, but they do not propose a general software solution.
We present a software stack of preCICE and the Micro Manager to couple simulations across scales in an application-agnostic way. preCICE is already able to facilitate coupling on one scale, and together with the Micro Manager, we are able to couple one macro scale simulation with several micro scale simulations.

# Overview

We introduced the concept of the Micro Manager in [@Desai2022micromanager]. The Micro Manager is written in Python and hosted on GitHub. In [@Desai2022micromanager], we described the version 0.2.1, and now we present version 0.3.0. The Micro Manager is essetially an executable which can be installed on a machine via pip and run directly on the command line. It is designed to work on all major Linux distributions which have reasonable Python support.





# Acknowledgements

The authors are funded by Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany's Excellence Strategy - EXC 2075 – 390740016. We acknowledge the support by the Stuttgart Center for Simulation Science (SimTech).

# References
