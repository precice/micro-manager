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

The Micro Manager aims to facilitate a particular type of coupling between simulations at different scales. Broadly speaking,physics-based simulation analysis is an effective tool to gain insights in real-world scenarios without incurring the high cost of prototyping and testing.
Complex simulations are oftentimes broken down into simpler components which are resolved by specific software tailored to solve the set of problems.
Such complex simulations can be of multi-physics nature, in which different physics is solved in different parts of the domain.
To perform such multi-physics simulations, we can couple different softwares together in a black-box manner. In some scenarios, coupling at just one physical scale may not be enough, and models needs to be solved on multiple scales. The Micro Manager enables such type of a multiscale, multi-physics coupling between two softwares.

# Statement of need

For multiscale simulations, the physics on different scales are essentially separate problems which are evaluated using dedicated software. To understand how the scales affect each other, we present a software stack of preCICE [preCICE-v2] and the Micro Manager. Our software stack aims to couple simulations across scales in an application-agnostic way. preCICE is already able to facilitate coupling on one scale. There are several application areas where such macro-micro coupled simulations have already been done, for example, porous media [1, 2], dual-phase steel simulation [3], computational mechanics [4] and biomechanics [5]. In each of these applications, the coupling methodologies are mostly developed from scratch. These methodologies typically involve efficient data transfer and communication between the scales, different coupling schemes, and technical solutions for how to combine different programming languages. This work builds on the functionality of the coupling library preCICE [6] to develop a software framework that can facilitate application-agnostic macro-micro coupling. The development of a flexible macro-micro coupling software has been previously discussed
from different perspectives. Groen et al. (2019) [7] states that the field of generic multiscale
coupling software is still maturing. Even though software packages such as MUSCLE3 [8],
MUI [9], or Amuse [10] exist, they are all tailored to particular multiscale computing patterns.
The macro-micro coupling we are addressing in this work falls into the heterogeneous multiscale
computing pattern [11]. For this pattern, high-performance computing (HPC) software is still
rare [11]. There exist, however, already application-tailored solutions for massively parallel
simulations, for example Klawonn et al. (2020) [3] or Di Natale et al. (2019) [12].


# Overview

The concept of the Micro Manager was first introduced via [Desai2022micromanager].

# Acknowledgements

The authors are funded by Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany's Excellence Strategy - EXC 2075 – 390740016. We acknowledge the support by the Stuttgart Center for Simulation Science (SimTech).

# References
