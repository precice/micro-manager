---
title: Usage
permalink: tooling-micro-manager-usage.html
keywords: tooling, macro-micro, two-scale
---

## Using the Micro Manager

The Micro Manager facilitates two-scale coupling between one macro-scale and many micro-scale simulations. It creates instances of several micro simulations and couples them to one macro simulation, using preCICE.

An existing micro simulation code needs to be converted into a library with a specific class name which has functions with specific names. The next section describes the required library structure of the micro simulation code. On the other hand, the macro simulation code is coupled to preCICE directly. The section [couple your code](couple-your-code-overview.html) of the preCICE documentation gives more details on coupling existing codes.

To run your first macro-micro simulation using the Micro Manager follow the three steps:

1. [Create a micro simulation library](tooling-micro-manager-usage-code-changes.html)
2. [Create a configuration file](tooling-micro-manager-usage-configuration.html)
3. [Run the Micro Manager](tooling-micro-manager-usage-running.html)
