---
title: Run the Micro Manager
permalink: tooling-micro-manager-running.html
keywords: tooling, macro-micro, two-scale
summary: Run the Micro Manager from the terminal with a configuration file as input argument or from a Python script.
---

The Micro Manager is run directly from the terminal by providing the path to the configuration file as an input argument in the following way

```bash
micro-manager-precice micro-manager-config.json
```

The Micro Manager can also be run in parallel

```bash
mpiexec -n micro-manager-precice micro-manager-config.json
```

## Handling Micro Simulation Crashes

In case of a micro simulation crash, the occurrence of an error with specific macro location of the crash and the error message itself are logged in the Micro Manager log-file. To continue a complete simulation run after a crash, results of the crashed micro simulation are interpolated using [Inverse Distance Weighting](https://en.wikipedia.org/wiki/Inverse_distance_weighting). If more than 20% of global micro simulations crash or if locally no neighbors are available for interpolation, the simulation run is terminated.
