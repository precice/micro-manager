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
