---
title: Run the Micro Manager
permalink: tooling-micro-manager-running.html
keywords: tooling, macro-micro, two-scale
summary: Run the Micro Manager from the terminal with a configuration file as input argument or from a Python script.
---

The Micro Manager is run directly from the terminal by providing the path to the configuration file as an input argument in the following way

```bash
micro_manager micro-manager-config.json
```

Alternatively the Manager can also be run by creating a Python script which imports the Micro Manager package and calls its run function. For example a run script `run-micro-manager.py` would look like

```python
from micro_manager import MicroManager

manager = MicroManager("micro-manager-config.json")

manager.solve()
```

The Micro Manager can also be run in parallel, using the same script as stated above

```bash
mpirun -n <number-of-procs> python3 run-micro-manager.py
```
