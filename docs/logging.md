---
title: Logging in the Micro Manager
permalink: tooling-micro-manager-logging.html
keywords: tooling, macro-micro, two-scale
summary: The Micro Manager logs relevant information and adaptivity metrics.
---

The Micro Manager uses the Python [logging](https://docs.python.org/3/library/logging.html) functionality. The format is:

```bash
(<rank>) <date and time> <part/functionality of Micro Manager> <log level> <message>
```

For example:

```bash
(0) 04/17/2025 02:54:02 PM - micro_manager.micro_manager - INFO - Time window 1 converged.
```

The information (`INFO` level) message `Time window 1 converged.` from the file `micro_manager/micro_manager.py` is logged by rank `0` at `02:54:02 PM` on `04/17/2025`.

## Logging adaptivity metrics

If the Micro Manager is run with adaptivity, rank-wise and global metrics are output in CSV files. By default, the files are created in the working directory. To have the Micro Manager create the files in a specific folder, provide the folder path via the configuration parameter `output_dir`.

The following global metrics are logged to the file `adaptivity-metrics-global.csv`:

- Time window at which the metrics are logged
- Average number of active simulations
- Average number of inactive simulations
- Maximum number of active simulations
- Maximum number of inactive simulations

The CSV file heading is `n,avg active,avg inactive,max active,max inactive`.

The following local metrics are logged to the file `adaptivity-metrics-`rank`.csv`:

for local adaptivity:

- Time window at which the metrics are logged
- Number of active simulations
- Number of inactive simulations

The CSV file heading is `n,n active,n inactive`.

for global adaptivity:

- Time window at which the metrics are logged
- Number of active simulations
- Number of inactive simulations
- Ranks to which inactive simulations on this rank are associated

The CSV file heading is `n,n active,n inactive,assoc ranks`.

To set the output interval of adaptivity metrics, set `output_n` in the [adaptivity configuration](tooling-micro-manager-configuration.html/#adaptivity).
