# Micro Manager changelog

## latest

- Improve logging wrapper function names to be more clear https://github.com/precice/micro-manager/pull/153
- Remove adaptivity computation CPU time export functionality https://github.com/precice/micro-manager/pull/152
- Replace `Allgatherv` with `allgather` to avoid running into the error of size buffer https://github.com/precice/micro-manager/pull/151
- Update Actions workflows due to updates in `precice/precice:nightly` https://github.com/precice/micro-manager/pull/150
- Move adaptivity CPU time output from preCICE export to metrics logging https://github.com/precice/micro-manager/pull/149
- Fix bug in the domain decomposition which was returning incorrect bounding box limits for the decomposition of `[2, 2, 1]` and similar https://github.com/precice/micro-manager/pull/146
- Fix bug in calling of the adaptivity computation for explicit coupling scenarios https://github.com/precice/micro-manager/pull/145
- Fix bug in handling of vector data returned by the MicroSimulation `solve()` method, for scenarios with adaptivity https://github.com/precice/micro-manager/pull/143
- Remove the `scalar` and `vector` keyword values from data names in configuration https://github.com/precice/micro-manager/pull/142
- Set default logger to stdout and add output directory setting option for file loggers https://github.com/precice/micro-manager/pull/139
- Remove the `adaptivity_data` data structure and handle all adaptivity data internally https://github.com/precice/micro-manager/pull/137
- Improve logging by wrapping Python logger in a class https://github.com/precice/micro-manager/pull/133
- Refactor large parts of solve and adaptivity to group datasets and simplify handling https://github.com/precice/micro-manager/pull/135
- Add information about adaptivity tuning parameters https://github.com/precice/micro-manager/pull/131
- Put computation of counting active steps inside the adaptivity variant `if` condition https://github.com/precice/micro-manager/pull/130

## v0.5.0

- Use absolute values to calculate normalizing factor for relative norms in adaptivity https://github.com/precice/micro-manager/pull/125
- Add option to use only one micro simulation object in the snapshot computation https://github.com/precice/micro-manager/pull/123
- Explicitly check if time window has converged using the API function `is_time_window_complete()` https://github.com/precice/micro-manager/pull/118
- Add `MicroManagerSnapshot` enabling snapshot computation and storage of microdata in HDF5 format https://github.com/precice/micro-manager/pull/101
- Make `sklearn` an optional dependency
- Move the config variable `micro_dt` from the coupling parameters section to the simulation parameters section https://github.com/precice/micro-manager/pull/114
- Set time step of micro simulation in the configuration, and use it in the coupling https://github.com/precice/micro-manager/pull/112
- Add a base class called `MicroManager` with minimal API and member function definitions, rename the existing `MicroManager` class to `MicroManagerCoupling` https://github.com/precice/micro-manager/pull/111
- Handle calling `initialize()` function of micro simulations written in languages other than Python https://github.com/precice/micro-manager/pull/110
- Check if initial data returned from the micro simulation is the data that the adaptivity computation requires https://github.com/precice/micro-manager/pull/109
- Use executable `micro-manager-precice` by default, and stop using the script `run_micro_manager.py` https://github.com/precice/micro-manager/pull/105
- Make `initialize()` method of the MicroManager class public https://github.com/precice/micro-manager/pull/105
- Optionally use initial macro data to initialize micro simulations https://github.com/precice/micro-manager/pull/104
- Use `pyproject.toml` instead of `setup.py` to configure the build. Package name is now `micro_manager_precice` https://github.com/precice/micro-manager/pull/84
- Add handling of crashing micro simulations https://github.com/precice/micro-manager/pull/85
- Add switch to turn adaptivity on and off in configuration https://github.com/precice/micro-manager/pull/93

## v0.4.0

- Add note in the cpp-dummy that pickling support does not work due to no good way to pass the sim id to the new micro simulation instance [commit](https://github.com/precice/micro-manager/commit/0a82966676717a533aca9bffa4a110453158f29c)
- Reintroduce initialize function in the micro simulation API https://github.com/precice/micro-manager/pull/79
- Use Allgatherv instead of allgather when collecting number of micro simulations on each rank in initialization https://github.com/precice/micro-manager/pull/81
- Remove the callable function `initialize()` from the micro simulation API [commit](https://github.com/precice/micro-manager/commit/bed5a4cc0f03b780da7f62b3f51ed1df2796588c)
- Pass an ID to the micro simulation object so that it is aware of its own uniqueness https://github.com/precice/micro-manager/pull/66
- Resolve bug which led to an error when global adaptivity was used with unequal number of simulations on each rank https://github.com/precice/micro-manager/pull/78
- Make the `initialize()` method of the MicroManager class private https://github.com/precice/micro-manager/pull/77
- Add reference paper via a CITATION.cff file [commit](https://github.com/precice/micro-manager/commit/6c08889c658c889d6ab5d0867802522585abcee5)
- Add JOSS DOI badge [commit](https://github.com/precice/micro-manager/commit/2e3c2a4c77732f56a957abbad9e4d0cb64029725)
- Update pyprecice API calls to their newer variants https://github.com/precice/micro-manager/pull/51

## v0.3.0

- Add global variant to adaptivity (still experimental) https://github.com/precice/micro-manager/pull/42
- Add norm-based (L1 and L2) support for functions in similarity distance calculation with absolute and relative variants https://github.com/precice/micro-manager/pull/40
- New domain decomposition strategy based on user input of number of processors along each axis https://github.com/precice/micro-manager/pull/41
- Add pickling support for C++ solver dummy https://github.com/precice/micro-manager/pull/30
- Add C++ solver dummy to show how a C++ micro simulation can be controlled by the Micro Manager https://github.com/precice/micro-manager/pull/22
- Add local adaptivity https://github.com/precice/micro-manager/pull/21

## v0.2.1

- Fixing the broken action workflow `run-macro-micro-dummy`

## v0.2.0

- Change package from `micro-manager` to `micro-manager-precice` and upload to PyPI.

## v0.2.0rc1

- Change package from `micro-manager` to `micro-manager-precice`.

## v0.1.0

- First release of Micro Manager prototype. Important features: Micro Manager can run in parallel, capability to handle bi-directional implicit coupling
