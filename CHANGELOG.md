# Micro Manager changelog

## latest

- Fixed duplicate micro simulations for macro-points on rank boundaries by filtering coordinates already claimed by lower-ranked ranks [#230](https://github.com/precice/micro-manager/pull/230)
- Exposed `MicroSimulationInterface` as a public abstract base class for user subclassing [#224](https://github.com/precice/micro-manager/pull/224)
- Added option to use compute instances to reduce memory consumption [#226](https://github.com/precice/micro-manager/pull/226)
- Added support to run micro simulations in separate processes with workers [#219](https://github.com/precice/micro-manager/pull/219)
- Added abstraction layers to micro simulations to support more features [#218](https://github.com/precice/micro-manager/pull/218)

## v0.8.0

- Conformed to naming standard in precice/tutorials [#215](https://github.com/precice/micro-manager/pull/215)
- Changed default values of adaptivity metrics output and similarity distance calculation norm [#213](https://github.com/precice/micro-manager/pull/213)
- Fixed ordering of global IDs of micro simulation for load balancing to ensure consistency [#210](https://github.com/precice/micro-manager/pull/210)
- Initiated triggering of load balancing at the start of the simulation when adaptivity is triggered [#207](https://github.com/precice/micro-manager/pull/207)
- Added functionality to adaptively switch micro-scale models [#198](https://github.com/precice/micro-manager/pull/198)
- Changed locations of profiling sections in the code base to reflect operations being profiled correctly [#205](https://github.com/precice/micro-manager/pull/205)
- Added global metrics logging to local adaptivity [commit](https://github.com/precice/micro-manager/commit/1be626ff852a2e3e03a59d5859e5be0e1dd5d67a)
- A rank-based round-robin scheme for deactivating simulations in global adaptivity [#202](https://github.com/precice/micro-manager/pull/202)
- Added more metrics to adaptivity logging, and some general refactoring [#201](https://github.com/precice/micro-manager/pull/201)
- Fix bug in load balancing when a rank has exactly as many active simulation as the global average [#200](https://github.com/precice/micro-manager/pull/200)
- Use global maximum similarity distance in local adaptivity [#197](https://github.com/precice/micro-manager/pull/197)
- Log adaptivity metrics at t=0 [#194](https://github.com/precice/micro-manager/pull/194)
- Use `|` delimiter in CSV files of adaptivity metrics data [#193](https://github.com/precice/micro-manager/pull/193)
- Add profiling sections for solving the micro simulations [commit](https://github.com/precice/micro-manager/commit/fa81f6a7e8f494a3e441fee7f70de5271ae8d83b)
- Fix bug in load balancing when more than one rank has as many simulations as the balancing bound [#192](https://github.com/precice/micro-manager/pull/192)
- Remove two-step balancing from load balancing [#191](https://github.com/precice/micro-manager/pull/191)
- Remove option of simulation solve time from diagnostics [#188](https://github.com/precice/micro-manager/pull/188)
- Add dynamic load balancing capability to global adaptivity [#144](https://github.com/precice/micro-manager/pull/141)
- Revert to using (default) mpich as error is fixed in setup-mpi action [#187](https://github.com/precice/micro-manager/pull/187)
- Refactor adaptivity: simplify logic and shorten iterator variable names [#186](https://github.com/precice/micro-manager/pull/186)
- Use boolean parameter values in JSON config files of unit cube integration test [#185](https://github.com/precice/micro-manager/pull/185)
- Use OpenMPI in mpi4py setup action in parallel tests action [#184](https://github.com/precice/micro-manager/pull/184)
- Software improvements to the snapshot computation functionality [#178](https://github.com/precice/micro-manager/pull/178)

## v0.7.0

- Share similarity distance matrix between ranks on a compute node [#176](https://github.com/precice/micro-manager/pull/176)
- Use booleans instead of strings `"True"` and `"False"` in the configuration [#175](https://github.com/precice/micro-manager/pull/175)
- Renaming and using a newer workflow for publishing according to the trusted publishing in PyPI [#173](https://github.com/precice/micro-manager/pull/173)
- Add config options for adaptivity metrics and memory usage output to allow for different levels [#172](https://github.com/precice/micro-manager/pull/172)
- Fix bug in adaptivity computation when an active simulation with associations is deactivated [#171](https://github.com/precice/micro-manager/pull/171)
- Properly handle micro simulation initialization for lazy initialization [#169](https://github.com/precice/micro-manager/pull/169)
- Delete the simulation object when the simulation is deactivated [#167](https://github.com/precice/micro-manager/pull/167)
- Remove float32 data type restriction for adaptivity data [commit](https://github.com/precice/micro-manager/commit/bfa44ff4d3432c6ac0f3b1311274308d2ec9c2a4)
- Trigger adaptivity when all the adaptivity data is the same [#170](https://github.com/precice/micro-manager/pull/170)
- Add configuration option to control frequency of adaptivity computation [#168](https://github.com/precice/micro-manager/pull/168)
- Remove checkpointing of adaptivity and fix output of memory usage [#166](https://github.com/precice/micro-manager/pull/166)
- Performance improvements: restricting data types, in-place modifications [#162](https://github.com/precice/micro-manager/pull/162)
- Handle adaptivity case when deactivation and activation happens in the same time window [#165](https://github.com/precice/micro-manager/pull/165)
- Add command line input argument to set log file [#163](https://github.com/precice/micro-manager/pull/163)
- Fix adaptivity metrics logging and add logging documentation [#160](https://github.com/precice/micro-manager/pull/160)
- Checkpoint lazily created simulation only if a checkpoint is necessary [#161](https://github.com/precice/micro-manager/pull/161)

## v0.6.0

- Add functionality for lazy creation and initialization of micro simulations [#117](https://github.com/precice/micro-manager/pull/117)
- Improve logging wrapper function names to be more clear [#153](https://github.com/precice/micro-manager/pull/153)
- Remove adaptivity computation CPU time export functionality [#152](https://github.com/precice/micro-manager/pull/152)
- Replace `Allgatherv` with `allgather` to avoid running into the error of size buffer [#151](https://github.com/precice/micro-manager/pull/151)
- Update Actions workflows due to updates in `precice/precice:nightly` [#150](https://github.com/precice/micro-manager/pull/150)
- Move adaptivity CPU time output from preCICE export to metrics logging [#149](https://github.com/precice/micro-manager/pull/149)
- Fix bug in the domain decomposition which was returning incorrect bounding box limits for the decomposition of `[2, 2, 1]` and similar [#146](https://github.com/precice/micro-manager/pull/146)
- Fix bug in calling of the adaptivity computation for explicit coupling scenarios [#145](https://github.com/precice/micro-manager/pull/145)
- Fix bug in handling of vector data returned by the MicroSimulation `solve()` method, for scenarios with adaptivity [#143](https://github.com/precice/micro-manager/pull/143)
- Remove the `scalar` and `vector` keyword values from data names in configuration [#142](https://github.com/precice/micro-manager/pull/142)
- Set default logger to stdout and add output directory setting option for file loggers [#139](https://github.com/precice/micro-manager/pull/139)
- Remove the `adaptivity_data` data structure and handle all adaptivity data internally [#137](https://github.com/precice/micro-manager/pull/137)
- Improve logging by wrapping Python logger in a class [#133](https://github.com/precice/micro-manager/pull/133)
- Refactor large parts of solve and adaptivity to group datasets and simplify handling [#135](https://github.com/precice/micro-manager/pull/135)
- Add information about adaptivity tuning parameters [#131](https://github.com/precice/micro-manager/pull/131)
- Put computation of counting active steps inside the adaptivity variant `if` condition [#130](https://github.com/precice/micro-manager/pull/130)

## v0.5.0

- Use absolute values to calculate normalizing factor for relative norms in adaptivity [#125](https://github.com/precice/micro-manager/pull/125)
- Add option to use only one micro simulation object in the snapshot computation [#123](https://github.com/precice/micro-manager/pull/123)
- Explicitly check if time window has converged using the API function `is_time_window_complete()` [#118](https://github.com/precice/micro-manager/pull/118)
- Add `MicroManagerSnapshot` enabling snapshot computation and storage of microdata in HDF5 format [#101](https://github.com/precice/micro-manager/pull/101)
- Make `sklearn` an optional dependency
- Move the config variable `micro_dt` from the coupling parameters section to the simulation parameters section [#114](https://github.com/precice/micro-manager/pull/114)
- Set time step of micro simulation in the configuration, and use it in the coupling [#112](https://github.com/precice/micro-manager/pull/112)
- Add a base class called `MicroManager` with minimal API and member function definitions, rename the existing `MicroManager` class to `MicroManagerCoupling` [#111](https://github.com/precice/micro-manager/pull/111)
- Handle calling `initialize()` function of micro simulations written in languages other than Python [#110](https://github.com/precice/micro-manager/pull/110)
- Check if initial data returned from the micro simulation is the data that the adaptivity computation requires [#109](https://github.com/precice/micro-manager/pull/109)
- Use executable `micro-manager-precice` by default, and stop using the script `run_micro_manager.py` [#105](https://github.com/precice/micro-manager/pull/105)
- Make `initialize()` method of the MicroManager class public [#105](https://github.com/precice/micro-manager/pull/105)
- Optionally use initial macro data to initialize micro simulations [#104](https://github.com/precice/micro-manager/pull/104)
- Use `pyproject.toml` instead of `setup.py` to configure the build. Package name is now `micro_manager_precice` [#84](https://github.com/precice/micro-manager/pull/84)
- Add handling of crashing micro simulations [#85](https://github.com/precice/micro-manager/pull/85)
- Add switch to turn adaptivity on and off in configuration [#93](https://github.com/precice/micro-manager/pull/93)

## v0.4.0

- Add note in the cpp-dummy that pickling support does not work due to no good way to pass the sim id to the new micro simulation instance [commit](https://github.com/precice/micro-manager/commit/0a82966676717a533aca9bffa4a110453158f29c)
- Reintroduce initialize function in the micro simulation API [#79](https://github.com/precice/micro-manager/pull/79)
- Use Allgatherv instead of allgather when collecting number of micro simulations on each rank in initialization [#81](https://github.com/precice/micro-manager/pull/81)
- Remove the callable function `initialize()` from the micro simulation API [commit](https://github.com/precice/micro-manager/commit/bed5a4cc0f03b780da7f62b3f51ed1df2796588c)
- Pass an ID to the micro simulation object so that it is aware of its own uniqueness [#66](https://github.com/precice/micro-manager/pull/66)
- Resolve bug which led to an error when global adaptivity was used with unequal number of simulations on each rank [#78](https://github.com/precice/micro-manager/pull/78)
- Make the `initialize()` method of the MicroManager class private [#77](https://github.com/precice/micro-manager/pull/77)
- Add reference paper via a CITATION.cff file [commit](https://github.com/precice/micro-manager/commit/6c08889c658c889d6ab5d0867802522585abcee5)
- Add JOSS DOI badge [commit](https://github.com/precice/micro-manager/commit/2e3c2a4c77732f56a957abbad9e4d0cb64029725)
- Update pyprecice API calls to their newer variants [#51](https://github.com/precice/micro-manager/pull/51)

## v0.3.0

- Add global variant to adaptivity (still experimental) [#42](https://github.com/precice/micro-manager/pull/42)
- Add norm-based (L1 and L2) support for functions in similarity distance calculation with absolute and relative variants [#40](https://github.com/precice/micro-manager/pull/40)
- New domain decomposition strategy based on user input of number of processors along each axis [#41](https://github.com/precice/micro-manager/pull/41)
- Add pickling support for C++ solver dummy [#30](https://github.com/precice/micro-manager/pull/30)
- Add C++ solver dummy to show how a C++ micro simulation can be controlled by the Micro Manager [#22](https://github.com/precice/micro-manager/pull/22)
- Add local adaptivity [#21](https://github.com/precice/micro-manager/pull/21)

## v0.2.1

- Fixing the broken action workflow `run-macro-micro-dummy`

## v0.2.0

- Change package from `micro-manager` to `micro-manager-precice` and upload to PyPI.

## v0.2.0rc1

- Change package from `micro-manager` to `micro-manager-precice`.

## v0.1.0

- First release of Micro Manager prototype. Important features: Micro Manager can run in parallel, capability to handle bi-directional implicit coupling
