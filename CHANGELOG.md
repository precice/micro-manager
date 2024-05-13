# Micro Manager changelog

## latest

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
