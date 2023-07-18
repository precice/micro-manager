# Tests

This folder contains everything needed for testing. The tests are split into two categories:

* `unit` contains unit tests that only check independent functions and modules.
* `integration` contains an integration test which uses preCICE and a Micro Manager.

## Unit tests

The unit tests can be run with [unittest](https://docs.python.org/3/library/unittest.html). For example, the tests for domain decomposition can be run with `python -m unittest test_domain_decomposition.py`.

The tests in `test_adaptivity_parallel.py` are designed to be run with 2 MPI processes. This can be run in the following way: `mpiexec -n 2 python -m unittest test_adaptivity_parallel.py`.

## Integration test

The integration test is a macro-micro case where the macro simulation is a unit cube. The micro simulation is a dummy which increments the data received from preCICE.
