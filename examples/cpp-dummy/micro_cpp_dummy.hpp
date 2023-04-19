// This is the header file for the micro simulation class.
// It is included in the micro_cpp_dummy.cpp file and the micro_cpp_dummy.cpp file is compiled with pybind11 to create a python module.
// The python module is then imported in the Micro Manager.

#pragma once
#include <iostream>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h> // numpy arrays
#include <pybind11/stl.h>   // std::vector conversion

namespace py = pybind11;

class MicroSimulation
{
public:
    MicroSimulation();
    void initialize();
    // solve takes a python dict data, and the timestep dt as inputs, and returns a python dict
    py::dict solve(py::dict macro_write_data, double dt);
    void save_checkpoint();
    void reload_checkpoint();
    MicroSimulation __deepcopy__(py::dict memo);

    void setState(double micro_scalar_data, double checkpoint);
    py::tuple getState() const;

private:
    double _micro_scalar_data;
    std::vector<double> _micro_vector_data;
    double _checkpoint;
};
