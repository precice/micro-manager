// Micro simulation
// In this file we solve a dummy micro problem which is controlled by the Micro Manager
// This dummy is written in C++ and is controllable via Python using pybind11
//
// Compile your pybind-11 wrapped code with:
//
// c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) micro_cpp_dummy.cpp -o micro_dummy$(python3-config --extension-suffix)
//
// To check if python is able to import it, run: 
// python3 -c "import micro_dummy; micro_dummy.MicroSimulation(1)"
// from the same directory

#include "micro_cpp_dummy.hpp"

// Constructor
MicroSimulation::MicroSimulation(int sim_id) : _sim_id(sim_id), _micro_scalar_data(0), _checkpoint(0) {}

// Initialize
void MicroSimulation::initialize()
{
    std::cout << "Initialize micro problem (" << _sim_id << ")\n";
    _micro_scalar_data = 0;
    _micro_vector_data.clear();
    _checkpoint = 0;
}

// Solve
py::dict MicroSimulation::solve(py::dict macro_data, double dt)
{
    std::cout << "Solve timestep of micro problem (" << _sim_id << ")\n";

    //! Code below shows how to convert input macro data and use it in your C++ solver

    // Create a double from macro_data["micro_scalar_data"], which is a Python float
    double macro_scalar_data = macro_data["macro-scalar-data"].cast<double>();

    // Create a pybind style Numpy array from macro_write_data["micro_vector_data"], which is a Numpy array
    py::array_t<double> macro_vector_data = macro_data["macro-vector-data"].cast<py::array_t<double>>();
    _micro_vector_data = std::vector<double>(macro_vector_data.data(), macro_vector_data.data() + macro_vector_data.size()); // convert numpy array to std::vector.

    // Change data
    _micro_scalar_data = macro_scalar_data + 1.;
    for (uint i = 0; i < _micro_vector_data.size(); i++)
    {
        _micro_vector_data[i] += 1.;
    }

    // Convert data to a py::dict again to send it back to the Micro Manager
    py::dict micro_write_data;

    // add micro_scalar_data and micro_vector_data to micro_write_data
    micro_write_data["micro-scalar-data"] = _micro_scalar_data;
    micro_write_data["micro-vector-data"] = _micro_vector_data; // numpy array is automatically converted to python list

    return micro_write_data;
}

// Save Checkpoint -- only valid for implicit coupling
void MicroSimulation::save_checkpoint()
{
    std::cout << "Saving state of micro problem (" << _sim_id << ")\n";
    _checkpoint = _micro_scalar_data;
}

// Reload Checkpoint -- only valid for implicit coupling
void MicroSimulation::reload_checkpoint()
{
    std::cout << "Reverting to old state of micro problem (" << _sim_id << ")\n";
    _micro_scalar_data = _checkpoint;
}

PYBIND11_MODULE(micro_dummy, m) {
    // optional docstring
    m.doc() = "pybind11 micro dummy plugin";

    py::class_<MicroSimulation>(m, "MicroSimulation")
        .def(py::init<int>())
        .def("initialize", &MicroSimulation::initialize)
        .def("solve", &MicroSimulation::solve)
        .def("save_checkpoint", &MicroSimulation::save_checkpoint)
        .def("reload_checkpoint", &MicroSimulation::reload_checkpoint);
}
