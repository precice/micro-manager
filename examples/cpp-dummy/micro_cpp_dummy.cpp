// Micro simulation
// In this script we solve a dummy micro problem to show how to adjust the macro-micro coupling
// This dummy is written in C++ and is bound to python using pybind11
#include <iostream>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h> // numpy arrays
#include <pybind11/stl.h> // std::vector conversion

namespace py = pybind11;

class MicroSimulation
{
public:
    MicroSimulation(int sim_id);
    void initialize();
    // solve takes python dict for macro_write data, dt, and returns python dict for macro_read data
    py::dict solve(py::dict macro_write_data, double dt);
    void save_checkpoint();
    void reload_checkpoint();
    int get_dims();

private:
    int _sim_id;
    int _dims;
    double _micro_scalar_data;
    std::vector<double> _micro_vector_data;
    double _checkpoint;
};

// Constructor
MicroSimulation::MicroSimulation(int sim_id) : _sim_id(sim_id), _dims(3), _micro_scalar_data(0), _checkpoint(0) {}

// Initialize
void MicroSimulation::initialize()
{
    std::cout << "Initialize micro problem (" << _sim_id << ")\n";
    _micro_scalar_data = 0;
    _micro_vector_data.clear();
    _checkpoint = 0;
}

// Solve
py::dict MicroSimulation::solve(py::dict macro_write_data, double dt)
{
    std::cout << "Solve timestep of micro problem (" << _sim_id << ")\n";

    // assert(dt != 0);
    if (dt == 0)
    {
        std::cout << "dt is zero\n";
        exit(1);
    }

    //! Here, insert your code, changing the data and casting it to the correct type
    // create double variable from macro_write_data["micro_scalar_data"]; which is a python float
    double macro_scalar_data = macro_write_data["macro-scalar-data"].cast<double>();
    // macro_write_data["micro_vector_data"] is a numpy array
    py::array_t<double> macro_vector_data = macro_write_data["macro-vector-data"].cast<py::array_t<double>>(); // doc on numpy arrays: https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html#arrays
    _micro_vector_data = std::vector<double>(macro_vector_data.data(), macro_vector_data.data() + macro_vector_data.size()); // convert numpy array to std::vector. directly casting to std::vector does not work?

    // micro_scalar_data and micro_vector_data are writedata+1
    _micro_scalar_data = macro_scalar_data + 1.;
    for (uint i = 0; i < _micro_vector_data.size(); i++)
    {
        _micro_vector_data[i] += 1.;
    }

    // create python dict for micro_write_data
    py::dict micro_write_data;
    // add micro_scalar_data and micro_vector_data to micro_write_data
    micro_write_data["micro-scalar-data"] = _micro_scalar_data;
    micro_write_data["micro-vector-data"] = _micro_vector_data; // numpy array is automatically converted to python list

    // return micro_write_data
    return micro_write_data;
}
// Save Checkpoint
void MicroSimulation::save_checkpoint()
{
    std::cout << "Saving state of micro problem (" << _sim_id << ")\n";
    _checkpoint = _micro_scalar_data;
}

// Reload Checkpoint
void MicroSimulation::reload_checkpoint()
{
    std::cout << "Reverting to old state of micro problem (" << _sim_id << ")\n";
    _micro_scalar_data = _checkpoint;
}

int MicroSimulation::get_dims()
{
    return _dims;
}

PYBIND11_MODULE(micro_dummy, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    py::class_<MicroSimulation>(m, "MicroSimulation")
        .def(py::init<int>())
        .def("initialize", &MicroSimulation::initialize)
        .def("solve", &MicroSimulation::solve)
        .def("save_checkpoint", &MicroSimulation::save_checkpoint)
        .def("reload_checkpoint", &MicroSimulation::reload_checkpoint)
        .def("get_dims", &MicroSimulation::get_dims);
}

// compile with
// c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) micro_cpp_dummy.cpp -o micro_cpp_dummy$(python3-config --extension-suffix)
// then from the same directory run python3 -c "import micro_dummy; micro_dummy.MicroSimulation(1)"