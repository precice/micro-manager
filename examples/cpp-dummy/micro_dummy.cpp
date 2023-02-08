// Micro simulation
// In this script we solve a dummy micro problem to just show the working of the macro-micro coupling
#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>

class MicroSimulation
{
public:
    MicroSimulation(int sim_id);
    void initialize();
    void solve(double *macro_write_scalar_data, double* macro_write_vector_data, double dt, double *macro_read_scalar_data, double* macro_read_vector_data);
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
void MicroSimulation::solve(double *macro_write_scalar_data, double* macro_write_vector_data, double dt, double *macro_read_scalar_data, double* macro_read_vector_data)
{
    std::cout << "Solve timestep of micro problem (" << _sim_id << ")\n";
    // assert(dt != 0);
    if (dt == 0)
    {
        std::cout << "dt is zero\n";
        exit(1);
    }
    
    _micro_scalar_data = *macro_write_scalar_data;
    _micro_vector_data.clear();
    for (int d = 0; d < _dims; d++)
    {
        _micro_vector_data.push_back(macro_write_vector_data[d]);
    }

    macro_write_scalar_data = &_micro_scalar_data;
    macro_write_vector_data = &_micro_vector_data[0];
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

extern "C" {
    MicroSimulation* MicroSimulation_new(int sim_id) {
        return new MicroSimulation(sim_id);
    }

    void MicroSimulation_initialize(MicroSimulation* microsim) {
        microsim->initialize();
    }

    void MicroSimulation_solve(MicroSimulation* microsim, double* macro_read_scalar_data, double* macro_read_vector_data, double dt, double* macro_write_scalar_data, double* macro_write_vector_data) 
    {
        std::cout << "Macro read scalar data " << *macro_read_scalar_data << "\n";
        std::cout << "Macro read vector data " << *macro_read_vector_data << "\n";
        std::cout << "Solve timestep " << dt << "\n";
        std::cout << "Macro write scalar data " << *macro_write_scalar_data << "\n";
        std::cout << "Macro write vector data " << *macro_write_vector_data << "\n";

        // create std::vector from macro_vector_data

        microsim->solve(macro_write_scalar_data, macro_write_vector_data, dt, macro_read_scalar_data, macro_read_vector_data);
    }

    void MicroSimulation_save_checkpoint(MicroSimulation* microsim) {
        microsim->save_checkpoint();
    }

    void MicroSimulation_reload_checkpoint(MicroSimulation* microsim) {
        microsim->reload_checkpoint();
    }
}

// compile with g++ -std=c++11 -shared -fPIC micro_dummy.cpp -o micro_dummy.so