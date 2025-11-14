---
title: Adaptive switching of simulation models
permalink: tooling-micro-manager-model-adaptivity.html
keywords: tooling, macro-micro, two-scale, model-adaptivity
summary: Micro Manager can adaptively switch models of micro simulations.
---

## Main Concept

For scenarios such as FE2 simulations computing all or perhaps only active micro simulations
may be still computationally infeasible. The alternative here is to reduce model complexity via reduced order models (ROMs) or similar.
Towards this the model adaptivity feature allows for the definition of multiple
model fidelities and the switching between those at run-time.

### Iterative Process

**Without** model adaptivity the call to `micro_sim_solve(micro_sims_input, dt)` within the micro_manager would just
provide each micro simulation with its input, solve it and return the output.

**With** model adaptivity this becomes an iterative process, as a model may not be sufficiently accurate (given the current input).
Thus, the call to `micro_sim_solve(micro_sims_input, dt)` with model adaptivity results in the following:

```python
self._model_adaptivity_controller.initialise_solve()

active_sim_ids = None
if self._is_adaptivity_on:
    active_sim_ids = self._adaptivity_controller.get_active_sim_local_ids()
output = None

while self._model_adaptivity_controller.should_iterate():
    self._model_adaptivity_controller.switch_models(
        self._mesh_vertex_coords,
        self._t,
        micro_sims_input,
        output,
        self._micro_sims,
        active_sim_ids,
    )
    output = solve_variant(micro_sims_input, dt)
    self._model_adaptivity_controller.check_convergence(
        self._mesh_vertex_coords,
        self._t,
        micro_sims_input,
        output,
        self._micro_sims,
        active_sim_ids,
    )

self._model_adaptivity_controller.finalise_solve()
return output
```

Here, after initialization and active sim acquisition, models will be switched, evaluated and checked for convergence
as long as the `switching_function` contains values other than 0.
Model evaluation - in the call `solve_variant(micro_sims_input, dt)` - is delegated to the regular
(non-model-adaptive) `micro_sim_solve(micro_sims_input, dt)` method.

### Interfaces

```python
class MicroSimulation: # Name is fixed
    def __init__(self, sim_id):
        """
        Constructor of class MicroSimulation.

        Parameters
        ----------
        sim_id : int
            ID of the simulation instance, that the Micro Manager has set for it.
        """

    def initialize(self) -> dict:
        """
        Initialize the micro simulation and return initial data which will be used in computing adaptivity before the first time step.

        Defining this function is OPTIONAL.

        Returns
        -------
        initial_data : dict
            Dictionary with names of initial data as keys and the initial data itself as values.
        """

    def solve(self, macro_data: dict, dt: float) -> dict:
        """
        Solve one time step of the micro simulation for transient problems or solve until steady state for steady-state problems.

        Parameters
        ----------
        macro_data : dict
            Dictionary with names of macro data as keys and the data as values.
        dt : float
            Current time step size.

        Returns
        -------
        micro_data : dict
            Dictionary with names of micro data as keys and the updated micro data a values.
        """

    def set_state(self, state):
        """
        Set the state of the micro simulation.
        """

    def get_state(self):
        """
        Return the state of the micro simulation.
        """

    def output(self):
        """
        This function writes output of the micro simulation in some form.
        It will be called with frequency set by configuration option `simulation_params: micro_output_n`
        This function is *optional*.
        """
```

For this the default MicroSimulation still serves as the model interface, while the `(set)|(get)_state()` methods
are called to transfer internal model parameters from one to another.
The list of provided models is interpreted in decreasing fidelity order. In other words, the first one
is likely to be the full order model, while subsequent ones are ROMs.

```python
def switching_function(
    resolutions: np.ndarray,
    locations: np.ndarray,
    t: float,
    inputs: list[dict],
    prev_output: dict,
    active: np.ndarray,
) -> np.ndarray:
    """
    Switching interface function, use as reference

    Parameters
    ----------
    resolutions : np.array - shape(N,)
        Array with resolution information as get_sim_class_resolution would return for a sim obj.
    locations : np.array - shape(N,D)
        Array with gaussian points for all sims. D is the mesh dimension.
    t : float
        Current time in simulation.
    inputs : list[dict]
        List of input objects.
    prev_output : [None, dict-like]
        Contains the outputs of the previous model evaluation.
    active : np.array - shape(N,)
        Bool array indicating whether the model is active or not.

    """
    return np.zeros_like(resolutions)
```

The switching of models is governed by the `switching_function`.
The output is expected to be a np.ndarray of shape (N,) and is interpreted in the following manner:

Value | Action
--- | ---
0 | No resolution change
-1 | Increase model fidelity by one (go back one in list)
1 | Decrease model fidelity by one (go one ahead in list)
