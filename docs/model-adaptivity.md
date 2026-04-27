---
title: Adaptive switching of simulation models
permalink: tooling-micro-manager-model-adaptivity.html
keywords: tooling, macro-micro, two-scale, model-adaptivity
summary: Micro Manager can adaptively switch models of micro simulations.
---

## Main Concept

For certain multiscale scenarios, having an adaptivity strategy that groups the micro simulations into active and inactive simulations
may be insufficient. Alternatively, a hierarchy of micro-scale models, for example, reduced order models (ROMs) can be used.
The model adaptivity functionality allows for the definition of multiple
model fidelities and the switching between them at run-time.

### Iterative Process

**Without** model adaptivity, the Micro Manager calls the `solve(micro_sims_input, dt)` routine of all active simulations
and copies their output to their closest similar inactive counterparts.

**With** model adaptivity, there is an iterative process, because a model may not be sufficiently accurate (given the current input).
The call to `solve(micro_sims_input, dt)` leads to the following logic:

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

    def get_global_id(self):
        """
        Return the assigned global id.
        """
```

For this the default MicroSimulation still serves as the model interface, while the `(set)|(get)_state()` methods
are called to transfer internal model parameters from one to another.
The list of provided models is interpreted in decreasing fidelity order. In other words, the first one
is likely to be the full order model, while subsequent ones are ROMs.

```python
def switching_function(
    resolution: int,
    location: np.ndarray,
    t: float,
    input: dict,
    prev_output: dict,
) -> int:
    """
    Switching interface function, use as reference

    Parameters
    ----------
    resolution : int
        Current resolution as get_sim_class_resolution would return for the respective sim obj.
    location : np.array - shape(D,)
        Array with gaussian points.
    t : float
        Current time in simulation.
    input : dict
        Input object.
    prev_output : [None, dict]
        Contains the outputs of the previous model evaluation.

    Returns
    -------
    switch_direction: int
        0 if resolution should not change
        -1 if resolution should increase
        1 if resolution should decrease
    """
    return 0
```

The switching of models is governed by the `switching_function`, which is evaluated for each micro simulation.
The output is expected to be an integer and is interpreted in the following manner:

| Value | Action                                                |
|-------|-------------------------------------------------------|
| 0     | No resolution change                                  |
| -1    | Increase model fidelity by one (go back one in list)  |
| 1     | Decrease model fidelity by one (go one ahead in list) |

If the switching function requests a change beyond the available resolution range, the request is ignored and does not trigger another model-adaptivity iteration.
