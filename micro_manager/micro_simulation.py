"""
Functionality to create MicroSimulation class objects which inherit from user provided base_micro_simulation class.
"""


def create_micro_problem_class(base_micro_simulation):
    """
    Creates a class MicroSimulation which inherits from the class of the micro simulation.

    Parameters
    ----------
    base_micro_simulation : class
        The base class from the micro simulation script.

    Returns
    -------
    MicroSimulation : class
        Definition of class MicroSimulation defined in this function.
    """
    class MicroSimulation(base_micro_simulation):
        def __init__(self, local_id, global_id):
            base_micro_simulation.__init__(self, local_id)
            self._local_id = local_id
            self._global_id = global_id
            self._is_active = False  # Simulation is created in an inactive state

            # Only defined when simulation is inactive
            self._associated_active_local_id = None
            self._associated_active_global_id = None

            # Only defined when simulation is active
            self._associated_inactive_local_ids = None
            self._associated_inactive_global_ids = None

        def get_local_id(self) -> int:
            return self._local_id

        def get_global_id(self) -> int:
            return self._global_id

        def set_local_id(self, local_id) -> None:
            self._local_id = local_id

        def set_global_id(self, global_id) -> None:
            self._global_id = global_id

        def activate(self) -> None:
            self._is_active = True

        def deactivate(self) -> None:
            self._is_active = False

        def is_active(self) -> bool:
            return self._is_active

        def is_associated_to_active_sim(self, similar_active_local_id: int, similar_active_global_id: int) -> None:
            assert not self._is_active, "Micro simulation {} is active and hence cannot be associated to another active simulation".format(
                self._global_id)
            self._associated_active_local_id = similar_active_local_id
            self._associated_active_global_id = similar_active_global_id

        def get_associated_active_local_id(self) -> int:
            assert not self._is_active, "Micro simulation {} is active and hence cannot have an associated active local ID".format(
                self._global_id)
            return self._associated_active_local_id

        def get_associated_active_global_id(self) -> int:
            assert not self._is_active, "Micro simulation {} is active and hence cannot have an associated active global ID".format(
                self._global_id)
            return self._associated_active_global_id

        def is_associated_to_inactive_sim(self, similar_inactive_local_id: int,
                                          similar_inactive_global_id: int) -> None:
            assert self._is_active, "Micro simulation {} is inactive and hence cannot be associated to an inactive simulation".format(
                self._global_id)
            self._associated_inactive_local_ids.append(similar_inactive_local_id)
            self._associated_inactive_global_ids.append(similar_inactive_global_id)

        def is_associated_to_inactive_sims(self, similar_inactive_local_ids: list,
                                           similar_inactive_global_ids: list) -> None:
            assert self._is_active, "Micro simulation {} is inactive and hence cannot be associated to inactive simulations".format(
                self._global_id)
            self._associated_inactive_local_ids = similar_inactive_local_ids
            self._associated_inactive_global_ids = similar_inactive_global_ids

        def get_associated_inactive_local_id(self) -> int:
            assert self._is_active, "Micro simulation {} is inactive and hence cannot have an associated inactive local ID".format(
                self._global_id)
            return self._associated_inactive_local_ids[0]

        def get_associated_inactive_global_id(self) -> int:
            assert self._is_active, "Micro simulation {} is inactive and hence cannot have an associated inactive global ID".format(
                self._global_id)
            return self._associated_inactive_global_ids[0]

        def get_associated_inactive_local_ids(self) -> list:
            assert self._is_active, "Micro simulation {} is inactive and hence cannot have associated inactive local IDs".format(
                self._global_id)
            return self._associated_inactive_local_ids

        def get_associated_inactive_global_ids(self) -> list:
            assert self._is_active, "Micro simulation {} is active and hence cannot have associated inactive global IDs".format(
                self._global_id)
            return self._associated_inactive_global_ids

    return MicroSimulation
