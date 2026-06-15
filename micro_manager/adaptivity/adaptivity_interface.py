from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union

from micro_manager.tools.logging_wrapper import Logger


class AdaptivityInterface(ABC):
    def compute(self, dt: float) -> None:
        """
        Performs the internal adaptivity computation.
        Will be called by compute_step, if the computation criteria are met.

        Parameters
        ----------
        dt : float
            Delta time within current time window.
        """
        return

    def compute_step(self, n: int, first_iteration: bool, dt: float) -> None:
        """
        Main driving method of adaptivity.
        Checks the provided parameters whether adaptivity should be performed.
        Calls compute.

        Parameters
        ----------
        n : int
            Current time step number.
        first_iteration : bool
            True if this is the first iteration of the current time window.
        dt : float
            Delta time within current time window.
        """
        return

    def postprocess_active_output(self, micro_output: Dict[str, Any], gid: int) -> None:
        """
        Attached adaptivity data to the given micro simulation output.

        Parameters
        ----------
        micro_output : Dict[str, Any]
            Micro simulation output. (active simulation)
        gid : int
            GID of the corresponding micro simulation.
        """
        return

    def postprocess_inactive_output(
        self, micro_output: Dict[str, Any], gid: int
    ) -> None:
        """
        Attached adaptivity data to the given micro simulation output.

        Parameters
        ----------
        micro_output : Dict[str, Any]
            Micro simulation output. (inactive simulation)
        gid : int
            GID of the corresponding micro simulation.
        """
        return

    def postprocess_remove(self, micro_output: Dict[str, Any]) -> None:
        """
        Removes any attached adaptivity data from the given micro simulation output.

        Parameters
        ----------
        micro_output : Dict[str, Any]
            Micro simulation output.
        """
        return

    def log_metrics(self, n: int) -> None:
        """
        Optionally logs the configured adaptivity metrics for the provided time step number.

        Parameters
        ----------
        n : int
            Current time step number.
        """
        return

    def update_buffers(
        self,
        micro_data: Optional[Union[List[Dict[str, Any]], Dict[str, List[Any]]]] = None,
        macro_data: Optional[Union[List[Dict[str, Any]], Dict[str, List[Any]]]] = None,
        invert: bool = False,
        alloc: bool = False,
    ) -> None:
        """
        Updates the rank local computation buffer with the provided data.
        If alloc is True, then the rank local buffers are reallocated to the current
        simulation container size. micro_data and macro_data can be either provided
        as lists of data dictionaries or as a dictionary of data lists.
        In case of the latter, invert must be set to True.

        Parameters
        ----------
        micro_data : Optional[Union[List[Dict[str, Any]], Dict[str, List[Any]]]]
            Micro simulation output data.
        macro_data : Optional[Union[List[Dict[str, Any]], Dict[str, List[Any]]]]
            Micro simulation input data.
        invert : bool
            If True, then the expected data format is a dictionary of lists.
        alloc : bool
            If True, reallocates rank local data buffers.
        """
        return

    def get_read_buffer(self) -> Optional[Dict[str, List[Any]]]:
        """
        If required by the underlying implementation, returns a buffer into which
        the CouplingHandler can write the read data directly into.

        Returns
        -------
        read_buffer : Optional[Dict[str, List[Any]]]
            Read buffer for CouplingHandler. (can be None)
        """
        return None

    def get_macro_data_names(self) -> Optional[List[str]]:
        """
        Gets the relevant data names for adaptivity from the micro simulation inputs.

        Returns
        -------
        macro_data_names : Optional[List[str]]
            Micro simulation input data names.
        """
        return None

    def get_micro_data_names(self) -> Optional[List[str]]:
        """
        Gets the relevant data names for adaptivity from the micro simulation outputs.

        Returns
        -------
        micro_data_names : Optional[List[str]]
            Micro simulation output data names.
        """
        return None

    @abstractmethod
    def get_active_steps(self) -> Dict[int, int]:
        """
        Gets a map from GIDs to their number of active simulation steps.

        Returns
        -------
        active_steps : Dict[int, int]
            Map from GIDs to their number of active simulation steps.
        """
        ...

    @abstractmethod
    def get_active_lids(self) -> List[int]:
        """
        Gets a list of the (rank local) active LIDs.

        Returns
        -------
        active_lids : List[int]
            List of the (rank local) active LIDs.
        """
        ...

    @abstractmethod
    def get_inactive_lids(self) -> List[int]:
        """
        Gets a list of the (rank local) inactive LIDs.

        Returns
        -------
        inactive_lids : List[int]
            List of the (rank local) inactive LIDs.
        """
        ...

    @abstractmethod
    def get_active_gids(self) -> List[int]:
        """
        Gets a list of the (rank local) active GIDs.

        Returns
        -------
        active_gids : List[int]
            List of the (rank local) active GIDs.
        """
        ...

    @abstractmethod
    def get_inactive_gids(self) -> List[int]:
        """
        Gets a list of the (rank local) inactive GIDs.

        Returns
        -------
        inactive_gids : List[int]
            List of the (rank local) inactive GIDs.
        """
        ...

    @abstractmethod
    def get_full_field_micro_output(
        self,
        micro_input: List[Dict[str, Any]],
        micro_output: List[Optional[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """
        Constructs the simulation outputs for all inactive simulations (given by missing output data).

        Parameters
        ----------
        micro_input : List[Dict[str, Any]]
            Micro simulation input data.
        micro_output : List[Optional[Dict[str, Any]]]
            Micro simulation output data buffer. Active simulation outputs are available, inactive ones are None.

        Returns
        -------
        full_field_micro_output : List[Dict[str, Any]]
             Fully populated simulation output data.
        """
        ...

    @abstractmethod
    def get_adaptivity_buffer(self) -> Dict[str, List[Any]]:
        """
        Gets the rank local data buffer used for adaptivity computation.

        Returns
        -------
        adaptivity_buffer : Dict[str, List[Any]]
            Keys are data names, values are rank local data buffers.
        """
        ...

    @abstractmethod
    def get_associated_map(self) -> Dict[int, int]:
        """
        Gets an association map of inactive simulations to active simulations given by IDs.
        For LocalAdaptivity the IDs are LIDs. For GlobalAdaptivity the IDs are GIDs.

        Returns
        -------
        associated_map : Dict[int, int]
            Keys are inactive IDs, values are active IDs.
        """
        ...

    def check_micro_simulation_initialize(
        self, logger: Logger, micro_init_return_value: Optional[Dict[str, Any]]
    ) -> None:
        """
        Performs an initial check on the keys of the micro simulation output. (used during initialization)

        Parameters
        ----------
        logger : Logger
            Logger object.
        micro_init_return_value : Optional[Dict[str, Any]]
            Optional micro simulation return value.
        """
        if micro_init_return_value is None:
            return

        macro_names = self.get_macro_data_names()
        micro_names = self.get_micro_data_names()
        if macro_names is None or micro_names is None:
            return

        # Check for missing data
        expected = set(micro_names)
        provided = set(micro_init_return_value.keys())
        if missing := expected - provided:
            raise Exception(
                "The initialize() method needs to return data which is required for the adaptivity calculation. "
                f'Of the expected data {", ".join(expected)}, the following is missing: {", ".join(missing)}'
            )
        elif extra := provided - set(macro_names + micro_names):
            logger.log_warning_rank_zero(
                f'The initialize() method of the Micro simulation returns extra initial data which isn\'t used by the adaptivity: {", ".join(extra)}'
            )
