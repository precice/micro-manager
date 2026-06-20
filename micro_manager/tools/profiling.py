import precice as p


class ProfilingContext:
    """
    Provides an automated context in which profiling is performed by preCICE.
    When the context is entered profiling is started.
    Upon exiting it, profiling is stopped.
    Only one context may be active at a time. This is checked during debug mode.
    """

    CONTEXT_ACTIVE: bool = False

    def __init__(self, participant: p.Participant, name: str):
        """
        Creates a new ProfilingContext instance.

        Parameters
        ----------
        participant : p.Participant
            preCICE participant
        name : str
            profiling name
        """
        self._participant: p.Participant = participant
        self._name: str = name

    def __enter__(self) -> "ProfilingContext":
        """
        Called when entering the context in:
            with obj: ...
        """
        assert not self.CONTEXT_ACTIVE
        self.CONTEXT_ACTIVE = True

        self._participant.start_profiling_section(self._name)
        return self

    def __exit__(self, *args) -> None:
        """
        Called when exiting the context in:
           with obj: ...
        """
        self._participant.stop_last_profiling_section()
        self.CONTEXT_ACTIVE = False


class Profiler:
    """
    Provides methods to perform profiling with preCICE.
    """

    def __init__(self, participant: p.Participant):
        """
        Creates a new Profiler instance.

        Parameters
        ----------
        participant : p.Participant
            preCICE participant
        """
        self._participant: p.Participant = participant

    def measure(self, name: str) -> ProfilingContext:
        """
        Creates an automated profiling context. Usage:

        with profiler.measure("name"):
            ... calls to be measured ...

        ... other code ...

        Parameters
        ----------
        name : str
           profiling name

        Returns
        -------
        ctx : ProfilingContext
            profiling context
        """
        return ProfilingContext(self._participant, name)

    def begin(self, name: str) -> None:
        """
        Begins manual profiling. Must be ended with a corresponding call to end.

        Parameters
        ----------
        name : str
            profiling name
        """
        assert not ProfilingContext.CONTEXT_ACTIVE
        self._participant.start_profiling_section(name)

    def end(self):
        """
        Ends manual profiling.
        """
        self._participant.stop_last_profiling_section()
        ProfilingContext.CONTEXT_ACTIVE = False
