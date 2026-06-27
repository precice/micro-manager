import csv
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

from psutil import Process
import numpy as np

from micro_manager.tools.mpi_handler import MPIHandler
from micro_manager.tools.logging_wrapper import Logger
from micro_manager.config import Config


class DiagnosticModule(ABC):
    """
    Interface for all diagnostic modules.
    """
    def __init__(self, logger: Logger, mpi: MPIHandler, output_dir: str):
        """
        Base Constructor.

        Parameters
        ----------
        logger : Logger
            Logging object.
        mpi : MPIHandler
            MPIHandler object.
        output_dir : str
            Output directory.
        """
        self._logger : Logger = logger
        self._mpi : MPIHandler = mpi
        self._output_dir = output_dir

    @abstractmethod
    def log(self, n: int, force_log: bool) -> None:
        """
        Request logging of the tracked statistic at timestep n.
        The implementation may not do so due to its configuration.
        By setting force_log to True, this may be circumvented.

        Parameters
        ----------
        n : int
            Timestep
        force_log : bool

        """
        pass

    @abstractmethod
    def output(self) -> None:
        """
        Outputs the tracked statistic.
        """
        pass


class MemoryUsage(DiagnosticModule):
    def __init__(
            self,
            logger: Logger,
            mpi: MPIHandler,
            output_dir: str,
            is_global: bool,
            output_n: int,
    ):
        """
        Constructs a MemoryUsage object.
        With is_global either local or global memory usage is tracked.

        Parameters
        ----------
        logger : Logger
            Logging object.
        mpi : MPIHandler
            MPIHandler object.
        output_dir : str
            Output directory.
        is_global : bool
            Track global memory usage.
        output_n : int
            Frequency of memory usage tracking.
        """
        super().__init__(logger, mpi, output_dir)

        self._process : Process = Process()
        self._is_global : bool = is_global
        self._output_n : int = output_n
        self._output_file : str = (
            f"{output_dir}global_avg_peak_mem_usage.csv" if is_global
            else f"{output_dir}peak_mem_usage_{self._mpi.rank}.csv"
        )
        self._mem_usage : List[Tuple[int, int]] = []

    def log(self, n: int, force_log: bool) -> None:
        should_log = n % self._output_n == 0 or n == 1
        if not force_log and not should_log:
            return
        if force_log and n % self._output_n == 0:
            return

        self._mem_usage.append(
            (n, self._process.memory_info().rss / 1024 ** 2)
        )

    def output(self) -> None:
        output_data: Optional[List[Tuple[int, int]]] = self._collect_usage(self._mem_usage)
        if output_data is None:
            return

        with open(self._output_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Time window", "RSS (MB)"])
            for n, rss_mb in output_data:
                writer.writerow([n, rss_mb])


    def _collect_usage(self, output_data: List[Tuple[int, int]]) -> Optional[List[Tuple[int, int]]]:
        """
        Gathers the data to be written into CSV file.
        For local memory usage tracking, the rank local data is passed through.
        For global memory usage tracking, global averages are collected. Only rank 0 will receive data.

        Parameters
        ----------
        output_data : List[Tuple[int, int]]
            rank local memory usage data

        Returns
        -------
        write_data : Optional[List[Tuple[int, int]]]
            output_data if not global, else if rank 0 global write data, else None
        """
        if not self._is_global:
            return output_data

        local_usage = np.array([m for _, m in output_data])
        num_data = len(local_usage)
        global_usage = (
            None if self._mpi.rank != 0 else
            np.empty([self._mpi.size, num_data], dtype=np.float64)
        )
        self._mpi.comm.Gather(local_usage, global_usage, root=0)

        if self._mpi.rank != 0:
            return None

        avg_usage = np.sum(global_usage, axis=0) / self._mpi.size
        result = list(zip([n for n, _ in output_data], avg_usage))
        return result


class Diagnostics:
    """
    Manages enabled diagnostics modules.
    Handles initialization and requests statistics tracking and writing.
    """
    def __init__(self, output_dir: str):
        self._modules : List[DiagnosticModule] = []
        self._output_dir : str = output_dir

    def initialize(self, config: Config, logger: Logger, mpi: MPIHandler) -> None:
        """
        Initializes the diagnostics module and creates the modules.

        Parameters
        ----------
        config : Config
            Configuration object.
        logger : Logger
            Logger object.
        mpi : MPIHandler
            MPIHandler object.
        """
        # MEMORY USAGE
        memory_usage_type = config.memory_usage_output_type()
        memory_output_n = config.memory_usage_output_n()
        if memory_usage_type in ["all", "local"]:
            self._modules.append(MemoryUsage(
                logger=logger,
                mpi=mpi,
                output_dir=self._output_dir,
                is_global=False,
                output_n=memory_output_n
            ))

        if memory_usage_type in ["all", "global"]:
            self._modules.append(MemoryUsage(
                logger=logger,
                mpi=mpi,
                output_dir=self._output_dir,
                is_global=True,
                output_n=memory_output_n
            ))

        # OTHER MODULES
        # ...

    def log(self, n: int, force_log: bool) -> None:
        """
        Calls log for all modules.

        Parameters
        ----------
        n : int
            Current time step.
        force_log : bool
            Attempt to force statistics tracking.
        """
        for module in self._modules:
            module.log(n, force_log)

    def output(self) -> None:
        """
        Calls output for all modules.
        """
        for module in self._modules:
            module.output()
