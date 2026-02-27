import argparse
import os


def _check_dependencies():
    import importlib.metadata

    required = {
        "pyprecice": ("precice", "3.2"),
        "numpy": ("numpy", None),
        "mpi4py": ("mpi4py", None),
        "psutil": ("psutil", None),
    }

    missing = []
    version_errors = []

    for package, (import_name, min_version) in required.items():
        try:
            __import__(import_name)
            if min_version:
                installed_version = importlib.metadata.version(package)
                from packaging.version import Version
                if Version(installed_version) < Version(min_version):
                    version_errors.append(
                        "{} (installed: {}, required: >={})".format(
                            package, installed_version, min_version
                        )
                    )
        except ImportError:
            missing.append(package)

    errors = []
    if missing:
        errors.append(
            "Missing packages: {}. Install via: pip install {}".format(
                ", ".join(missing), " ".join(missing)
            )
        )
    if version_errors:
        errors.append(
            "Version requirements not met: {}".format(", ".join(version_errors))
        )

    if errors:
        raise ImportError("\n".join(errors))

    print("All dependencies are correctly installed.")


from .config import Config
from .micro_simulation import MicroSimulationInterface
from .micro_manager import MicroManagerCoupling

try:
    from .snapshot.snapshot import MicroManagerSnapshot

    is_snapshot_possible = True
except ImportError:
    is_snapshot_possible = False


def main():
    parser = argparse.ArgumentParser(description=".")
    parser.add_argument(
        "config_file",
        type=str,
        nargs="?",
        help="Path to the JSON config file of the manager.",
    )
    parser.add_argument(
        "--snapshot", action="store_true", help="compute offline snapshot database"
    )
    parser.add_argument(
        "--test-dependencies",
        action="store_true",
        help="Check if all required dependencies are correctly installed.",
    )
    parser.add_argument(
        "log_file",
        type=str,
        nargs="?",
        default="",
        help="Path to the log file. If not provided, logs are printed to stdout.",
    )

    args = parser.parse_args()

    if args.test_dependencies:
        _check_dependencies()
        return

    if not args.config_file:
        parser.error("config_file is required unless --test-dependencies is used.")

    config_file_path = args.config_file
    if not os.path.isabs(config_file_path):
        config_file_path = os.getcwd() + "/" + config_file_path

    if not args.snapshot:
        manager = MicroManagerCoupling(config_file_path, log_file=args.log_file)
    else:
        if is_snapshot_possible:
            manager = MicroManagerSnapshot(config_file_path, log_file=args.log_file)
        else:
            raise ImportError(
                "The Micro Manager snapshot computation requires the h5py package."
            )

    manager.initialize()
    manager.solve()
