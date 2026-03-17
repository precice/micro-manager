import argparse
import os


def _check_dependencies():
    import importlib.metadata

    from packaging.requirements import Requirement
    from packaging.version import Version

    _import_name_map = {"pyprecice": "precice"}
    required = {}
    _pkg_requires = importlib.metadata.requires("micro-manager-precice") or []
    for _dep in _pkg_requires:
        if "; extra ==" in _dep:
            continue
        _req = Requirement(_dep)
        _import_name = _import_name_map.get(_req.name, _req.name)
        _min_version = None
        for _spec in _req.specifier:
            if _spec.operator == ">=":
                _min_version = _spec.version
        required[_req.name] = (_import_name, _min_version)

    missing = []
    version_errors = []

    for package, (import_name, min_version) in required.items():
        try:
            __import__(import_name)
            if min_version:
                installed_version = importlib.metadata.version(package)
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


import sys

# Delay heavy imports if only running dependency check
if "--test-dependencies" not in sys.argv:
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
