"""
Unit tests for config validation: mandatory parameters, optional parameters with defaults.
"""
import json
import os
import tempfile
from unittest import TestCase
from unittest.mock import MagicMock

from micro_manager.config import Config, ConfigError


class TestConfigValidation(TestCase):
    """Test config mandatory/optional parameter validation."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        if os.path.exists(self.tmpdir):
            shutil.rmtree(self.tmpdir)

    def _write_config(self, data, filename="test_config.json"):
        path = os.path.join(self.tmpdir, filename)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path

    def test_missing_micro_file_name_raises_config_error(self):
        """Missing mandatory micro_file_name raises ConfigError with clear path."""
        config_data = {
            "coupling_params": {
                "precice_config_file_name": "precice.xml",
                "macro_mesh_name": "Mesh",
                "read_data_names": [],
                "write_data_names": [],
            },
            "simulation_params": {
                "micro_dt": 1.0,
                "macro_domain_bounds": [0, 1, 0, 1],
            },
        }
        path = self._write_config(config_data)
        config = Config(path)
        config.set_logger(MagicMock())
        with self.assertRaises(ConfigError) as ctx:
            config.read_json_micro_manager()
        self.assertIn("micro_file_name", str(ctx.exception))
        self.assertIn("Missing required", str(ctx.exception))

    def test_missing_coupling_params_raises_config_error(self):
        """Missing mandatory coupling_params raises ConfigError."""
        config_data = {
            "micro_file_name": "dummy",
            "simulation_params": {
                "micro_dt": 1.0,
                "macro_domain_bounds": [0, 1, 0, 1],
            },
        }
        path = self._write_config(config_data)
        config = Config(path)
        config.set_logger(MagicMock())
        with self.assertRaises(ConfigError) as ctx:
            config.read_json_micro_manager()
        self.assertIn("coupling_params", str(ctx.exception))

    def test_missing_precice_config_file_name_raises_config_error(self):
        """Missing precice_config_file_name in coupling_params raises ConfigError."""
        config_data = {
            "micro_file_name": "dummy",
            "coupling_params": {
                "macro_mesh_name": "Mesh",
                "read_data_names": [],
                "write_data_names": [],
            },
            "simulation_params": {
                "micro_dt": 1.0,
                "macro_domain_bounds": [0, 1, 0, 1],
            },
        }
        path = self._write_config(config_data)
        config = Config(path)
        config.set_logger(MagicMock())
        with self.assertRaises(ConfigError) as ctx:
            config.read_json_micro_manager()
        self.assertIn("precice_config_file_name", str(ctx.exception))

    def test_optional_output_directory_uses_none_default(self):
        """When output_directory is omitted, config uses None (logs to cwd)."""
        config_data = {
            "micro_file_name": "dummy",
            "coupling_params": {
                "precice_config_file_name": "precice.xml",
                "macro_mesh_name": "Mesh",
                "read_data_names": [],
                "write_data_names": [],
            },
            "simulation_params": {
                "micro_dt": 1.0,
                "macro_domain_bounds": [0, 1, 0, 1],
            },
        }
        path = self._write_config(config_data)
        config = Config(path)
        config.set_logger(MagicMock())
        config.read_json_micro_manager()
        self.assertIsNone(config._output_dir)

    def test_optional_diagnostics_uses_defaults(self):
        """When diagnostics section is omitted, micro_output_n defaults to 1."""
        config_data = {
            "micro_file_name": "dummy",
            "coupling_params": {
                "precice_config_file_name": "precice.xml",
                "macro_mesh_name": "Mesh",
                "read_data_names": [],
                "write_data_names": [],
            },
            "simulation_params": {
                "micro_dt": 1.0,
                "macro_domain_bounds": [0, 1, 0, 1],
            },
        }
        path = self._write_config(config_data)
        config = Config(path)
        config.set_logger(MagicMock())
        config.read_json_micro_manager()
        self.assertEqual(config._micro_output_n, 1)
