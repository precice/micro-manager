from unittest import TestCase
from unittest.mock import patch
import sys
import importlib


class TestDependencyCheck(TestCase):
    def test_missing_required_dependency_raises_import_error(self):
        """
        Check that a clear ImportError is raised when a required dependency is missing.
        """
        with patch.dict(sys.modules, {"precice": None}):
            import micro_manager
            with self.assertRaises(ImportError) as context:
                importlib.reload(micro_manager)
            self.assertIn("pyprecice", str(context.exception))
            self.assertIn("pip install", str(context.exception))

    def test_all_dependencies_present_no_error(self):
        """
        Check that no error is raised when all required dependencies are installed.
        """
        try:
            import micro_manager
        except ImportError:
            self.fail("micro_manager raised ImportError unexpectedly")
