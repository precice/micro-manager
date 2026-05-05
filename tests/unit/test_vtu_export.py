import importlib.util
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np

vtu_export_path = (
    Path(__file__).parents[2] / "micro_manager" / "tools" / "vtu_export.py"
)
spec = importlib.util.spec_from_file_location("vtu_export", vtu_export_path)
assert spec is not None
assert spec.loader is not None
vtu_export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vtu_export)
write_pvtu = vtu_export.write_pvtu
write_vtu = vtu_export.write_vtu


class TestVTUExport(TestCase):
    def test_write_vtu_with_empty_rank_and_schema(self):
        with TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "empty.vtu")

            write_vtu(
                filename,
                np.empty((0, 2)),
                {},
                {"Scalar": 1, "Vector": 3},
            )

            root = ET.parse(filename).getroot()
            piece = root.find("./UnstructuredGrid/Piece")
            assert piece is not None
            self.assertEqual(piece.get("NumberOfPoints"), "0")
            self.assertEqual(piece.get("NumberOfCells"), "0")

            data_arrays = piece.findall("./PointData/DataArray")
            self.assertEqual(
                [array.get("Name") for array in data_arrays], ["Scalar", "Vector"]
            )
            self.assertEqual(
                [array.get("NumberOfComponents") for array in data_arrays], ["1", "3"]
            )

    def test_write_vtu_pads_two_component_vectors(self):
        with TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "rank.vtu")

            write_vtu(
                filename,
                np.array([[1.0, 2.0]]),
                {"Vector": np.array([[3.0, 4.0]])},
                {"Vector": 3},
            )

            root = ET.parse(filename).getroot()
            vector_data = root.find("./UnstructuredGrid/Piece/PointData/DataArray")
            assert vector_data is not None
            self.assertEqual(vector_data.get("NumberOfComponents"), "3")
            self.assertEqual(vector_data.text, "3.0 4.0 0.0")

    def test_write_pvtu_references_rank_files(self):
        with TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "all.pvtu")

            write_pvtu(filename, ["rank0.vtu", "rank1.vtu"], {"Scalar": 1})

            root = ET.parse(filename).getroot()
            pieces = root.findall("./PUnstructuredGrid/Piece")
            self.assertEqual(
                [piece.get("Source") for piece in pieces], ["rank0.vtu", "rank1.vtu"]
            )


if __name__ == "__main__":
    import unittest

    unittest.main()
