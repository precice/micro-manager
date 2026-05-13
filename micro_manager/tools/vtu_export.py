import os
import xml.etree.ElementTree as ET
from typing import Dict, Optional

import numpy as np


def _as_3d_points(coords: np.ndarray) -> np.ndarray:
    coords_np = np.asarray(coords, dtype=np.float64)

    if coords_np.ndim != 2 or coords_np.shape[1] not in (2, 3):
        raise ValueError("VTU coordinates must have shape (N, 2) or (N, 3).")

    if coords_np.shape[1] == 3:
        return coords_np

    coords_3d = np.zeros((coords_np.shape[0], 3), dtype=np.float64)
    coords_3d[:, :2] = coords_np
    return coords_3d


def _data_array_and_components(
    values: np.ndarray, n_points: int, n_components: Optional[int] = None
) -> tuple[np.ndarray, int]:
    values_np = np.asarray(values, dtype=np.float64)

    if values_np.ndim == 0:
        values_np = np.full(n_points, values_np, dtype=np.float64)

    if values_np.shape[0] != n_points:
        raise ValueError(
            "VTU point data arrays must have the same first dimension as coordinates."
        )

    if values_np.ndim == 1:
        inferred_components = 1
    elif values_np.ndim == 2:
        inferred_components = values_np.shape[-1]
    else:
        raise ValueError("VTU point data arrays must be scalar or vector arrays.")

    component_count = n_components
    if component_count is None:
        component_count = 3 if inferred_components == 2 else inferred_components

    if inferred_components == 2 and component_count == 3:
        values_3d = np.zeros((n_points, 3), dtype=np.float64)
        values_3d[:, :2] = values_np
        values_np = values_3d
    elif inferred_components != component_count:
        raise ValueError(
            "VTU point data component count does not match the provided schema."
        )

    return values_np, component_count


def write_vtu(
    filename: str,
    coords: np.ndarray,
    data: dict,
    data_schema: Optional[Dict[str, int]] = None,
) -> None:
    """
    Writes a VTU file containing locally owned points and point data

    The file is an ``UnstructuredGrid`` with one ``VTK_VERTEX`` cell per point.
    ``data_schema`` can be used to force all ranks, including empty ranks, to
    write the same point-data arrays. This keeps the corresponding ``.pvtu``
    file valid even if a rank currently owns no points.
    """
    coords_3d = _as_3d_points(coords)
    n_points = coords_3d.shape[0]

    vtk_file = ET.Element(
        "VTKFile", type="UnstructuredGrid", version="0.1", byte_order="LittleEndian"
    )
    unstr_grid = ET.SubElement(vtk_file, "UnstructuredGrid")
    piece = ET.SubElement(
        unstr_grid, "Piece", NumberOfPoints=str(n_points), NumberOfCells=str(n_points)
    )

    points = ET.SubElement(piece, "Points")
    pts_arr = ET.SubElement(
        points,
        "DataArray",
        type="Float64",
        NumberOfComponents="3",
        format="ascii",
    )
    pts_arr.text = " ".join(map(str, coords_3d.ravel()))

    cells = ET.SubElement(piece, "Cells")

    conn_arr = ET.SubElement(
        cells, "DataArray", type="Int32", Name="connectivity", format="ascii"
    )
    conn_arr.text = " ".join(map(str, range(n_points)))

    off_arr = ET.SubElement(
        cells, "DataArray", type="Int32", Name="offsets", format="ascii"
    )
    off_arr.text = " ".join(map(str, range(1, n_points + 1)))

    type_arr = ET.SubElement(
        cells, "DataArray", type="UInt8", Name="types", format="ascii"
    )
    type_arr.text = " ".join(["1"] * n_points)

    schema = dict(data_schema or {})
    if not schema:
        for key, val in data.items():
            _, n_components = _data_array_and_components(np.asarray(val), n_points)
            schema[key] = n_components

    point_data = ET.SubElement(piece, "PointData")
    for key, n_components in schema.items():
        if key in data:
            val_np, n_components = _data_array_and_components(
                np.asarray(data[key]), n_points, n_components
            )
        elif n_components == 1:
            val_np = np.zeros(n_points, dtype=np.float64)
        else:
            val_np = np.zeros((n_points, n_components), dtype=np.float64)

        data_arr = ET.SubElement(
            point_data,
            "DataArray",
            type="Float64",
            Name=key,
            NumberOfComponents=str(n_components),
            format="ascii",
        )
        data_arr.text = " ".join(map(str, val_np.ravel()))

    tree = ET.ElementTree(vtk_file)
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    tree.write(filename, xml_declaration=True, encoding="utf-8")


def write_pvtu(filename: str, source_files: list, data_keys: dict) -> None:
    """
    Writes a Parallel VTU (.pvtu) file referencing rank-local .vtu files.
    """
    vtk_file = ET.Element(
        "VTKFile", type="PUnstructuredGrid", version="0.1", byte_order="LittleEndian"
    )
    p_grid = ET.SubElement(vtk_file, "PUnstructuredGrid", GhostLevel="0")

    p_points = ET.SubElement(p_grid, "PPoints")
    ET.SubElement(p_points, "PDataArray", type="Float64", NumberOfComponents="3")

    p_point_data = ET.SubElement(p_grid, "PPointData")
    for key, num_comp in data_keys.items():
        ET.SubElement(
            p_point_data,
            "PDataArray",
            type="Float64",
            Name=key,
            NumberOfComponents=str(num_comp),
        )

    for sf in source_files:
        ET.SubElement(p_grid, "Piece", Source=os.path.basename(sf))

    tree = ET.ElementTree(vtk_file)
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    tree.write(filename, xml_declaration=True, encoding="utf-8")
