import os
import xml.etree.ElementTree as ET
import numpy as np


def write_vtu(filename: str, coords: np.ndarray, data: dict) -> None:
    """
    Writes a VTU file (UnstructuredGrid with VTK_VERTEX cells) containing
    points (coords) and associated scalar/vector point data.

    Parameters
    ----------
    filename : str
        Output file path (e.g., "output.vtu").
    coords : numpy array
        2D or 3D numpy array of shape (N, 2) or (N, 3).
    data : dict
        Dictionary of point data fields. Keys are names, values are scalar (N,) or vector (N, d) arrays.
    """
    n_points = coords.shape[0]

    if n_points == 0:
        return

    dim = coords.shape[1]

    if dim == 2:
        coords_3d = np.zeros((n_points, 3), dtype=np.float64)
        coords_3d[:, :2] = coords
    else:
        coords_3d = np.asarray(coords, dtype=np.float64)

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

    point_data = ET.SubElement(piece, "PointData")
    for key, val in data.items():
        val_np = np.asarray(val, dtype=np.float64)
        if val_np.ndim == 1:
            n_comp = 1
        else:
            n_comp = val_np.shape[1]
            if n_comp == 2:
                val_3d = np.zeros((n_points, 3), dtype=np.float64)
                val_3d[:, :2] = val_np
                val_np = val_3d
                n_comp = 3

        data_arr = ET.SubElement(
            point_data,
            "DataArray",
            type="Float64",
            Name=key,
            NumberOfComponents=str(n_comp),
            format="ascii",
        )
        data_arr.text = " ".join(map(str, val_np.ravel()))

    tree = ET.ElementTree(vtk_file)
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    tree.write(filename, xml_declaration=True, encoding="utf-8")


def write_pvtu(filename: str, source_files: list, data_keys: dict) -> None:
    """
    Writes a Parallel VTU (.pvtu) file referencing the multiple subset .vtu files.

    Parameters
    ----------
    filename : str
        Output file path for the PVTU.
    source_files : list
        List of VTU file names that this PVTU references.
    data_keys : dict
        Dictionary mapping data array names to their number of components.
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
