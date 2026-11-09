"""Shared VTK XML mesh serializer for Stage 10G-A2.

Private module, no dependency on vtk Python or solver internals. VTK ASCII
Float64 uses the same round-trip format as the Stage 10G-A1 particle writer.
"""

from __future__ import annotations

from pathlib import Path
import math
import xml.etree.ElementTree as ET

import numpy as np

from .vtk_writer import _append_array


def _time_and_step(time: float | None, step: int | None) -> None:
    if time is not None:
        if (isinstance(time, (bool, np.bool_)) or not np.isscalar(time)
                or not math.isfinite(float(time)) or float(time) < 0.0):
            raise ValueError('time must be finite and nonnegative')
    if step is not None:
        if (isinstance(step, (bool, np.bool_)) or not isinstance(step, (int, np.integer))
                or step < 0):
            raise ValueError('step must be a nonnegative integer')


def _selected_fields(available: dict[str, np.ndarray], fields, label: str):
    names = tuple(available) if fields is None else tuple(fields)
    if (len(names) != len(set(names)) or any(name not in available for name in names)):
        raise ValueError(f'{label} fields must be unique names from {tuple(available)}')
    return {name: available[name] for name in names}


def write_mesh_vtu(
    filename: str | Path, *, points: np.ndarray, connectivity: np.ndarray,
    offsets: np.ndarray, types: np.ndarray, point_data: dict[str, np.ndarray],
    cell_data: dict[str, np.ndarray], time=None, step=None, overwrite=False,
) -> Path:
    """Write a VTK XML UnstructuredGrid; no mutation and no creation on error."""
    path = Path(filename)
    if path.suffix.lower() != '.vtu':
        raise ValueError('VTK mesh filename must end in .vtu')
    if path.exists() and not overwrite:
        raise FileExistsError(f'Refusing to overwrite VTK mesh output: {path}')
    _time_and_step(time, step)

    xyz = np.asarray(points)
    conn = np.asarray(connectivity)
    off = np.asarray(offsets)
    typ = np.asarray(types)
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or not np.issubdtype(xyz.dtype, np.number)
            or not np.all(np.isfinite(xyz))):
        raise ValueError('points must have shape (n,3) with finite coordinates')
    if (conn.ndim != 1 or off.ndim != 1 or typ.ndim != 1
            or not all(np.issubdtype(a.dtype, np.integer) for a in (conn, off, typ))
            or len(off) != len(typ)):
        raise ValueError('connectivity/offsets/types must be integer arrays of valid lengths')
    if (len(off) > 0 and (off[-1] != len(conn) or np.any(np.diff(off) <= 0))):
        raise ValueError('cell offsets must be strictly increasing and cover connectivity')
    if len(conn) and (np.min(conn) < 0 or np.max(conn) >= len(xyz)):
        raise ValueError('connectivity contains a point index outside the mesh')
    if not np.all(typ == 9):
        raise ValueError('Stage 10G-A2 mesh writer expects VTK_QUAD (type 9) cells')
    if len(off) and not np.array_equal(np.diff(np.r_[0, off]), np.full(len(off), 4)):
        raise ValueError('VTK_QUAD must contain exactly four nodes')

    npoints, ncells = len(xyz), len(typ)
    for label, arrays, expected in [('PointData', point_data, npoints),
                                    ('CellData', cell_data, ncells)]:
        for name, values in arrays.items():
            arr = np.asarray(values)
            if (arr.ndim not in (1, 2) or len(arr) != expected or arr.dtype.kind not in 'fiub'
                    or not np.all(np.isfinite(arr))):
                raise ValueError(f'{label} {name} must contain {expected} finite numeric tuples')

    root = ET.Element('VTKFile', {'type': 'UnstructuredGrid',
                                  'version': '1.0', 'byte_order': 'LittleEndian'})
    ug = ET.SubElement(root, 'UnstructuredGrid')
    if time is not None or step is not None:
        fd = ET.SubElement(ug, 'FieldData')
        if time is not None:
            _append_array(fd, 'TimeValue', np.array([float(time)], dtype=np.float64))
        if step is not None:
            _append_array(fd, 'Step', np.array([int(step)], dtype=np.int64))
    piece = ET.SubElement(ug, 'Piece', {'NumberOfPoints': str(npoints),
                                       'NumberOfCells': str(ncells)})
    pd = ET.SubElement(piece, 'PointData')
    for name, values in point_data.items():
        _append_array(pd, name, np.asarray(values, dtype=np.uint8) if np.asarray(values).dtype.kind == 'b' else values)
    cd = ET.SubElement(piece, 'CellData')
    for name, values in cell_data.items():
        _append_array(cd, name, np.asarray(values, dtype=np.uint8) if np.asarray(values).dtype.kind == 'b' else values)
    geom = ET.SubElement(piece, 'Points')
    _append_array(geom, None, xyz)
    cells = ET.SubElement(piece, 'Cells')
    _append_array(cells, 'connectivity', conn.astype(np.int64, copy=False))
    _append_array(cells, 'offsets', off.astype(np.int64, copy=False))
    _append_array(cells, 'types', typ.astype(np.uint8, copy=False))
    ET.indent(root, space='  ')
    path.parent.mkdir(parents=True, exist_ok=True)
    xml = ET.tostring(root, encoding='utf-8', xml_declaration=True)
    with path.open('wb' if overwrite else 'xb') as stream:
        stream.write(xml)
    return path
