"""Stage 10G-A2: export the structured 2D MPM background grid as VTK_QUAD.

Nodal transient fields may be zero immediately after ``reset_nodal_state``;
exports always reflect the provided snapshot, never a reconstructed value.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
from geopympm.core import BackgroundGrid
from .vtk_writer import _pad_vectors
from ._vtu_mesh import _selected_fields, write_mesh_vtu

GRID_POINT_FIELDS = (
    'node_id', 'mass', 'momentum', 'velocity', 'acceleration',
    'force_internal', 'force_external', 'velocity_bc_mask', 'velocity_bc_value',
)
GRID_CELL_FIELDS = ('cell_id', 'cell_ix', 'cell_iy')


def _grid_fields(grid: BackgroundGrid):
    n = grid.n_nodes
    pdata = {
        'node_id': np.arange(n, dtype=np.int64),
        'mass': grid.mass,
        'momentum': _pad_vectors(grid.momentum, 2),
        'velocity': _pad_vectors(grid.velocity, 2),
        'acceleration': _pad_vectors(grid.acceleration, 2),
        'force_internal': _pad_vectors(grid.force_internal, 2),
        'force_external': _pad_vectors(grid.force_external, 2),
        'velocity_bc_mask': np.column_stack((grid.velocity_bc_mask.astype(np.uint8),
                                            np.zeros(n, dtype=np.uint8))),
        'velocity_bc_value': _pad_vectors(grid.velocity_bc_value, 2),
    }
    nx, ny = grid.n_cells
    cdata = {
        'cell_id': np.arange(nx*ny, dtype=np.int64),
        'cell_ix': np.tile(np.arange(nx, dtype=np.int32), ny),
        'cell_iy': np.repeat(np.arange(ny, dtype=np.int32), nx),
    }
    return pdata, cdata


def write_grid_vtu(grid: BackgroundGrid, filename: str | Path, *, time=None,
                   step=None, fields=None, cell_fields=None, overwrite=False) -> Path:
    """Write one full-background-grid snapshot, with x-fast Quad4 connectivity.

    ``fields`` selects named PointData; ``cell_fields`` selects CellData.
    Both default to all known fields. The grid and solver are never modified.
    """
    if not isinstance(grid, BackgroundGrid) or grid.ndim != 2:
        raise TypeError('grid must be a 2D BackgroundGrid')
    n = grid.n_nodes
    shapes = {
        'node_coordinates': (n, 2), 'mass': (n,),
        'momentum': (n, 2), 'velocity': (n, 2),
        'acceleration': (n, 2), 'force_internal': (n, 2),
        'force_external': (n, 2), 'velocity_bc_mask': (n, 2),
        'velocity_bc_value': (n, 2),
    }
    for name, shape in shapes.items():
        values = np.asarray(getattr(grid, name))
        if values.shape != shape or not np.all(np.isfinite(values)):
            raise ValueError(f'grid {name} must have shape {shape} and finite values')
    pdata, cdata = _grid_fields(grid)
    pdata = _selected_fields(pdata, fields, 'grid point')
    cdata = _selected_fields(cdata, cell_fields, 'grid cell')
    nx, ny = grid.n_cells
    ids = np.arange(nx*ny, dtype=np.int64)
    ix, iy = ids % nx, ids // nx
    bl = iy*(nx+1) + ix
    # CCW node order must match BackgroundGrid.cell_node_ids(): BL BR TR TL.
    connectivity = np.column_stack((bl, bl+1, bl+nx+2, bl+nx+1)).ravel()
    return write_mesh_vtu(
        filename, points=_pad_vectors(grid.node_coordinates, 2),
        connectivity=connectivity,
        offsets=4 * np.arange(1, nx*ny+1, dtype=np.int64),
        types=np.full(nx*ny, 9, dtype=np.uint8),
        point_data=pdata, cell_data=cdata, time=time, step=step,
        overwrite=overwrite,
    )


__all__ = ['GRID_POINT_FIELDS', 'GRID_CELL_FIELDS', 'write_grid_vtu']
