"""Stage 10G-A2: CPDI1 physical parallelogram domains as VTK_QUAD cells.

The geometry uses the SAME ``candidate_cpdi_geometry`` function used by the
CPDI interpolation kernel. It is not the axis-aligned ``domain_size`` envelope.
Each domain has four private vertices to preserve distinct particle cells.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np

from geopympm.core import BackgroundGrid, ParticleSet
from geopympm.interpolation.cpdi import candidate_cpdi_geometry
from .vtk_writer import _field_arrays, _pad_vectors, _validate_state
from ._vtu_mesh import _selected_fields, write_mesh_vtu

CPDI_POINT_FIELDS = ('particle_id', 'corner_id')
CPDI_CELL_FIELDS = (
    'particle_id', 'material_id', 'domain_area', 'reference_domain_area',
    'area_ratio', 'jacobian', 'pressure', 'von_mises', 'stress',
    'deformation_gradient',
)


def write_cpdi_domains_vtu(particles: ParticleSet, grid: BackgroundGrid,
                           filename: str | Path, *, time=None, step=None,
                           fields=None, point_fields=None, overwrite=False) -> Path:
    """Export CPDI domains at the supplied snapshot, not a timestep collection.

    ``fields`` is the selectable **CellData** field list; ``point_fields``
    selects per-corner PointData. Uses original full domain sides and full F.
    Rejects collapsed, inverted, or out-of-grid domains with CPDI's own checks.
    """
    _validate_state(particles)
    if not isinstance(grid, BackgroundGrid):
        raise TypeError('grid must be a BackgroundGrid')
    corners, _, _, _ = candidate_cpdi_geometry(particles, grid)
    n = particles.n_particles
    x0 = corners[:, 0, :]
    x1 = corners[:, 1, :]
    x3 = corners[:, 3, :]
    # Oriented 2D polygon area (positive for valid CPDI parallelograms).
    e1, e2 = x1-x0, x3-x0
    area = e1[:, 0]*e2[:, 1] - e1[:, 1]*e2[:, 0]
    area0 = particles.domain_size0[:, 0] * particles.domain_size0[:, 1]
    quantities = _field_arrays(particles)
    cdata = {
        'particle_id': np.arange(n, dtype=np.int64),
        'material_id': particles.material_id,
        'domain_area': area,
        'reference_domain_area': area0,
        'area_ratio': area / area0,
        'jacobian': quantities['jacobian'],
        'pressure': quantities['pressure'],
        'von_mises': quantities['von_mises'],
        'stress': quantities['stress'],
        'deformation_gradient': quantities['deformation_gradient'],
    }
    pdata = {
        'particle_id': np.repeat(np.arange(n, dtype=np.int64), 4),
        'corner_id': np.tile(np.arange(4, dtype=np.int32), n),
    }
    return write_mesh_vtu(
        filename, points=_pad_vectors(corners.reshape(4*n, 2), 2),
        connectivity=np.arange(4*n, dtype=np.int64),
        offsets=4 * np.arange(1, n+1, dtype=np.int64),
        types=np.full(n, 9, dtype=np.uint8),
        point_data=_selected_fields(pdata, point_fields, 'CPDI point'),
        cell_data=_selected_fields(cdata, fields, 'CPDI cell'),
        time=time, step=step, overwrite=overwrite,
    )


__all__ = ['CPDI_POINT_FIELDS', 'CPDI_CELL_FIELDS', 'write_cpdi_domains_vtu']
