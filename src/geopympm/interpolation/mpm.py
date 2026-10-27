"""Standard (point-particle) MPM interpolation on a structured 2D Quad4 grid.

This module determines the *single* cell containing each particle and
computes the shape functions and spatial gradients of that cell's four nodes.
This is **conventional MPM**, not GIMP: particle ``domain_size`` is unused.

Policy for boundary points:
- On an interior vertical/horizontal grid line, select the cell to the right/up.
- On the upper/right *outer* boundary, select the last cell.
- Reject particles outside the grid, apart from a small roundoff tolerance;
  values within that tolerance are snapped to the boundary for evaluation.
- Recompute after particle positions change; results are not a persistent cache.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geopympm.core.grid import BackgroundGrid
from geopympm.core.particles import ParticleSet
from geopympm.kernels.interpolation_quad4 import quad4_kernel


@dataclass(slots=True)
class Quad4Interpolation:
    """Evaluated interpolation data for all particles at one configuration.

    All fields are NumPy arrays with one leading row per particle:

    - ``cell_indices``: (n, 2) int64, zero-based cell (ix, iy)
    - ``local_coordinates``: (n, 2) float64, cell coordinates (s, t) in [0, 1]
    - ``node_ids``: (n, 4) int64, corners BL, BR, TR, TL
    - ``weights``: (n, 4) float64, N_i(x_p)
    - ``gradients``: (n, 4, 2) float64, global (dN_i/dx, dN_i/dy)

    Results depend on current particle positions; do not reuse them after G2P
    has moved the particles or after the grid geometry changes.
    """

    cell_indices: np.ndarray
    local_coordinates: np.ndarray
    node_ids: np.ndarray
    weights: np.ndarray
    gradients: np.ndarray


def evaluate_quad4(particles: ParticleSet, grid: BackgroundGrid) -> Quad4Interpolation:
    """Locate particles and evaluate standard 2D Quad4 MPM interpolation.

    Raises
    ------
    TypeError
        When the provided data containers are not supported types.
    ValueError
        For mismatched spatial dimensions, nonfinite or out-of-grid particle
        positions. The complete particle must be inside the background grid;
        no extrapolation or periodic wrapping is performed.

    Notes
    -----
    Conventional MPM interpolation uses only the four surrounding grid nodes.
    A later GIMP method will require different supports and domain-aware shape
    functions, and should not be obtained by modifying these Quad4 weights.
    """
    if not isinstance(particles, ParticleSet):
        raise TypeError("particles must be a ParticleSet.")
    if not isinstance(grid, BackgroundGrid):
        raise TypeError("grid must be a BackgroundGrid.")
    if particles.ndim != 2 or grid.ndim != 2:
        raise ValueError("Quad4 interpolation requires 2D ParticleSet and 2D BackgroundGrid.")

    positions = particles.position
    if positions.ndim != 2 or positions.shape != (particles.n_particles, 2):
        raise ValueError("Particle positions must have shape (n_particles, 2).")
    if not np.all(np.isfinite(positions)):
        raise ValueError("Particle positions contain non-finite values.")

    # Normalized coordinates u=(x-origin_x)/dx, v=(y-origin_y)/dy.
    uv = (positions - grid.origin) / grid.cell_size
    if not np.all(np.isfinite(uv)):
        raise ValueError("Grid-relative particle coordinates are not finite.")

    n_cells = np.asarray(grid.n_cells, dtype=np.int64)
    # Relative floating-point tolerance in *cell-coordinate* units. We only
    # clamp roundoff near the domain boundary; larger excursions are errors.
    tolerance = 64.0 * np.finfo(np.float64).eps * max(1, *grid.n_cells)
    outside = np.any((uv < -tolerance) | (uv > n_cells + tolerance), axis=1)
    if np.any(outside):
        p = int(np.flatnonzero(outside)[0])
        raise ValueError(
            f"Particle {p} at {positions[p].tolist()} lies outside the "
            f"background grid domain (origin={grid.origin.tolist()}, "
            f"extent={(grid.origin + grid.cell_size * n_cells).tolist()})."
        )

    bounded_uv = np.clip(uv, 0.0, n_cells.astype(np.float64))
    cell_indices = np.floor(bounded_uv).astype(np.int64)
    # Points at the top/right outer boundary belong to the last available cell.
    cell_indices[:, 0] = np.minimum(cell_indices[:, 0], grid.n_cells[0] - 1)
    cell_indices[:, 1] = np.minimum(cell_indices[:, 1], grid.n_cells[1] - 1)
    local_coordinates = np.ascontiguousarray(bounded_uv - cell_indices)
    cell_indices = np.ascontiguousarray(cell_indices)

    node_ids, weights, gradients = quad4_kernel(
        cell_indices, local_coordinates, grid.n_cells[0],
        float(grid.cell_size[0]), float(grid.cell_size[1])
    )
    return Quad4Interpolation(
        cell_indices=cell_indices,
        local_coordinates=local_coordinates,
        node_ids=node_ids,
        weights=weights,
        gradients=gradients,
    )
