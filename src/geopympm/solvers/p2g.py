"""Stage 3A: high-level Particle-to-Grid mass/momentum projection.

2D MPM using either Stage-2 Quad4 or Stage-10 uGIMP shape weights. This function resets
transient grid data before assembling nodal mass and momentum and recovers
unconstrained nodal velocity. It does *not* compute nodal forces, apply
boundary conditions, advance the grid, or move material particles.

Interpolation must be evaluated from current particle positions. Re-evaluate
it whenever positions change. A consistency check catches stale results.
"""

from __future__ import annotations

import numpy as np

from geopympm.core.grid import BackgroundGrid
from geopympm.core.particles import ParticleSet
from geopympm.interpolation.mpm import Quad4Interpolation
from geopympm.interpolation.gimp import GIMPInterpolation, validate_current_gimp
from geopympm.kernels.p2g import (
    p2g_mass_momentum_kernel,
    recover_nodal_velocity_kernel,
)


def particle_to_grid(
    particles: ParticleSet,
    grid: BackgroundGrid,
    interpolation: Quad4Interpolation | GIMPInterpolation,
) -> None:
    """Reset grid, project particle mass/momentum, then recover nodal velocity.

    Requires the current interpolation evaluation returned by
    ``evaluate_quad4(particles, grid)``. It uses one velocity field per grid
    node (single-velocity-field MPM; multimaterial contact is out of scope).

    Grid velocity is the projected velocity before force integration and
    boundary-condition enforcement. Grid boundary-condition *data* persist.
    """
    if not isinstance(particles, ParticleSet):
        raise TypeError("particles must be a ParticleSet.")
    if not isinstance(grid, BackgroundGrid):
        raise TypeError("grid must be a BackgroundGrid.")
    if not isinstance(interpolation, (Quad4Interpolation, GIMPInterpolation)):
        raise TypeError("interpolation must be Quad4Interpolation or GIMPInterpolation.")
    if particles.ndim != 2 or grid.ndim != 2:
        raise ValueError("Stage 3A P2G requires 2D particles and grid.")

    is_gimp = isinstance(interpolation, GIMPInterpolation)
    n_support = 16 if is_gimp else 4
    n = particles.n_particles
    ids = interpolation.node_ids
    weights = interpolation.weights
    if ids.shape != (n, n_support) or ids.dtype != np.int64:
        raise ValueError("interpolation.node_ids must have shape (n, 4) and dtype int64.")
    if weights.shape != (n, n_support) or weights.dtype != np.float64:
        raise ValueError("interpolation.weights must have shape (n, 4) and dtype float64.")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0) or np.any(weights > 1.0):
        raise ValueError("interpolation.weights must be finite and lie in [0, 1].")
    if not np.allclose(weights.sum(axis=1), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("interpolation.weights must sum to 1 for every particle.")
    if np.any(ids < 0) or np.any(ids >= grid.n_nodes):
        raise ValueError("interpolation.node_ids contains out-of-range grid node IDs.")
    if not ids.flags.c_contiguous or not weights.flags.c_contiguous:
        raise ValueError("interpolation arrays must be C-contiguous.")

    # Protect against reusing interpolation after a particle has moved.
    cells = interpolation.cell_indices
    local = interpolation.local_coordinates
    if cells.shape != (n, 2) or local.shape != (n, 2):
        raise ValueError("interpolation cell indices/local coordinates have invalid shapes.")
    if not np.issubdtype(cells.dtype, np.integer):
        raise ValueError("interpolation.cell_indices must contain integer indices.")
    if not np.all(np.isfinite(local)) or np.any(local < 0.0) or np.any(local > 1.0):
        raise ValueError("interpolation local coordinates must be finite in [0, 1].")
    if (np.any(cells[:, 0] < 0) or np.any(cells[:, 0] >= grid.n_cells[0]) or
            np.any(cells[:, 1] < 0) or np.any(cells[:, 1] >= grid.n_cells[1])):
        raise ValueError("interpolation.cell_indices contains out-of-range cells.")
    # Ensure that a stale result computed on a different grid cannot silently
    # deposit particle mass/momentum to incorrect (but in-range) node IDs.
    if not is_gimp:
        bl = cells[:, 1] * (grid.n_cells[0] + 1) + cells[:, 0]
        if (np.any(ids[:, 0] != bl) or np.any(ids[:, 1] != bl + 1) or
                np.any(ids[:, 2] != bl + grid.n_cells[0] + 2) or
                np.any(ids[:, 3] != bl + grid.n_cells[0] + 1)):
            raise ValueError("interpolation.node_ids do not match grid connectivity.")
    particle_uv = (particles.position - grid.origin) / grid.cell_size
    expected_uv = cells + local
    tolerance = 64.0 * np.finfo(np.float64).eps * max(1, *grid.n_cells)
    if (not np.all(np.isfinite(particle_uv)) or
            np.any(np.abs(particle_uv - expected_uv) > tolerance)):
        raise ValueError("Interpolation is stale or inconsistent; call evaluate_quad4 again.")

    if is_gimp:
        validate_current_gimp(particles, grid, interpolation)

    if (particles.mass.shape != (n,) or not np.all(np.isfinite(particles.mass))
            or np.any(particles.mass <= 0.0)):
        raise ValueError("Particle mass must remain finite, positive and shape (n,).")
    if (particles.velocity.shape != (n, 2) or
            not np.all(np.isfinite(particles.velocity))):
        raise ValueError("Particle velocity must remain finite with shape (n, 2).")

    # Validation precedes reset, so a bad input does not destroy existing grid state.
    grid.reset_nodal_state()
    p2g_mass_momentum_kernel(ids, weights, particles.mass, particles.velocity,
                             grid.mass, grid.momentum)
    recover_nodal_velocity_kernel(grid.mass, grid.momentum, grid.velocity)
