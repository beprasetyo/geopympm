"""Stage 3B: assemble nodal forces and advance grid momentum explicitly.

Call AFTER Stage 3A ``particle_to_grid``. The function consumes interpolation
computed for the CURRENT particle positions and uses the PRE-STEP projected
nodal momentum. It does not move particles, perform G2P, or update stress.

Gravity is spatially uniform; optional nodal forces are applied in addition.
The supported grid and interpolation are 2D structured Quad4 only.
"""

from __future__ import annotations

import numpy as np

from geopympm.core.grid import BackgroundGrid
from geopympm.core.particles import ParticleSet
from geopympm.interpolation.mpm import Quad4Interpolation
from geopympm.kernels.grid_update import (
    assemble_internal_force_quad4_kernel,
    explicit_grid_momentum_kernel,
)


def update_grid_momentum(
    particles: ParticleSet,
    grid: BackgroundGrid,
    interpolation: Quad4Interpolation,
    dt: float,
    *,
    gravity=(0.0, 0.0),
    nodal_external_force=None,
) -> None:
    """Assemble current forces and advance projected nodal momenta one step.

    Parameters
    ----------
    particles, grid : ParticleSet, BackgroundGrid
        Stage-1 2D objects. ``grid`` must already contain P2G mass/momentum.
    interpolation : Quad4Interpolation
        Current Stage-2 shape gradients and node IDs (recompute after motion).
    dt : float
        Positive finite timestep, seconds in SI.
    gravity : (2,) sequence, optional
        Uniform acceleration [gx, gy], m/s^2. Default (0,0) is load-free.
    nodal_external_force : (grid.n_nodes, 2) array, optional
        Additional nodal force in N (not an acceleration), *not* including
        gravity. For a 2D unit-thickness model use consistent particle mass
        and volume units. Forces on massless nodes are rejected.

    Side effects
    ------------
    - Replaces grid.force_internal and grid.force_external with this step's
      force assembly. Does not retain external forces from a prior step.
    - Updates grid.momentum, grid.velocity, grid.acceleration in place.
    - Keeps particle states, grid.mass, geometry, and BC masks/values intact.

    Notes
    -----
    This is an explicit grid substep only; *not* a complete MPM timestep.
    It should be called once after each P2G, before the future G2P stage.
    For constrained DOFs, ``m*a`` includes the momentum change caused by
    velocity enforcement, although that reaction is not stored separately.
    """
    if not isinstance(particles, ParticleSet):
        raise TypeError("particles must be a ParticleSet.")
    if not isinstance(grid, BackgroundGrid):
        raise TypeError("grid must be a BackgroundGrid.")
    if not isinstance(interpolation, Quad4Interpolation):
        raise TypeError("interpolation must be a Quad4Interpolation.")
    if particles.ndim != 2 or grid.ndim != 2:
        raise ValueError("Stage 3B grid update requires 2D particles and grid.")

    try:
        dt = float(dt)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("dt must be a finite, strictly positive number.") from exc
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be a finite, strictly positive number.")

    acceleration = np.asarray(gravity, dtype=np.float64)
    if acceleration.shape != (2,) or not np.all(np.isfinite(acceleration)):
        raise ValueError("gravity must be a finite length-2 vector.")

    n = particles.n_particles
    ids = interpolation.node_ids
    grad = interpolation.gradients
    if (ids.shape != (n, 4) or ids.dtype != np.int64 or
            not ids.flags.c_contiguous or
            np.any(ids < 0) or np.any(ids >= grid.n_nodes)):
        raise ValueError("interpolation.node_ids must be valid contiguous (n,4) int64 IDs.")
    if (grad.shape != (n, 4, 2) or grad.dtype != np.float64 or
            not grad.flags.c_contiguous or not np.all(np.isfinite(grad))):
        raise ValueError("interpolation.gradients must be finite contiguous (n,4,2) float64.")
    if not np.allclose(grad.sum(axis=1), 0.0, rtol=0.0, atol=1e-10):
        raise ValueError("interpolation.gradients must sum to zero per particle.")

    cells = interpolation.cell_indices
    local = interpolation.local_coordinates
    if cells.shape != (n, 2) or not np.issubdtype(cells.dtype, np.integer):
        raise ValueError("interpolation.cell_indices must contain integer (n,2) cell IDs.")
    if (local.shape != (n, 2) or not np.all(np.isfinite(local)) or
            np.any(local < 0.0) or np.any(local > 1.0)):
        raise ValueError("interpolation.local_coordinates must be finite (n,2) in [0,1].")
    if (np.any(cells[:, 0] < 0) or np.any(cells[:, 0] >= grid.n_cells[0]) or
            np.any(cells[:, 1] < 0) or np.any(cells[:, 1] >= grid.n_cells[1])):
        raise ValueError("interpolation.cell_indices are outside the grid.")
    bl = cells[:, 1] * (grid.n_cells[0] + 1) + cells[:, 0]
    if (np.any(ids[:, 0] != bl) or np.any(ids[:, 1] != bl + 1) or
            np.any(ids[:, 2] != bl + grid.n_cells[0] + 2) or
            np.any(ids[:, 3] != bl + grid.n_cells[0] + 1)):
        raise ValueError("interpolation.node_ids do not match the current grid connectivity.")
    uv = (particles.position - grid.origin) / grid.cell_size
    tol = 64.0 * np.finfo(np.float64).eps * max(1, *grid.n_cells)
    if (not np.all(np.isfinite(uv)) or
            np.any(np.abs(uv - (cells + local)) > tol)):
        raise ValueError("Interpolation is stale; re-evaluate it after particle motion.")
    # Checks of geometrical gradients protect against a result from a different
    # spacing while still leaving full interpolation re-evaluation to Stage 2.
    from geopympm.kernels.interpolation_quad4 import quad4_kernel
    expected_ids, _, expected_grad = quad4_kernel(
        np.ascontiguousarray(cells, dtype=np.int64),
        np.ascontiguousarray(local, dtype=np.float64),
        grid.n_cells[0], float(grid.cell_size[0]), float(grid.cell_size[1])
    )
    if (not np.array_equal(ids, expected_ids) or
            not np.allclose(grad, expected_grad, rtol=1e-13, atol=1e-12)):
        raise ValueError("interpolation.gradients are inconsistent with current grid/particles.")

    if (particles.volume.shape != (n,) or not np.all(np.isfinite(particles.volume)) or
            np.any(particles.volume <= 0.0)):
        raise ValueError("Particle volumes must be finite and strictly positive.")
    if (particles.stress.shape != (n, 6) or
            not np.all(np.isfinite(particles.stress))):
        raise ValueError("Particle stresses must be finite and shape (n,6).")
    if (grid.mass.shape != (grid.n_nodes,) or not np.all(np.isfinite(grid.mass)) or
            np.any(grid.mass < 0.0)):
        raise ValueError("Grid mass must be finite and nonnegative.")
    if (grid.momentum.shape != (grid.n_nodes, 2) or
            not np.all(np.isfinite(grid.momentum))):
        raise ValueError("Grid momentum must be finite with shape (n_nodes,2).")
    # A useful fail-fast guard for an omitted Stage 3A P2G call. This is not a
    # conservation proof of all particle/node fields; Stage 3A owns that.
    if not np.isclose(grid.mass.sum(), particles.mass.sum(), rtol=1e-11, atol=0.0):
        raise ValueError("Grid mass is inconsistent with particles; call particle_to_grid first.")

    bc_mask = grid.velocity_bc_mask
    bc_value = grid.velocity_bc_value
    if bc_mask.shape != (grid.n_nodes, 2) or bc_mask.dtype != np.bool_:
        raise ValueError("grid.velocity_bc_mask must be a boolean (n_nodes,2) array.")
    if bc_value.shape != (grid.n_nodes, 2) or not np.all(np.isfinite(bc_value)):
        raise ValueError("grid.velocity_bc_value must be finite (n_nodes,2).")

    extra = None
    if nodal_external_force is not None:
        extra = np.asarray(nodal_external_force, dtype=np.float64)
        if extra.shape != (grid.n_nodes, 2) or not np.all(np.isfinite(extra)):
            raise ValueError("nodal_external_force must be finite with shape (n_nodes,2).")
        if np.any((grid.mass == 0.0) & np.any(extra != 0.0, axis=1)):
            raise ValueError("nodal_external_force cannot act on massless nodes.")

    # Do not overwrite any nodal state until *all* input checks have passed.
    grid.force_internal.fill(0.0)
    assemble_internal_force_quad4_kernel(
        ids, grad, particles.volume, particles.stress, grid.force_internal
    )
    grid.force_external[:] = grid.mass[:, None] * acceleration
    if extra is not None:
        grid.force_external += extra
    explicit_grid_momentum_kernel(
        grid.mass, grid.momentum, grid.velocity, grid.acceleration,
        grid.force_internal, grid.force_external,
        bc_mask, bc_value, dt,
    )
