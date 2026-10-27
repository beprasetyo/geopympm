"""Stage 3C: high-level PIC/FLIP Grid-to-Particle (G2P) update.

Expected order for ONE kinematic substep:
    ip = evaluate_quad4(particles, grid)
    particle_to_grid(particles, grid, ip)            # Stage 3A
    update_grid_momentum(particles, grid, ip, dt, ...)  # Stage 3B
    grid_to_particle(particles, grid, ip, dt, ...)   # Stage 3C

No deformation gradient, strain, volume, or stress update is performed here.
This stage does not constitute a validated full explicit MPM solver.
"""

from __future__ import annotations

import numpy as np

from geopympm.core.grid import BackgroundGrid
from geopympm.core.particles import ParticleSet
from geopympm.interpolation.mpm import Quad4Interpolation
from geopympm.kernels.g2p import g2p_quad4_kernel


def grid_to_particle(
    particles: ParticleSet,
    grid: BackgroundGrid,
    interpolation: Quad4Interpolation,
    dt: float,
    *,
    pic_fraction: float = 1.0,
) -> None:
    """Advance particle velocity and position from updated grid state.

    Parameters
    ----------
    particles : ParticleSet
        Mutable particle state at time n; velocity and position are overwritten.
    grid : BackgroundGrid
        Nodal mass, updated velocity and acceleration from Stage 3B.
    interpolation : Quad4Interpolation
        Stage-2 result computed BEFORE this particle motion, after last movement.
    dt : float
        The same positive time step used in Stage 3B (seconds in SI).
    pic_fraction : float, default 1
        1.0 = PIC; 0.0 = FLIP; intermediate values linearly blend particle
        velocities. Particle position is ALWAYS advected with updated PIC
        grid velocity: x_new = x_old + dt * sum_i(N_i * v_i_new).

    Notes
    -----
    - FLIP increments use ``grid.acceleration`` from Stage 3B. As designed,
      these accelerations include enforcement of velocity BCs on grid nodes.
    - Mass, volume, stress, strain, deformation gradient and domain_size
      are not changed here. Stress/volume evolution belongs to a later stage.
    - The next G2P/P2G step MUST evaluate a fresh interpolation after motion.
    - Particles leaving the background-grid extent are rejected without
      partially modifying particle velocity or position. Enlarge the grid or
      define appropriate boundary handling in a later physics module.
    - No CFL / critical time-step estimate is provided by this function.
    """
    if not isinstance(particles, ParticleSet):
        raise TypeError("particles must be a ParticleSet.")
    if not isinstance(grid, BackgroundGrid):
        raise TypeError("grid must be a BackgroundGrid.")
    if not isinstance(interpolation, Quad4Interpolation):
        raise TypeError("interpolation must be a Quad4Interpolation.")
    if particles.ndim != 2 or grid.ndim != 2:
        raise ValueError("Stage 3C G2P requires 2D particles and grid.")

    try:
        dt = float(dt)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("dt must be finite and strictly positive.") from exc
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and strictly positive.")
    if isinstance(pic_fraction, (bool, np.bool_)):
        raise ValueError("pic_fraction must be a number in [0, 1], not a boolean.")
    try:
        pic_fraction = float(pic_fraction)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("pic_fraction must be in [0, 1].") from exc
    if not np.isfinite(pic_fraction) or not (0.0 <= pic_fraction <= 1.0):
        raise ValueError("pic_fraction must be in [0, 1].")

    n = particles.n_particles
    ids = interpolation.node_ids
    w = interpolation.weights
    cells = interpolation.cell_indices
    local = interpolation.local_coordinates

    if (ids.shape != (n, 4) or ids.dtype != np.int64 or
            not ids.flags.c_contiguous or np.any(ids < 0) or
            np.any(ids >= grid.n_nodes)):
        raise ValueError("interpolation.node_ids must be valid contiguous (n,4) int64 IDs.")
    if (w.shape != (n, 4) or w.dtype != np.float64 or
            not w.flags.c_contiguous or not np.all(np.isfinite(w)) or
            np.any(w < 0.0) or np.any(w > 1.0) or
            not np.allclose(w.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)):
        raise ValueError("interpolation.weights must be contiguous finite (n,4) and sum to one.")
    if (cells.shape != (n, 2) or not np.issubdtype(cells.dtype, np.integer) or
            np.any(cells[:, 0] < 0) or np.any(cells[:, 0] >= grid.n_cells[0]) or
            np.any(cells[:, 1] < 0) or np.any(cells[:, 1] >= grid.n_cells[1])):
        raise ValueError("interpolation.cell_indices are invalid for this grid.")
    if (local.shape != (n, 2) or not np.all(np.isfinite(local)) or
            np.any(local < 0.0) or np.any(local > 1.0)):
        raise ValueError("interpolation.local_coordinates must lie in [0,1].")

    bl = cells[:, 1] * (grid.n_cells[0] + 1) + cells[:, 0]
    if (np.any(ids[:, 0] != bl) or np.any(ids[:, 1] != bl + 1) or
            np.any(ids[:, 2] != bl + grid.n_cells[0] + 2) or
            np.any(ids[:, 3] != bl + grid.n_cells[0] + 1)):
        raise ValueError("interpolation.node_ids are inconsistent with the grid.")
    s = local[:, 0]
    t = local[:, 1]
    expected_w = np.column_stack((
        (1.0 - s) * (1.0 - t),
        s * (1.0 - t),
        s * t,
        (1.0 - s) * t,
    ))
    if not np.allclose(w, expected_w, rtol=1e-13, atol=1e-13):
        raise ValueError("interpolation.weights are inconsistent with local coordinates.")
    particle_uv = (particles.position - grid.origin) / grid.cell_size
    tolerance = 64.0 * np.finfo(np.float64).eps * max(1, *grid.n_cells)
    if (not np.all(np.isfinite(particle_uv)) or
            np.any(np.abs(particle_uv - (cells + local)) > tolerance)):
        raise ValueError("Interpolation is stale; re-evaluate after particle motion.")

    if (particles.velocity.shape != (n, 2) or
            not np.all(np.isfinite(particles.velocity))):
        raise ValueError("Particle velocities must be finite with shape (n,2).")
    if (grid.mass.shape != (grid.n_nodes,) or not np.all(np.isfinite(grid.mass))
            or np.any(grid.mass < 0.0)):
        raise ValueError("Grid mass must be finite and nonnegative.")
    if (grid.velocity.shape != (grid.n_nodes, 2) or
            not np.all(np.isfinite(grid.velocity))):
        raise ValueError("Updated grid velocity must be finite (n_nodes,2).")
    if (grid.acceleration.shape != (grid.n_nodes, 2) or
            not np.all(np.isfinite(grid.acceleration))):
        raise ValueError("Updated grid acceleration must be finite (n_nodes,2).")
    if (not np.isclose(grid.mass.sum(), particles.mass.sum(), rtol=1e-11, atol=0.0)):
        raise ValueError("Grid mass is inconsistent with particles; call particle_to_grid first.")
    if np.any((w > 0.0) & (grid.mass[ids] <= 0.0)):
        raise ValueError("Positive interpolation weights reference massless nodes.")

    # Produce both updated arrays before committing either one.
    new_position, new_velocity = g2p_quad4_kernel(
        ids, w, particles.position, particles.velocity,
        grid.velocity, grid.acceleration, dt, pic_fraction,
    )
    if not np.all(np.isfinite(new_position)) or not np.all(np.isfinite(new_velocity)):
        raise ValueError("G2P created non-finite particle state; no changes committed.")

    lower = grid.origin
    upper = grid.origin + grid.cell_size * np.asarray(grid.n_cells)
    uv_new = (new_position - lower) / grid.cell_size
    outside = np.any(
        (uv_new < -tolerance) |
        (uv_new > np.asarray(grid.n_cells) + tolerance),
        axis=1,
    )
    if np.any(outside):
        p = int(np.flatnonzero(outside)[0])
        raise ValueError(
            f"G2P moved particle {p} outside background grid: "
            f"new position={new_position[p].tolist()}, "
            f"domain=[{lower.tolist()}, {upper.tolist()}]. "
            "Increase the background grid extent or handle the boundary."
        )

    # Correct only roundoff-sized excursions; never clip real particle motion.
    np.clip(new_position, lower, upper, out=new_position)
    particles.position[:] = new_position
    particles.velocity[:] = new_velocity
