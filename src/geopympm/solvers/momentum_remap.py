"""Stage 12 MUSL second (updated-particle-momentum) particle-to-grid remap.

Remapping is evaluated on the OLD particle positions x_n, using the original
particle interpolation. This is not an additional physical force update.
Grid mass is unchanged, grid acceleration retains the force-step value,
and prescribed-velocity constraints remain enforced.
"""
from __future__ import annotations

import numpy as np
from geopympm.kernels.p2g import p2g_mass_momentum_kernel, recover_nodal_velocity_kernel


def remap_updated_particle_momentum(particles, grid, interpolation) -> None:
    """Reassemble nodal momentum from newly updated particle velocities."""
    # Input interpolation is prevalidated by the first P2G and G2P phase.
    if not np.all(np.isfinite(particles.velocity)):
        raise ValueError("Cannot remap nonfinite particle velocity")
    new_mass = np.zeros_like(grid.mass)
    new_momentum = np.zeros_like(grid.momentum)
    p2g_mass_momentum_kernel(interpolation.node_ids, interpolation.weights,
                             particles.mass, particles.velocity, new_mass,
                             new_momentum)
    if not np.allclose(new_mass, grid.mass, rtol=1e-12, atol=1e-13):
        raise ValueError("MUSL second projection changed grid mass")
    new_velocity = np.empty_like(grid.velocity)
    recover_nodal_velocity_kernel(new_mass, new_momentum, new_velocity)
    mask = grid.velocity_bc_mask
    new_velocity[mask] = grid.velocity_bc_value[mask]
    new_momentum[mask] = (new_mass[:, None] * new_velocity)[mask]
    if not np.all(np.isfinite(new_velocity)) or not np.all(np.isfinite(new_momentum)):
        raise ValueError("MUSL second projection produced nonfinite state")
    grid.momentum[:] = new_momentum
    grid.velocity[:] = new_velocity

__all__ = ["remap_updated_particle_momentum"]
