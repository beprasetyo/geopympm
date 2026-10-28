"""Stage 10B: classical axis-aligned contiguous-particle GIMP domain evolution.

The reference ``domain_size0`` is FULL particle-side length at the MPM
reference configuration. For the cumulative in-plane deformation gradient F,

    length_x = length_x0 * F_xx
    length_y = length_y0 * F_yy

This classical approximation is appropriate for positive diagonal stretches
aligned with the grid, NOT general rotations or large shear. It should not be
silently confused with rotation-invariant principal stretches or CPDI.

Candidates are checked against the finite 4x4 support (length <= cell_size)
and grid boundary before committing. Domain size is used in interpolation at
the NEXT step; the current step's G2P retains its pre-step shape functions.
"""
from __future__ import annotations

import numpy as np
from geopympm.core.particles import ParticleSet
from geopympm.core.grid import BackgroundGrid
from geopympm.interpolation.gimp import validate_gimp_domain
from geopympm.kernels.domain_update import cpgimp_domain_from_F_kernel


def candidate_cpgimp_domain(particles: ParticleSet, grid: BackgroundGrid,
                            *, position=None) -> np.ndarray:
    """Return checked candidate full (n,2) lengths from cumulative diag(F).

    Does not alter particle state. An invalid F diagonal or domain that exceeds
    16-node GIMP support is an explicit error, NOT clamped/renormalized.
    """
    if not isinstance(particles, ParticleSet) or not isinstance(grid, BackgroundGrid):
        raise TypeError("cpGIMP requires ParticleSet and BackgroundGrid.")
    if particles.ndim != 2 or grid.ndim != 2:
        raise ValueError("Stage 10B cpGIMP requires two-dimensional particles and grid.")
    F = particles.deformation_gradient
    if F.shape != (particles.n_particles, 2, 2) or not np.all(np.isfinite(F)):
        raise ValueError("cpGIMP requires a finite (n,2,2) deformation gradient.")
    det = F[:, 0, 0] * F[:, 1, 1] - F[:, 0, 1] * F[:, 1, 0]
    if np.any(~np.isfinite(det)) or np.any(det <= 0.0):
        raise ValueError("cpGIMP requires det(F)>0 for every particle.")
    if np.any(F[:, 0, 0] <= 0.0) or np.any(F[:, 1, 1] <= 0.0):
        raise ValueError("cpGIMP requires F_xx>0 and F_yy>0 (axis-aligned domain); "
                         "rotation/shear may require CPDI.")
    reference = particles.domain_size0
    if (reference.shape != (particles.n_particles, 2) or
            not np.all(np.isfinite(reference)) or np.any(reference <= 0.0)):
        raise ValueError("cpGIMP requires initialized positive domain_size0 (n,2).")
    candidate = cpgimp_domain_from_F_kernel(
        np.ascontiguousarray(reference), np.ascontiguousarray(F)
    )
    validate_gimp_domain(particles, grid, position=position, domain_size=candidate)
    return candidate


def update_cpgimp_domain(particles: ParticleSet, grid: BackgroundGrid) -> np.ndarray:
    """Atomically update current full lengths after kinematics AND G2P motion."""
    candidate = candidate_cpgimp_domain(particles, grid)
    particles.domain_size[:] = candidate
    return particles.domain_size
