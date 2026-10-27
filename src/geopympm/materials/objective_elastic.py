"""Stage 7: objective in-plane Cauchy-stress integration for 2D plane strain.

Uses an exact incremental in-plane rotation for constant spin, plus a
midpoint-rotated linear elastic stress increment. This is a corotational
integration of the *Jaumann rate of Cauchy stress*, not a finite-strain
hyperelastic energy potential, and not a rotation-corrected particle F.

Engineering strain increment (Stage 3D): [xx, yy, zz, gamma_xy, yz, xz].
Stress convention: tension-positive, [xx, yy, zz, xy, yz, xz] in Pa.
"""
from __future__ import annotations

from collections.abc import Mapping
import math

import numpy as np

from geopympm.core.particles import ParticleSet
from geopympm.kernels.objective_elastic import objective_elastic_plane_strain_kernel
from .linear_elastic import LinearElastic, _particle_moduli


def update_objective_linear_elastic_stress(
    particles: ParticleSet,
    strain_increment: np.ndarray,
    velocity_gradient: np.ndarray,
    materials: LinearElastic | Mapping[int, LinearElastic],
    dt: float,
) -> None:
    """Update ``particles.stress`` atomically with an objective 2D spin update.

    Requires Stage-3D kinematics for the SAME substep, especially the
    spatial velocity gradient. Preserves sigma_zz under pure in-plane rotation,
    while in-plane stress components rotate; shear uses engineering convention.
    """
    if not isinstance(particles, ParticleSet) or particles.ndim != 2:
        raise ValueError("particles must be a 2D ParticleSet.")
    if isinstance(dt, (bool, np.bool_)):
        raise ValueError("dt must be finite and strictly positive.")
    try:
        dt = float(dt)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("dt must be finite and strictly positive.") from exc
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and strictly positive.")

    n = particles.n_particles
    delta = np.asarray(strain_increment, dtype=np.float64)
    L = np.asarray(velocity_gradient, dtype=np.float64)
    s = particles.stress
    if delta.shape != (n, 6) or not np.all(np.isfinite(delta)):
        raise ValueError("strain_increment must be finite and have shape (n_particles, 6).")
    if np.any(delta[:, (2, 4, 5)] != 0.0):
        raise ValueError("Plane strain requires zero zz, yz and xz strain increments.")
    if L.shape != (n, 2, 2) or not np.all(np.isfinite(L)):
        raise ValueError("velocity_gradient must be finite and have shape (n_particles, 2, 2).")
    if s.shape != (n, 6) or not np.all(np.isfinite(s)):
        raise ValueError("particles.stress must be finite with shape (n_particles, 6).")
    if np.any(s[:, (4, 5)] != 0.0):
        raise ValueError("Plane strain requires zero yz and xz initial stresses.")
    # Must agree with Stage-3D symmetric velocity gradient, so the material
    # cannot silently receive inconsistent D and W from unrelated steps.
    from_L = np.empty_like(delta)
    from_L[:] = 0.0
    from_L[:, 0] = dt * L[:, 0, 0]
    from_L[:, 1] = dt * L[:, 1, 1]
    from_L[:, 3] = dt * (L[:, 0, 1] + L[:, 1, 0])
    if not np.allclose(delta, from_L, rtol=2e-11, atol=1e-13):
        raise ValueError("strain_increment is inconsistent with dt * sym(velocity_gradient).")

    lam, mu = _particle_moduli(particles, materials)
    candidate = objective_elastic_plane_strain_kernel(
        np.ascontiguousarray(s), np.ascontiguousarray(delta),
        np.ascontiguousarray(L), lam, mu, dt
    )
    if not np.all(np.isfinite(candidate)):
        raise ValueError("Objective stress update produced nonfinite values; no changes committed.")
    particles.stress[:] = candidate


__all__ = ["update_objective_linear_elastic_stress"]
