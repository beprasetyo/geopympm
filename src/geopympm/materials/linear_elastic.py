"""Stage 4: incremental isotropic linear elasticity in 2D plane strain.

This is a small-increment, NON-objective Cauchy-stress update for checking
MPM kinematics and constitutive interfaces. It is NOT a finite-strain
constitutive law for large rotations or production landslide runout.

Conventions:
  - Tension-positive Cauchy stress, SI units (Pa).
  - Voigt [xx, yy, zz, xy, yz, xz].
  - Strain increment uses ENGINEERING shear gamma_xy = 2 epsilon_xy.
  - Plane strain: delta_epsilon_zz = gamma_yz = gamma_xz = 0.

No state history is needed for a linear elastic material except the universal
particle stress. This module deliberately does not alter particle strain, F,
volume, position, or velocity. The step's strain increment comes from Stage 3D.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
import math

import numpy as np

from geopympm.core.particles import ParticleSet
from geopympm.kernels.linear_elastic import linear_elastic_plane_strain_kernel


@dataclass(frozen=True, slots=True)
class LinearElastic:
    """Material parameters for isotropic small-strain elasticity.

    Parameters
    ----------
    young_modulus : float
        Young's modulus E > 0, in Pa when using SI units.
    poisson_ratio : float
        Poisson's ratio -1 < nu < 0.5.
    """

    young_modulus: float
    poisson_ratio: float

    def __post_init__(self) -> None:
        for field_name in ("young_modulus", "poisson_ratio"):
            value = getattr(self, field_name)
            if isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{field_name} must be a finite real number, not bool.")
            try:
                value = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{field_name} must be a finite real number.") from exc
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite.")
            object.__setattr__(self, field_name, value)
        if self.young_modulus <= 0.0:
            raise ValueError("young_modulus must be strictly positive.")
        if not (-1.0 < self.poisson_ratio < 0.5):
            raise ValueError("poisson_ratio must be strictly between -1 and 0.5.")
        if not np.isfinite(self.lame_lambda) or not np.isfinite(self.shear_modulus):
            raise ValueError("Lamé parameters are non-finite; choose safer material parameters.")

    @property
    def shear_modulus(self) -> float:
        return self.young_modulus / (2.0 * (1.0 + self.poisson_ratio))

    @property
    def lame_lambda(self) -> float:
        nu = self.poisson_ratio
        return self.young_modulus * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    @property
    def bulk_modulus(self) -> float:
        return self.young_modulus / (3.0 * (1.0 - 2.0 * self.poisson_ratio))


def _particle_moduli(particles: ParticleSet, materials) -> tuple[np.ndarray, np.ndarray]:
    """Resolve uniform material or a mapping of material IDs to properties.

    In this first version the moduli are expanded per particle for a simple
    Numba signature. Future MaterialRegistry may cache this dispatch.
    """
    n = particles.n_particles
    if isinstance(materials, LinearElastic):
        return (
            np.full(n, materials.lame_lambda, dtype=np.float64),
            np.full(n, materials.shear_modulus, dtype=np.float64),
        )
    if not isinstance(materials, Mapping) or not materials:
        raise TypeError("materials must be a LinearElastic or a nonempty {material_id: LinearElastic} mapping.")

    for material_id, material in materials.items():
        if (isinstance(material_id, (bool, np.bool_)) or
                not isinstance(material_id, (int, np.integer)) or
                not 0 <= int(material_id) <= np.iinfo(np.int32).max):
            raise ValueError("material mapping keys must be nonnegative int32 IDs.")
        if not isinstance(material, LinearElastic):
            raise TypeError("All mapped materials must be LinearElastic instances.")

    particle_ids = particles.material_id
    if (particle_ids.shape != (n,) or
            not np.issubdtype(particle_ids.dtype, np.integer) or
            np.any(particle_ids < 0)):
        raise ValueError("particles.material_id must be a nonnegative integer vector.")

    lam = np.empty(n, dtype=np.float64)
    mu = np.empty(n, dtype=np.float64)
    for material_id in np.unique(particle_ids):
        key = int(material_id)
        if key not in materials:
            raise ValueError(f"No LinearElastic definition for material_id={key}.")
        model = materials[key]
        mask = particle_ids == material_id
        lam[mask] = model.lame_lambda
        mu[mask] = model.shear_modulus
    return lam, mu


def update_linear_elastic_stress(
    particles: ParticleSet,
    strain_increment: np.ndarray,
    materials: LinearElastic | Mapping[int, LinearElastic],
) -> None:
    """Update in-place particle Cauchy stress from a Stage-3D strain increment.

    Use exactly ONCE per kinematics substep. A repeated call with the same
    increment applies the same delta stress twice. No automatic step tracking.

    The increment is a *small-strain approximation* in the spatial frame.
    Rigid rotations of an existing stress tensor are NOT handled. For large
    rotations, implement a consistent objective stress integration before use.

    Validation and kernel calculations finish before committing stress. If
    any particle fails, no particle stress is modified.
    """
    if not isinstance(particles, ParticleSet):
        raise TypeError("particles must be a ParticleSet.")
    if particles.ndim != 2:
        raise ValueError("Stage 4 supports 2D plane strain only.")
    n = particles.n_particles
    delta = np.asarray(strain_increment, dtype=np.float64)
    if delta.shape != (n, 6) or not np.all(np.isfinite(delta)):
        raise ValueError("strain_increment must be finite and have shape (n_particles, 6).")
    if np.any(delta[:, (2, 4, 5)] != 0.0):
        raise ValueError("Plane strain requires zero zz, yz and xz strain increments.")
    if (particles.stress.shape != (n, 6) or
            not np.all(np.isfinite(particles.stress))):
        raise ValueError("particles.stress must be a finite (n_particles, 6) array.")
    if np.any(particles.stress[:, (4, 5)] != 0.0):
        raise ValueError("Plane strain requires initial yz and xz stresses to be zero.")

    lam, mu = _particle_moduli(particles, materials)
    updated = linear_elastic_plane_strain_kernel(
        np.ascontiguousarray(particles.stress),
        np.ascontiguousarray(delta), lam, mu,
    )
    if not np.all(np.isfinite(updated)):
        raise ValueError("Linear elastic stress update produced non-finite values; no changes committed.")
    particles.stress[:] = updated
