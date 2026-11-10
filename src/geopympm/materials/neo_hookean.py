"""Stage 11: compressible Neo-Hookean for 2D plane-strain explicit MPM.

Matches the *elastic, isothermal* UCNH stress and energy branch in Uintah
src/CCA/Components/MPM/Materials/ConstitutiveModel/UCNH.cc.
Does not implement UCNH's plastic, thermal or artificial-viscosity options.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

import numpy as np

from geopympm.core.particles import ParticleSet
from geopympm.kernels.neo_hookean import cnh_plane_strain_kernel


@dataclass(frozen=True, slots=True)
class CompressibleNeoHookean:
    """Hyperelastic parameters K and G in Pa, both strictly positive."""

    bulk_modulus: float
    shear_modulus: float

    def __post_init__(self) -> None:
        for name in ("bulk_modulus", "shear_modulus"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{name} must be a finite positive real")
            try:
                value = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{name} must be a finite positive real") from exc
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and strictly positive")
            object.__setattr__(self, name, value)

    @classmethod
    def from_young_poisson(cls, young_modulus: float, poisson_ratio: float):
        """Construct CNH with the same infinitesimal moduli as LinearElastic."""
        # Reuse validated, exact LinearElastic conversion.
        from .linear_elastic import LinearElastic
        linear = LinearElastic(young_modulus, poisson_ratio)
        return cls(linear.bulk_modulus, linear.shear_modulus)

    @property
    def lame_lambda(self) -> float:
        return self.bulk_modulus - (2.0 / 3.0) * self.shear_modulus

    @property
    def young_modulus(self) -> float:
        K, G = self.bulk_modulus, self.shear_modulus
        return 9.0 * K * G / (3.0 * K + G)

    @property
    def poisson_ratio(self) -> float:
        K, G = self.bulk_modulus, self.shear_modulus
        return (3.0 * K - 2.0 * G) / (2.0 * (3.0 * K + G))


def _particle_cnh_moduli(particles: ParticleSet, materials) -> tuple[np.ndarray, np.ndarray]:
    """Resolve particle K/G; reject missing IDs and heterogeneous model kinds."""
    n = particles.n_particles
    if type(materials) is CompressibleNeoHookean:
        return (np.full(n, materials.bulk_modulus, dtype=np.float64),
                np.full(n, materials.shear_modulus, dtype=np.float64))
    if not isinstance(materials, Mapping) or not materials:
        raise TypeError("materials must be CompressibleNeoHookean or nonempty material-ID mapping")
    for key, material in materials.items():
        if (isinstance(key, (bool, np.bool_)) or not isinstance(key, (int, np.integer))
                or not 0 <= int(key) <= np.iinfo(np.int32).max):
            raise ValueError("material mapping keys must be nonnegative int32 IDs")
        if type(material) is not CompressibleNeoHookean:
            raise TypeError("All mapped models must be CompressibleNeoHookean for a CNH solver")
    ids = particles.material_id
    if (ids.shape != (n,) or not np.issubdtype(ids.dtype, np.integer)
            or np.any(ids < 0)):
        raise ValueError("Invalid particle material_id vector")
    K = np.empty(n, dtype=np.float64)
    G = np.empty(n, dtype=np.float64)
    for key in np.unique(ids):
        i = int(key)
        if i not in materials:
            raise ValueError(f"No CompressibleNeoHookean definition for material_id={i}")
        mask = ids == key
        K[mask] = materials[i].bulk_modulus
        G[mask] = materials[i].shear_modulus
    return K, G


def evaluate_neo_hookean(particles: ParticleSet, materials) -> tuple[np.ndarray, np.ndarray]:
    """Return new Cauchy stress and strain-energy density per reference volume.

    Does NOT mutate particles. The reference configuration is F=I at MPM
    initialization, with zero elastic initial stress. Pre-stressed FEM transfer
    requires a deliberate elastic reference/prestress convention in future.
    """
    if not isinstance(particles, ParticleSet) or particles.ndim != 2:
        raise TypeError("CNH requires a 2D ParticleSet")
    F = particles.deformation_gradient
    n = particles.n_particles
    if F.shape != (n, 2, 2) or not np.all(np.isfinite(F)):
        raise ValueError("CNH requires finite deformation gradients (n,2,2)")
    J = np.linalg.det(F)
    if not np.all(np.isfinite(J)) or np.any(J <= 0.0):
        raise ValueError("CNH requires det(F)>0")
    if (particles.volume.shape != (n,) or particles.volume0.shape != (n,)
            or np.any(particles.volume0 <= 0.0)
            or not np.allclose(particles.volume, particles.volume0*J, rtol=1e-10, atol=1e-14)):
        raise ValueError("CNH volume must satisfy V = J*V0")
    K, G = _particle_cnh_moduli(particles, materials)
    stress, energy = cnh_plane_strain_kernel(F, K, G)
    if not np.all(np.isfinite(stress)) or not np.all(np.isfinite(energy)):
        raise ValueError("Nonfinite CNH stress or energy")
    return stress, energy


def update_neo_hookean_stress(particles: ParticleSet, materials) -> None:
    """Replace Cauchy stress from updated F; atomic on input-validation failure."""
    stress, _ = evaluate_neo_hookean(particles, materials)
    particles.stress[:] = stress


def neo_hookean_strain_energy(particles: ParticleSet, materials) -> float:
    """Compute total stored hyperelastic energy, sum(V0 * Psi(F)), in Joules."""
    _, energy = evaluate_neo_hookean(particles, materials)
    return float(np.dot(particles.volume0, energy))
