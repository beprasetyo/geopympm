"""Array-based particle state for GeoPyMPM.

The ParticleSet stores *universal* particle variables only. Constitutive-model
internal variables belong in the material subsystem, not in this class.

The Voigt component order for both stress and strain is:
    [xx, yy, zz, xy, yz, xz]
Strain shear components use engineering shear strains (gamma_ij).
Stress components are Cauchy stresses; the proposed internal convention is
positive tension. Conversion from external solvers belongs in transfer/.
"""

from __future__ import annotations

import numpy as np


def _float_array(values, shape: tuple[int, ...] | None, name: str) -> np.ndarray:
    """Return an independent C-contiguous float64 array with finite entries."""
    array = np.asarray(values, dtype=np.float64)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")
    return np.array(array, dtype=np.float64, order="C", copy=True)


class ParticleSet:
    """Collection of MPM particles stored as NumPy arrays (SoA).

    Parameters
    ----------
    position : array_like, shape (n, ndim)
        Initial particle coordinates, with ndim=2 or 3.
    mass : array_like, shape (n,)
        Particle mass. For 2D plane strain, use a consistent unit thickness.
    volume : array_like, shape (n,)
        Initial particle volume (including the selected 2D thickness).
    velocity : array_like, shape (n, ndim), optional
        Initial particle velocities, default zero.
    material_id : array_like, shape (n,), optional
        Nonnegative integer material IDs, default zero.
    domain_size : array_like, shape (n, ndim), optional
        Initial full side lengths of rectangular GIMP particle domains.
        If omitted, values are zero until a GIMP-specific setup supplies them.
    stress, strain : array_like, shape (n, 6), optional
        Initial Voigt stress and strain components, default zero.
    """

    def __init__(
        self,
        position,
        mass,
        volume,
        *,
        velocity=None,
        material_id=None,
        domain_size=None,
        stress=None,
        strain=None,
    ) -> None:
        position_array = _float_array(position, None, "position")
        if position_array.ndim != 2 or position_array.shape[1] not in (2, 3):
            raise ValueError("position must have shape (n, 2) or (n, 3).")
        if position_array.shape[0] == 0:
            raise ValueError("ParticleSet requires at least one particle.")

        n, ndim = position_array.shape
        self.position = position_array
        self.position0 = position_array.copy()
        self.mass = _float_array(mass, (n,), "mass")
        self.volume = _float_array(volume, (n,), "volume")
        if np.any(self.mass <= 0.0):
            raise ValueError("mass values must be strictly positive.")
        if np.any(self.volume <= 0.0):
            raise ValueError("volume values must be strictly positive.")
        self.volume0 = self.volume.copy()

        self.velocity = (
            np.zeros((n, ndim), dtype=np.float64)
            if velocity is None
            else _float_array(velocity, (n, ndim), "velocity")
        )

        if material_id is None:
            self.material_id = np.zeros(n, dtype=np.int32)
        else:
            ids = np.asarray(material_id)
            if ids.shape != (n,) or not np.issubdtype(ids.dtype, np.integer):
                raise ValueError("material_id must be an integer array of shape (n,).")
            if np.any(ids < 0) or np.any(ids > np.iinfo(np.int32).max):
                raise ValueError("material_id must fit in nonnegative int32 range.")
            self.material_id = np.array(ids, dtype=np.int32, order="C", copy=True)

        self.stress = (
            np.zeros((n, 6), dtype=np.float64)
            if stress is None
            else _float_array(stress, (n, 6), "stress")
        )
        self.strain = (
            np.zeros((n, 6), dtype=np.float64)
            if strain is None
            else _float_array(strain, (n, 6), "strain")
        )

        self.domain_size = (
            np.zeros((n, ndim), dtype=np.float64)
            if domain_size is None
            else _float_array(domain_size, (n, ndim), "domain_size")
        )
        if domain_size is not None and np.any(self.domain_size <= 0.0):
            raise ValueError("Provided GIMP domain_size values must be positive.")
        self.domain_size0 = self.domain_size.copy()

        # Kinematic quantity, independent of a particular constitutive model.
        self.deformation_gradient = np.repeat(
            np.eye(ndim, dtype=np.float64)[None, :, :], n, axis=0
        )

    @property
    def n_particles(self) -> int:
        return self.position.shape[0]

    @property
    def ndim(self) -> int:
        return self.position.shape[1]

    @property
    def density(self) -> np.ndarray:
        """Current densities. Mass is constant, volume may change in the solver."""
        return self.mass / self.volume

    @property
    def displacement(self) -> np.ndarray:
        """Particle displacement relative to the initialized MPM configuration."""
        return self.position - self.position0

    @property
    def has_gimp_domains(self) -> bool:
        """True if all full particle-domain side lengths have been initialized."""
        return bool(np.all(self.domain_size > 0.0))

    def __repr__(self) -> str:
        return f"ParticleSet(n_particles={self.n_particles}, ndim={self.ndim})"
