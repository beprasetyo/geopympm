"""Numba Stage-4 isotropic elastic stress increments for 2D plane strain.

Voigt order [xx, yy, zz, xy, yz, xz], engineering shear gamma_xy.
Tension-positive Cauchy stress. NO objective rotation in this first stage.
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def linear_elastic_plane_strain_kernel(old_stress, strain_increment, lam, mu):
    """Return candidate stress, preserving caller arrays for transactional commit."""
    n = old_stress.shape[0]
    new_stress = np.empty((n, 6), dtype=np.float64)

    for p in range(n):
        exx = strain_increment[p, 0]
        eyy = strain_increment[p, 1]
        gamma_xy = strain_increment[p, 3]
        volumetric_increment = exx + eyy  # plane strain: delta epsilon_zz = 0
        bulk_part = lam[p] * volumetric_increment
        two_mu = 2.0 * mu[p]

        new_stress[p, 0] = old_stress[p, 0] + bulk_part + two_mu * exx
        new_stress[p, 1] = old_stress[p, 1] + bulk_part + two_mu * eyy
        new_stress[p, 2] = old_stress[p, 2] + bulk_part
        new_stress[p, 3] = old_stress[p, 3] + mu[p] * gamma_xy
        new_stress[p, 4] = old_stress[p, 4]
        new_stress[p, 5] = old_stress[p, 5]

    return new_stress
