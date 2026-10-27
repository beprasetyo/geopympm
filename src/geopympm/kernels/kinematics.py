"""Stage 3D: 2D Quad4 particle kinematics; Numba-only numerical loops.

The update uses the spatial velocity gradient at pre-advection particle positions:
    L_p = sum_a v_(node_a) tensor grad N_(p,a)
    F_(n+1) = (I + dt L_p) F_n
    V_(n+1) = det(F_(n+1)) V_0

The strain field in Stage 3D is a DIAGNOSTIC accumulation of D dt, with
engineering xy shear. It is not an objective finite-strain measure. The
constitutive integrator must select its own consistent strain/stress measure.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def particle_kinematics_quad4_kernel(node_ids, gradients, node_velocity,
                                      old_F, volume0, old_strain, dt):
    """Compute candidate arrays without touching the caller's particle state."""
    n = node_ids.shape[0]
    velocity_gradient = np.empty((n, 2, 2), dtype=np.float64)
    strain_increment = np.zeros((n, 6), dtype=np.float64)
    new_F = np.empty((n, 2, 2), dtype=np.float64)
    new_volume = np.empty(n, dtype=np.float64)
    new_strain = np.empty((n, 6), dtype=np.float64)
    jacobian = np.empty(n, dtype=np.float64)

    for p in range(n):
        l00 = 0.0
        l01 = 0.0
        l10 = 0.0
        l11 = 0.0

        for a in range(4):
            i = node_ids[p, a]
            gx = gradients[p, a, 0]
            gy = gradients[p, a, 1]
            vx = node_velocity[i, 0]
            vy = node_velocity[i, 1]
            l00 += vx * gx
            l01 += vx * gy
            l10 += vy * gx
            l11 += vy * gy

        velocity_gradient[p, 0, 0] = l00
        velocity_gradient[p, 0, 1] = l01
        velocity_gradient[p, 1, 0] = l10
        velocity_gradient[p, 1, 1] = l11

        # Eulerian rate-of-deformation increment (engineering Voigt).
        strain_increment[p, 0] = dt * l00
        strain_increment[p, 1] = dt * l11
        strain_increment[p, 3] = dt * (l01 + l10)

        # Forward-Euler discretization of dot(F) = L F.
        a00 = 1.0 + dt * l00
        a01 = dt * l01
        a10 = dt * l10
        a11 = 1.0 + dt * l11

        f00 = a00 * old_F[p, 0, 0] + a01 * old_F[p, 1, 0]
        f01 = a00 * old_F[p, 0, 1] + a01 * old_F[p, 1, 1]
        f10 = a10 * old_F[p, 0, 0] + a11 * old_F[p, 1, 0]
        f11 = a10 * old_F[p, 0, 1] + a11 * old_F[p, 1, 1]
        new_F[p, 0, 0] = f00
        new_F[p, 0, 1] = f01
        new_F[p, 1, 0] = f10
        new_F[p, 1, 1] = f11

        jp = f00 * f11 - f01 * f10
        jacobian[p] = jp
        new_volume[p] = jp * volume0[p]
        for c in range(6):
            new_strain[p, c] = old_strain[p, c] + strain_increment[p, c]

    return velocity_gradient, strain_increment, new_F, jacobian, new_volume, new_strain
