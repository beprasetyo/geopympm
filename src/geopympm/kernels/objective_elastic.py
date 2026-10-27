"""Numba 2D-plane-strain corotational-midpoint Jaumann hypoelastic update.

The in-plane spin is W=(L-L.T)/2, with W=[[0,-omega],[omega,0]].
Rotation is integrated exactly over a step with constant W. The elastic
stress rate is approximated at the midpoint of that rotation:
 sigma_new = R(dt) sigma_old R(dt).T + R(dt/2) C:D*dt R(dt/2).T.
This is a rate-form elastic model, NOT finite-strain hyperelasticity.
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _rotate_inplane(xx, yy, xy, angle):
    c = np.cos(angle)
    s = np.sin(angle)
    cc, ss, cs = c*c, s*s, c*s
    return (cc*xx + ss*yy - 2.0*cs*xy,
            ss*xx + cc*yy + 2.0*cs*xy,
            cs*(xx-yy) + (cc-ss)*xy)


@njit(cache=True)
def objective_elastic_plane_strain_kernel(old_stress, strain_increment,
                                          velocity_gradient, lam, mu, dt):
    n = old_stress.shape[0]
    new_stress = np.empty((n, 6), dtype=np.float64)
    for p in range(n):
        w = 0.5*(velocity_gradient[p, 1, 0] - velocity_gradient[p, 0, 1])
        angle = w*dt
        old_xx, old_yy, old_xy = _rotate_inplane(
            old_stress[p, 0], old_stress[p, 1], old_stress[p, 3], angle
        )

        exx = strain_increment[p, 0]
        eyy = strain_increment[p, 1]
        gamma_xy = strain_increment[p, 3]  # ENGINEERING shear
        bulk = lam[p]*(exx+eyy)
        elastic_xx = bulk + 2.0*mu[p]*exx
        elastic_yy = bulk + 2.0*mu[p]*eyy
        elastic_xy = mu[p]*gamma_xy
        inc_xx, inc_yy, inc_xy = _rotate_inplane(
            elastic_xx, elastic_yy, elastic_xy, 0.5*angle
        )
        new_stress[p, 0] = old_xx + inc_xx
        new_stress[p, 1] = old_yy + inc_yy
        new_stress[p, 2] = old_stress[p, 2] + bulk
        new_stress[p, 3] = old_xy + inc_xy
        new_stress[p, 4] = old_stress[p, 4]
        new_stress[p, 5] = old_stress[p, 5]
    return new_stress
