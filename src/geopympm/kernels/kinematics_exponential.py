"""Stage 8: Numba 2-D exponential-map deformation-gradient integrator.

For a frozen spatial velocity gradient L on a substep:
    F(n+1) = exp(dt*L) F(n)
    det(F(n+1)) = exp(dt*trace(L)) det(F(n)) > 0

The closed-form real 2x2 exponential avoids a SciPy dependency inside JIT
kernels. This updates *only* deformation kinematics. It does not change the
strain-rate approximation or particle-position integration scheme.
"""
from __future__ import annotations

import math
import numpy as np
from numba import njit


@njit(cache=True)
def _matrix_exponential_2x2(b00, b01, b10, b11):
    """Compute exp(B) real 2x2 using traceless decomposition.

    B = a I + K, K^2 = q I; q may be positive, negative, or ~0.
    exp(B) = exp(a)[c(q) I + s(q) K],
    where c(q)=cosh(sqrt(q)) or cos(sqrt(-q));
          s(q)=sinh(sqrt(q))/sqrt(q) or sin(sqrt(-q))/sqrt(-q).
    Taylor series used around q=0 to avoid 0/0 and loss of significance.
    """
    a = 0.5 * (b00 + b11)
    d = 0.5 * (b00 - b11)
    q = d * d + b01 * b10
    if abs(q) < 1.0e-8:
        q2 = q*q
        q3 = q2*q
        c = 1.0 + q/2.0 + q2/24.0 + q3/720.0
        s = 1.0 + q/6.0 + q2/120.0 + q3/5040.0
    elif q > 0.0:
        x = math.sqrt(q)
        c = math.cosh(x)
        s = math.sinh(x)/x
    else:
        x = math.sqrt(-q)
        c = math.cos(x)
        s = math.sin(x)/x
    fac = math.exp(a)
    return fac*(c+s*d), fac*(s*b01), fac*(s*b10), fac*(c-s*d)


@njit(cache=True)
def particle_kinematics_quad4_exponential_kernel(node_ids, gradients, node_velocity,
                                                 old_F, volume0, old_strain, dt):
    """Return candidates (L, delta_eps, F_new, J, V_new, strain_new), no mutation."""
    n = node_ids.shape[0]
    velocity_gradient = np.empty((n,2,2), dtype=np.float64)
    strain_increment = np.zeros((n,6), dtype=np.float64)
    new_F = np.empty((n,2,2), dtype=np.float64)
    jacobian = np.empty(n, dtype=np.float64)
    new_volume = np.empty(n, dtype=np.float64)
    new_strain = np.empty((n,6), dtype=np.float64)

    for p in range(n):
        l00 = l01 = l10 = l11 = 0.0
        for a in range(4):
            i = node_ids[p,a]
            gx = gradients[p,a,0]
            gy = gradients[p,a,1]
            vx = node_velocity[i,0]
            vy = node_velocity[i,1]
            l00 += vx*gx
            l01 += vx*gy
            l10 += vy*gx
            l11 += vy*gy
        velocity_gradient[p,0,0] = l00
        velocity_gradient[p,0,1] = l01
        velocity_gradient[p,1,0] = l10
        velocity_gradient[p,1,1] = l11
        strain_increment[p,0] = dt*l00
        strain_increment[p,1] = dt*l11
        strain_increment[p,3] = dt*(l01+l10)
        a00,a01,a10,a11 = _matrix_exponential_2x2(dt*l00,dt*l01,dt*l10,dt*l11)
        f00 = a00*old_F[p,0,0] + a01*old_F[p,1,0]
        f01 = a00*old_F[p,0,1] + a01*old_F[p,1,1]
        f10 = a10*old_F[p,0,0] + a11*old_F[p,1,0]
        f11 = a10*old_F[p,0,1] + a11*old_F[p,1,1]
        new_F[p,0,0] = f00
        new_F[p,0,1] = f01
        new_F[p,1,0] = f10
        new_F[p,1,1] = f11
        jp = f00*f11 - f01*f10
        jacobian[p] = jp
        new_volume[p] = volume0[p]*jp
        for c in range(6):
            new_strain[p,c] = old_strain[p,c] + strain_increment[p,c]
    return velocity_gradient, strain_increment, new_F, jacobian, new_volume, new_strain
