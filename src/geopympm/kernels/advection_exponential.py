"""Stage 9: Numba local-affine exponential particle-position integration.

With frozen nodal velocity field around a particle position x_n, let
v(x) ~= v_p + L_p (x - x_n), where L_p is the spatial velocity gradient.
The exact increment for that *frozen affine field* is

    x_{n+1} = x_n + dt * phi1(dt * L_p) @ v_p,
    phi1(B) = integral_0^1 exp(s B) ds = sum_{k>=0} B^k/(k+1)!.

Unlike L^{-1}(exp(dt L)-I), phi1 remains well defined for singular L
(including rigid translation and simple shear). This is an advection
integrator, NOT a substitute for updating FLIP/PIC particle velocity, and is
not globally exact for bilinear Quad4 velocities across grid-cell boundaries.
"""
from __future__ import annotations

import math
import numpy as np
from numba import njit
from geopympm.kernels.kinematics_exponential import _matrix_exponential_2x2


@njit(cache=True)
def _phi1_2x2(b00: float, b01: float, b10: float, b11: float):
    """Stable 2x2 phi1 via small-matrix series and doubling.

    For A=B/2**n, evaluate phi1(A) by Taylor expansion; recover phi1(2A)
    from phi1(A) using phi1(2A)=(I+exp(A))@phi1(A)/2. Scaling keeps
    Taylor terms small for moderate/large B without inverting B.
    """
    max_b = max(abs(b00), abs(b01), abs(b10), abs(b11))
    nscale = 0
    while max_b > 0.25 and nscale < 1024:
        max_b *= 0.5
        nscale += 1
    factor = 2.0 ** (-nscale)
    a00, a01, a10, a11 = b00*factor, b01*factor, b10*factor, b11*factor

    # P = I initially; phi1 = I + A/2! + A^2/3! + ...
    p00 = 1.0; p01 = 0.0; p10 = 0.0; p11 = 1.0
    phi00 = 1.0; phi01 = 0.0; phi10 = 0.0; phi11 = 1.0
    for k in range(1, 65):
        # P <- P @ A / (k+1); P is A^k/(k+1)!
        q00 = (p00*a00 + p01*a10)/(k+1)
        q01 = (p00*a01 + p01*a11)/(k+1)
        q10 = (p10*a00 + p11*a10)/(k+1)
        q11 = (p10*a01 + p11*a11)/(k+1)
        p00,p01,p10,p11 = q00,q01,q10,q11
        phi00 += p00; phi01 += p01; phi10 += p10; phi11 += p11
        if max(abs(p00),abs(p01),abs(p10),abs(p11)) < 2.e-17:
            break

    # Recover phi1(B) by nscale doublings. At each level exp(A)
    # is evaluated from the same current A, with no matrix inverse.
    for _ in range(nscale):
        e00,e01,e10,e11 = _matrix_exponential_2x2(a00,a01,a10,a11)
        n00 = 0.5*((1.0+e00)*phi00 + e01*phi10)
        n01 = 0.5*((1.0+e00)*phi01 + e01*phi11)
        n10 = 0.5*(e10*phi00 + (1.0+e11)*phi10)
        n11 = 0.5*(e10*phi01 + (1.0+e11)*phi11)
        phi00,phi01,phi10,phi11 = n00,n01,n10,n11
        a00,a01,a10,a11 = 2*a00,2*a01,2*a10,2*a11
    return phi00,phi01,phi10,phi11


@njit(cache=True)
def affine_exponential_positions_kernel(node_ids, weights, old_position,
                                       node_velocity, velocity_gradient, dt):
    """Compute independent particle positions from frozen local affine flow."""
    n = old_position.shape[0]
    positions = np.empty((n, 2), dtype=np.float64)
    for p in range(n):
        vx = 0.0; vy = 0.0
        for a in range(4):
            i = node_ids[p,a]
            w = weights[p,a]
            vx += w*node_velocity[i,0]
            vy += w*node_velocity[i,1]
        b00 = dt*velocity_gradient[p,0,0]
        b01 = dt*velocity_gradient[p,0,1]
        b10 = dt*velocity_gradient[p,1,0]
        b11 = dt*velocity_gradient[p,1,1]
        f00,f01,f10,f11 = _phi1_2x2(b00,b01,b10,b11)
        positions[p,0] = old_position[p,0] + dt*(f00*vx+f01*vy)
        positions[p,1] = old_position[p,1] + dt*(f10*vx+f11*vy)
    return positions
