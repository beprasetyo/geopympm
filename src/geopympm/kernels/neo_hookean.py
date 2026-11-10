"""Stage 11: Numba plane-strain compressible Neo-Hookean UCNH elastic kernel.

F is 2x2, extended to 3x3 using Fzz=1, no out-of-plane shear.
Stress is tension-positive Cauchy in [xx,yy,zz,xy,yz,xz].
Energy density is per INITIAL particle volume, in J/m^3.
"""
from __future__ import annotations

import math
import numpy as np
from numba import njit


@njit(cache=True)
def cnh_plane_strain_kernel(F, bulk, shear):
    """Compute candidate stress and reference energy; no state mutation."""
    n = F.shape[0]
    stress = np.empty((n, 6), dtype=np.float64)
    energy = np.empty(n, dtype=np.float64)
    for p in range(n):
        a, b = F[p, 0, 0], F[p, 0, 1]
        c, d = F[p, 1, 0], F[p, 1, 1]
        J = a * d - b * c
        if not math.isfinite(J) or J <= 0.0:
            raise ValueError("Neo-Hookean requires a positive finite det(F)")
        j23 = J ** (-2.0 / 3.0)
        bxx = (a*a + b*b) * j23
        byy = (c*c + d*d) * j23
        bzz = j23
        bxy = (a*c + b*d) * j23
        trace3 = bxx + byy + bzz
        q = trace3 / 3.0
        hydro = 0.5 * bulk[p] * (J - 1.0 / J)
        s = shear[p] / J
        stress[p, 0] = hydro + s*(bxx - q)
        stress[p, 1] = hydro + s*(byy - q)
        stress[p, 2] = hydro + s*(bzz - q)
        stress[p, 3] = s*bxy
        stress[p, 4] = 0.0
        stress[p, 5] = 0.0
        U = 0.5*bulk[p]*(0.5*(J*J - 1.0) - math.log(J))
        W = 0.5*shear[p]*(trace3 - 3.0)
        energy[p] = U + W
    return stress, energy
