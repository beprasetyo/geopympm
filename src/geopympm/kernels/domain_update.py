"""Numba numerical kernel for Stage 10B axis-aligned cpGIMP domain lengths."""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def cpgimp_domain_from_F_kernel(domain_size0, F):
    """Full current axis-aligned lengths: l_x0*F_xx, l_y0*F_yy."""
    n = domain_size0.shape[0]
    updated = np.empty((n, 2), dtype=np.float64)
    for p in range(n):
        updated[p, 0] = domain_size0[p, 0] * F[p, 0, 0]
        updated[p, 1] = domain_size0[p, 1] * F[p, 1, 1]
    return updated
