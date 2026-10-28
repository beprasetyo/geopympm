"""Stage 10: 2D axis-aligned, unchanged-domain GIMP basis on a Quad4 grid.

The basis is the *domain average* of the standard one-dimensional nodal
linear hat over each full particle side length.  2D tensor product provides
weights and exact spatial derivatives.  The explicit 4x4 candidate stencil
is padded with zero-valued slots; at most 9 slots contribute for lp<=h.

Zero-weight outer stencil slots use clamped *valid* node IDs, never negative
sentinels, so standard Numba P2G/G2P loops can read them safely.
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _hat(r):
    ar = abs(r)
    return max(0.0, 1.0-ar)


@njit(cache=True)
def _weight_and_derivative(center, length, node, h):
    """Stable exact average of the 1-D hat over one finite particle side.

    Integrate the piecewise-linear hat by exact trapezoids with splits at
    -1, 0, +1; this avoids cancellation of near-equal primitives for narrow
    domains. Center and node are expressed in units of h.
    """
    width = length / h
    a = center - node - 0.5*width
    b = center - node + 0.5*width
    integral = 0.0
    left = a
    for knot in (-1.0, 0.0, 1.0):
        right = min(knot, b)
        if right > left:
            integral += 0.5 * (right - left) * (_hat(left) + _hat(right))
        left = max(left, right)
    if b > left:
        integral += 0.5 * (b-left) * (_hat(left) + _hat(b))
    # Use b-a instead of length/h only in the weight to avoid minuscule
    # discrepancy when subtracting the endpoints at machine precision.
    w = integral/(b-a)
    dw = (_hat(b) - _hat(a)) / length
    return w, dw


@njit(cache=True)
def gimp2d_kernel(cell_indices, uv, domain_size, nx, ny, dx, dy):
    """Return valid (n,16) IDs, (n,16) weights, (n,16,2) gradients.

    uv are normalized positions relative to grid origin. Domain full lengths
    are in physical units. Inputs are checked by the Python wrapper.
    """
    n = cell_indices.shape[0]
    ids = np.empty((n, 16), dtype=np.int64)
    weights = np.zeros((n, 16), dtype=np.float64)
    gradients = np.zeros((n, 16, 2), dtype=np.float64)
    for p in range(n):
        ix0 = cell_indices[p, 0]
        iy0 = cell_indices[p, 1]
        wx = np.empty(4, dtype=np.float64)
        gx = np.empty(4, dtype=np.float64)
        wy = np.empty(4, dtype=np.float64)
        gy = np.empty(4, dtype=np.float64)
        for a in range(4):
            jx = ix0 + a - 1
            jy = iy0 + a - 1
            wx[a], gx[a] = _weight_and_derivative(
                uv[p, 0], domain_size[p, 0], jx, dx
            )
            wy[a], gy[a] = _weight_and_derivative(
                uv[p, 1], domain_size[p, 1], jy, dy
            )
            if jx < 0 or jx > nx:
                wx[a] = 0.0
                gx[a] = 0.0
            if jy < 0 or jy > ny:
                wy[a] = 0.0
                gy[a] = 0.0
        for ay in range(4):
            jy = iy0 + ay - 1
            jy = max(0, min(jy, ny))
            for ax in range(4):
                jx = ix0 + ax - 1
                jx = max(0, min(jx, nx))
                k = ay*4 + ax
                ids[p, k] = jy*(nx+1) + jx
                weights[p, k] = wx[ax]*wy[ay]
                gradients[p, k, 0] = gx[ax]*wy[ay]
                gradients[p, k, 1] = wx[ax]*gy[ay]
    return ids, weights, gradients
