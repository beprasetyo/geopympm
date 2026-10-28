"""Numba CPDI1 (Sadeghirad et al., 2011) for a 2D parallelogram domain.

Corners ordered (-,-), (+,-), (+,+), (-,+). CPDI basis weights
are the average of grid Quad4 hats at the four particle corners.
The gradient is the corner-grid-hat weighted mean gradient of the
particle-domain bilinear functions; NOT the average of grid gradients.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def cpdi2d_kernel(corners, a_vec, b_vec, nx, ny, ox, oy, hx, hy):
    n = corners.shape[0]
    ids = np.zeros((n, 16), dtype=np.int64)
    weights = np.zeros((n, 16), dtype=np.float64)
    grad = np.zeros((n, 16, 2), dtype=np.float64)
    signs_x = (-1., 1., 1., -1.)
    signs_y = (-1., -1., 1., 1.)
    for p in range(n):
        ax, ay = a_vec[p, 0], a_vec[p, 1]
        bx, by = b_vec[p, 0], b_vec[p, 1]
        cross = ax * by - ay * bx
        # Positive nondegenerate area and within-domain checks belong to caller.
        xi_x, xi_y = by / cross, -bx / cross
        eta_x, eta_y = -ay / cross, ax / cross
        for c in range(4):
            # The means over the parallelogram of gradients of its Q4
            # basis: 1/4 (sign_x grad(xi) + sign_y grad(eta)).
            dcx = .25 * (signs_x[c] * xi_x + signs_y[c] * eta_x)
            dcy = .25 * (signs_x[c] * xi_y + signs_y[c] * eta_y)
            u = (corners[p, c, 0] - ox) / hx
            v = (corners[p, c, 1] - oy) / hy
            u = min(max(u, 0.), float(nx))
            v = min(max(v, 0.), float(ny))
            ix = min(int(np.floor(u)), nx - 1)
            iy = min(int(np.floor(v)), ny - 1)
            s = u - ix
            t = v - iy
            node00 = iy * (nx + 1) + ix
            # Each of 4 corners contributes to four grid nodes; duplicates
            # are intentionally allowed. Existing P2G/G2P kernels sum them.
            nodes = (node00, node00 + 1, node00 + nx + 2, node00 + nx + 1)
            wn = ((1.-s)*(1.-t), s*(1.-t), s*t, (1.-s)*t)
            for q in range(4):
                k = c * 4 + q
                ids[p, k] = nodes[q]
                weights[p, k] = .25 * wn[q]
                grad[p, k, 0] = dcx * wn[q]
                grad[p, k, 1] = dcy * wn[q]
    return ids, weights, grad
