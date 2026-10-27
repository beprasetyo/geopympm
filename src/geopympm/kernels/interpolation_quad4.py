"""Numba kernel for standard 2D MPM Quad4 interpolation.

The kernel works only with typed NumPy arrays and scalars.  Domain searches,
input validation and policy for points on/outside boundaries are deliberately
handled by :mod:`geopympm.interpolation.mpm`.

Node order (counterclockwise): bottom-left, bottom-right, top-right, top-left.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def quad4_kernel(cell_indices, local_coordinates, nx, dx, dy):
    """Compute cell-node IDs, shape values and spatial shape gradients.

    Parameters
    ----------
    cell_indices : int64[:, ::1], shape (n, 2)
        Zero-based cell indices (ix, iy).
    local_coordinates : float64[:, ::1], shape (n, 2)
        Local coordinates (s, t) in [0,1]^2 within each particle's cell.
    nx : int
        Number of *cells* along x (there are nx+1 nodes per row).
    dx, dy : float
        Positive cell sizes.

    Returns
    -------
    node_ids : int64[:, :], shape (n, 4)
    weights : float64[:, :], shape (n, 4)
    gradients : float64[:, :, :], shape (n, 4, 2)
        Spatial dN/dx and dN/dy in the global Cartesian system.
    """
    n = cell_indices.shape[0]
    node_ids = np.empty((n, 4), dtype=np.int64)
    weights = np.empty((n, 4), dtype=np.float64)
    gradients = np.empty((n, 4, 2), dtype=np.float64)

    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy

    for p in range(n):
        ix = cell_indices[p, 0]
        iy = cell_indices[p, 1]
        s = local_coordinates[p, 0]
        t = local_coordinates[p, 1]
        one_minus_s = 1.0 - s
        one_minus_t = 1.0 - t

        bottom_left = iy * (nx + 1) + ix
        node_ids[p, 0] = bottom_left
        node_ids[p, 1] = bottom_left + 1
        node_ids[p, 2] = bottom_left + nx + 2
        node_ids[p, 3] = bottom_left + nx + 1

        weights[p, 0] = one_minus_s * one_minus_t
        weights[p, 1] = s * one_minus_t
        weights[p, 2] = s * t
        weights[p, 3] = one_minus_s * t

        gradients[p, 0, 0] = -one_minus_t * inv_dx
        gradients[p, 1, 0] = one_minus_t * inv_dx
        gradients[p, 2, 0] = t * inv_dx
        gradients[p, 3, 0] = -t * inv_dx

        gradients[p, 0, 1] = -one_minus_s * inv_dy
        gradients[p, 1, 1] = -s * inv_dy
        gradients[p, 2, 1] = s * inv_dy
        gradients[p, 3, 1] = one_minus_s * inv_dy

    return node_ids, weights, gradients
