"""Structured Cartesian background grid for the initial 2D MPM implementation.

This is a *state container*, not an MPM solver. The grid holds geometry,
transient nodal quantities, and prescribed-velocity boundary-condition data.

Node numbering is zero-based with x varying fastest:
    node_id(ix, iy) = iy * (nx + 1) + ix
where ``(nx, ny)`` are the numbers of cells in x and y.

For example, n_cells=(2, 1) yields:
    3 -- 4 -- 5       y = origin_y + dy
    |    |    |
    0 -- 1 -- 2       y = origin_y
"""

from __future__ import annotations

import numpy as np


class BackgroundGrid:
    """Uniform, axis-aligned 2D background grid with array-based nodal state.

    Parameters
    ----------
    origin : array_like, shape (2,)
        Coordinates of the lower-left grid node (m for SI units).
    cell_size : array_like, shape (2,)
        Grid spacing (dx, dy), both strictly positive.
    n_cells : sequence of two integers
        Number of grid *cells* (nx, ny), both strictly positive. There will be
        (nx + 1) * (ny + 1) nodes.

    Notes
    -----
    - Only 2D structured grids are currently implemented. A 3D extension is
      planned after validating 2D MPM.
    - Node coordinates and boundary-condition arrays persist across steps.
      The nodal mass, momentum, velocity, acceleration and forces are reset
      at the beginning of each conventional explicit MPM step.
    - A True value in velocity_bc_mask means that the corresponding nodal
      velocity component is prescribed by velocity_bc_value. This class only
      stores the boundary conditions; the solver applies them.
    - Coordinates are fixed; material particles move through the grid.
    """

    def __init__(self, origin, cell_size, n_cells) -> None:
        origin_arr = np.asarray(origin, dtype=np.float64)
        size_arr = np.asarray(cell_size, dtype=np.float64)

        if origin_arr.shape != (2,) or not np.all(np.isfinite(origin_arr)):
            raise ValueError("origin must be a finite numeric array of shape (2,).")
        if size_arr.shape != (2,) or not np.all(np.isfinite(size_arr)):
            raise ValueError("cell_size must be a finite numeric array of shape (2,).")
        if np.any(size_arr <= 0.0):
            raise ValueError("cell_size components must be strictly positive.")

        counts = np.asarray(n_cells)
        if counts.shape != (2,) or not np.issubdtype(counts.dtype, np.integer):
            raise ValueError("n_cells must contain exactly two integers (nx, ny).")
        if np.any(counts <= 0):
            raise ValueError("n_cells components must be strictly positive.")

        self.origin = np.array(origin_arr, dtype=np.float64, order="C", copy=True)
        self.cell_size = np.array(size_arr, dtype=np.float64, order="C", copy=True)
        nx, ny = int(counts[0]), int(counts[1])
        self.n_cells = (nx, ny)
        self.node_shape = (nx + 1, ny + 1)
        self.n_nodes = (nx + 1) * (ny + 1)

        # Flat node-coordinate array in x-fast ordering; each row is [x, y].
        xs = self.origin[0] + self.cell_size[0] * np.arange(nx + 1)
        ys = self.origin[1] + self.cell_size[1] * np.arange(ny + 1)
        self.node_coordinates = np.column_stack((np.tile(xs, ny + 1),
                                                 np.repeat(ys, nx + 1)))

        self.mass = np.zeros(self.n_nodes, dtype=np.float64)
        self.momentum = np.zeros((self.n_nodes, 2), dtype=np.float64)
        self.velocity = np.zeros((self.n_nodes, 2), dtype=np.float64)
        self.acceleration = np.zeros((self.n_nodes, 2), dtype=np.float64)
        self.force_internal = np.zeros((self.n_nodes, 2), dtype=np.float64)
        self.force_external = np.zeros((self.n_nodes, 2), dtype=np.float64)

        # These are persistent boundary-condition *data*, not reset per step.
        self.velocity_bc_mask = np.zeros((self.n_nodes, 2), dtype=np.bool_)
        self.velocity_bc_value = np.zeros((self.n_nodes, 2), dtype=np.float64)

    @property
    def ndim(self) -> int:
        return 2

    @property
    def n_elements(self) -> int:
        """Number of rectangular grid cells."""
        return self.n_cells[0] * self.n_cells[1]

    def node_id(self, ix: int, iy: int) -> int:
        """Return flattened node index for integer (ix, iy), x-fast ordering."""
        if (not isinstance(ix, (int, np.integer)) or isinstance(ix, (bool, np.bool_))
                or not isinstance(iy, (int, np.integer)) or isinstance(iy, (bool, np.bool_))):
            raise TypeError("Node indices ix and iy must be integers.")
        nx, ny = self.n_cells
        if not (0 <= ix <= nx and 0 <= iy <= ny):
            raise IndexError(f"Node index ({ix}, {iy}) outside node_shape {self.node_shape}.")
        return int(iy * (nx + 1) + ix)

    def cell_node_ids(self, ix: int, iy: int) -> np.ndarray:
        """Four cell corner IDs, counterclockwise: BL, BR, TR, TL."""
        if (not isinstance(ix, (int, np.integer)) or isinstance(ix, (bool, np.bool_))
                or not isinstance(iy, (int, np.integer)) or isinstance(iy, (bool, np.bool_))):
            raise TypeError("Cell indices ix and iy must be integers.")
        nx, ny = self.n_cells
        if not (0 <= ix < nx and 0 <= iy < ny):
            raise IndexError(f"Cell index ({ix}, {iy}) outside n_cells {self.n_cells}.")
        bl = self.node_id(ix, iy)
        br = bl + 1
        tl = bl + (nx + 1)
        tr = tl + 1
        return np.array([bl, br, tr, tl], dtype=np.int64)

    def reset_nodal_state(self) -> None:
        """Clear transient nodal arrays for reassembly at the next time step.

        Geometry and prescribed-velocity BC masks/values are NOT cleared.
        This should be called before P2G; it does not perform P2G itself.
        """
        self.mass.fill(0.0)
        self.momentum.fill(0.0)
        self.velocity.fill(0.0)
        self.acceleration.fill(0.0)
        self.force_internal.fill(0.0)
        self.force_external.fill(0.0)

    def __repr__(self) -> str:
        return (f"BackgroundGrid(n_cells={self.n_cells}, "
                f"n_nodes={self.n_nodes}, ndim={self.ndim})")
