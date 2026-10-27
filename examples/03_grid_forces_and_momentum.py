"""Stage 3B: gravity and bottom velocity constraints on one Quad4 cell.

Run from repository root:
    python examples/03_grid_forces_and_momentum.py

This is ONE grid substep, not a complete MPM simulation: particles do not move.
"""

import numpy as np

from geopympm.core import BackgroundGrid, ParticleSet
from geopympm.interpolation import evaluate_quad4
from geopympm.solvers import particle_to_grid, update_grid_momentum


def main():
    grid = BackgroundGrid(
        origin=(0.0, 0.0), cell_size=(1.0, 1.0), n_cells=(1, 1)
    )
    particles = ParticleSet(
        position=[[0.5, 0.5]],
        mass=[4.0],
        volume=[0.002],
        velocity=[[0.0, 0.0]],
        # Initial Cauchy stress defaults to zero for this basic gravity test.
    )

    # Bottom nodes are 0 and 1 with x-fast global node numbering.
    grid.velocity_bc_mask[[0, 1], 1] = True
    grid.velocity_bc_value[[0, 1], 1] = 0.0

    interp = evaluate_quad4(particles, grid)
    particle_to_grid(particles, grid, interp)

    # Use a deliberately short, illustrative timestep; a complete solver must
    # later enforce a scheme-dependent stability timestep criterion.
    dt = 0.01
    update_grid_momentum(
        particles, grid, interp, dt,
        gravity=(0.0, -9.81),
    )

    print('node IDs (BL,BR,TR,TL):', interp.node_ids[0].tolist())
    print('nodal masses:', np.round(grid.mass, 5))
    print('nodal external forces (N):\n', np.round(grid.force_external, 5))
    print('nodal velocities (m/s):\n', np.round(grid.velocity, 5))
    print('nodal accelerations (m/s^2):\n', np.round(grid.acceleration, 5))
    print('particle position unchanged:', particles.position.tolist())

    # Expected: bottom vertical velocities 0, top vertical velocities -0.0981.
    np.testing.assert_allclose(grid.velocity[[0, 1], 1], 0.0)
    np.testing.assert_allclose(grid.velocity[[2, 3], 1], -0.0981)


if __name__ == '__main__':
    main()
