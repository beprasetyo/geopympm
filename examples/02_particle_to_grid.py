"""Stage 3A: P2G mass/momentum transfer without force integration.

    python examples/02_particle_to_grid.py
"""

import numpy as np

from geopympm.core import BackgroundGrid, ParticleSet
from geopympm.interpolation import evaluate_quad4
from geopympm.solvers import particle_to_grid


def main():
    grid = BackgroundGrid(origin=(0.0, 0.0), cell_size=(1.0, 1.0), n_cells=(2, 1))
    particles = ParticleSet(
        position=[[0.5, 0.5], [1.5, 0.5]],
        mass=[2.0, 4.0],
        volume=[0.001, 0.002],
        velocity=[[3.0, 1.0], [0.0, -2.0]],
    )
    interpolation = evaluate_quad4(particles, grid)
    particle_to_grid(particles, grid, interpolation)

    print("Particle mass:", particles.mass)
    print("Particle velocity:\n", particles.velocity)
    print("\nProjected nodal quantities:")
    for i in range(grid.n_nodes):
        print(f"node {i}: mass={grid.mass[i]:.3f}, "
              f"momentum={grid.momentum[i]}, velocity={grid.velocity[i]}")

    print("\nConservation checks:")
    print("total particle mass:", particles.mass.sum())
    print("total nodal mass:   ", grid.mass.sum())
    print("total particle momentum:", np.sum(particles.mass[:, None] * particles.velocity, axis=0))
    print("total nodal momentum:  ", grid.momentum.sum(axis=0))
    print("Note: forces, BC enforcement, grid time update, and G2P are not implemented here.")


if __name__ == "__main__":
    main()
