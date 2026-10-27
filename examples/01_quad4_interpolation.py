"""Stage 2 example: compute standard MPM Quad4 interpolation for three particles.

Run from the installed GeoPyMPM environment:
    python examples/01_quad4_interpolation.py
"""

from geopympm.core import BackgroundGrid, ParticleSet
from geopympm.interpolation import evaluate_quad4


def main():
    grid = BackgroundGrid(origin=(0.0, 0.0), cell_size=(0.5, 0.5), n_cells=(2, 2))
    particles = ParticleSet(
        position=[[0.25, 0.25], [0.75, 0.25], [0.5, 1.0]],
        mass=[1.0, 2.0, 3.0],
        volume=[0.001, 0.001, 0.001],
    )
    result = evaluate_quad4(particles, grid)
    print(grid)
    print(particles)
    for p in range(particles.n_particles):
        print(f"\nparticle {p}, position={particles.position[p]}")
        print("  cell (ix, iy):", result.cell_indices[p])
        print("  node IDs:", result.node_ids[p])
        print("  weights:", result.weights[p])
        print("  gradients (dN/dx, dN/dy):\n", result.gradients[p])
        print("  sum(weights):", result.weights[p].sum())


if __name__ == "__main__":
    main()
