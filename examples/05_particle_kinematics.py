"""Stage 3D: kinematics from the Stage-3B grid update, BEFORE Stage-3C G2P.

Run from root after editable install:
    python examples/05_particle_kinematics.py

One particle at a 2D Quad4 cell center under gravity, with bottom vertical
velocity constrained. This is a one-step kinematic demonstration, NOT a
physically validated landslide simulation or constitutive model.
"""
from __future__ import annotations

import numpy as np

from geopympm.core import BackgroundGrid, ParticleSet
from geopympm.interpolation import evaluate_quad4
from geopympm.solvers import (
    particle_to_grid, update_grid_momentum, update_particle_kinematics,
    grid_to_particle,
)


def main() -> None:
    particles = ParticleSet(
        position=[[0.5, 0.5]],
        mass=[450.0],
        volume=[0.25],
    )
    grid = BackgroundGrid(
        origin=(0.0, 0.0), cell_size=(1.0, 1.0), n_cells=(1, 1)
    )
    grid.velocity_bc_mask[[0, 1], 1] = True
    grid.velocity_bc_value[[0, 1], 1] = 0.0

    dt = 0.01
    interp = evaluate_quad4(particles, grid)
    particle_to_grid(particles, grid, interp)
    update_grid_momentum(particles, grid, interp, dt, gravity=(0.0, -9.81))

    # Stage 3D must precede G2P, which changes particles.position.
    result = update_particle_kinematics(particles, grid, interp, dt)
    grid_to_particle(particles, grid, interp, dt, pic_fraction=1.0)

    np.set_printoptions(precision=8, suppress=True)
    print("Velocity gradient [1/s]:")
    print(result.velocity_gradient[0])
    print("Strain increment (engineering Voigt):")
    print(result.strain_increment[0])
    print("Deformation gradient F:")
    print(particles.deformation_gradient[0])
    print("J = det(F):", result.jacobian[0])
    print("Updated volume [m^3]:", particles.volume[0])
    print("Updated density [kg/m^3]:", particles.density[0])
    print("Updated particle position [m]:", particles.position[0])
    print("Updated particle velocity [m/s]:", particles.velocity[0])
    print("Particle stress (unchanged):", particles.stress[0])


if __name__ == "__main__":
    main()
