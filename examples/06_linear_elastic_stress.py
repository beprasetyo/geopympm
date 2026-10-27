"""Stage 4 demonstration: from grid velocity to plane-strain elastic stress.

Run:
    python examples/06_linear_elastic_stress.py

This is an illustrative single time step, not a validated full MPM solver.
"""

import numpy as np

from geopympm.core import ParticleSet, BackgroundGrid
from geopympm.interpolation import evaluate_quad4
from geopympm.solvers import (
    particle_to_grid, update_grid_momentum, update_particle_kinematics,
    grid_to_particle,
)
from geopympm.materials import LinearElastic, update_linear_elastic_stress


def main():
    p = ParticleSet(position=[[.5, .5]], mass=[450.0], volume=[0.25])
    g = BackgroundGrid(origin=(0, 0), cell_size=(1, 1), n_cells=(1, 1))
    g.velocity_bc_mask[[0, 1], 1] = True

    elastic = LinearElastic(young_modulus=20e6, poisson_ratio=0.25)
    dt = 0.01

    interpolation = evaluate_quad4(p, g)
    particle_to_grid(p, g, interpolation)
    update_grid_momentum(p, g, interpolation, dt, gravity=(0, -9.81))

    # At x_n, using grid velocities from the momentum step:
    result = update_particle_kinematics(p, g, interpolation, dt)
    update_linear_elastic_stress(p, result.strain_increment, elastic)

    # Advection comes AFTER the kinematic and stress update on old supports.
    grid_to_particle(p, g, interpolation, dt, pic_fraction=1.0)

    np.set_printoptions(suppress=True, precision=8)
    print(f"Linear Elastic: E={elastic.young_modulus:g} Pa, nu={elastic.poisson_ratio:g}")
    print("Incremental engineering strain [xx yy zz xy yz xz]:")
    print(result.strain_increment[0])
    print("Updated stress [xx yy zz xy yz xz] [Pa]:")
    print(p.stress[0])
    print("Updated deformation gradient:")
    print(p.deformation_gradient[0])
    print("Updated particle position:", p.position[0])
    print("NOTE: one step only; stress update is nonobjective at large rotations.")


if __name__ == '__main__':
    main()
