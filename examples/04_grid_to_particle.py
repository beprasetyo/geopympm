"""Stage 3C examples: one explicit kinematic step and PIC/FLIP mixing.

Run from repository root:
    python examples/04_grid_to_particle.py

This is not yet a full material MPM solver: deformation gradient, material
volume, strain and stress are deliberately not evolved here.
"""

import numpy as np

from geopympm.core import BackgroundGrid, ParticleSet
from geopympm.interpolation import evaluate_quad4
from geopympm.solvers import particle_to_grid, update_grid_momentum, grid_to_particle


def example_gravity_with_fixed_bottom():
    print('=== Example A: gravity + fixed bottom (PIC) ===')
    p = ParticleSet(position=[[0.5, 0.5]], mass=[4.0], volume=[0.002])
    g = BackgroundGrid((0., 0.), (1., 1.), (1, 1))
    g.velocity_bc_mask[[0, 1], 1] = True
    dt = 0.01

    ip = evaluate_quad4(p, g)
    particle_to_grid(p, g, ip)
    update_grid_momentum(p, g, ip, dt, gravity=(0., -9.81))
    grid_to_particle(p, g, ip, dt, pic_fraction=1.0)

    print('particle velocity:', p.velocity[0].tolist())
    print('particle position:', p.position[0].tolist())
    print('initial position :', p.position0[0].tolist())
    np.testing.assert_allclose(p.velocity[0], [0., -.04905])
    np.testing.assert_allclose(p.position[0], [.5, .4995095])


def example_pic_vs_flip():
    print('\n=== Example B: opposite velocities, no forces ===')
    for alpha in (1.0, 0.5, 0.0):
        p = ParticleSet(
            position=[[.5, .5], [.5, .5]],
            velocity=[[2., 0.], [-2., 0.]],
            mass=[1., 1.],
            volume=[.25, .25],
        )
        g = BackgroundGrid((0., 0.), (1., 1.), (1, 1))
        ip = evaluate_quad4(p, g)
        particle_to_grid(p, g, ip)
        update_grid_momentum(p, g, ip, .01)
        grid_to_particle(p, g, ip, .01, pic_fraction=alpha)
        print(f'pic_fraction={alpha:.1f}: velocity_x =',
              np.round(p.velocity[:, 0], 6).tolist())


if __name__ == '__main__':
    example_gravity_with_fixed_bottom()
    example_pic_vs_flip()
