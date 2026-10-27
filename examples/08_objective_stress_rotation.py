"""Run with: python examples/08_objective_stress_rotation.py."""
import numpy as np

from geopympm.core import ParticleSet
from geopympm.materials import LinearElastic, update_objective_linear_elastic_stress

p = ParticleSet(position=[[0.5, 0.5]], mass=[1.0], volume=[1.0],
                stress=[[100.0, 0.0, 20.0, 0.0, 0.0, 0.0]])
# Pure 90 degree rotation over one time step; D=0, W != 0.
dt = 0.1
omega = (np.pi / 2) / dt
L = np.array([[[0.0, -omega], [omega, 0.0]]])
inc = np.zeros((1, 6))
update_objective_linear_elastic_stress(p, inc, L, LinearElastic(20e6, 0.25), dt)
print('After +90 degree rigid rotation (Pa):')
print('sigma_xx, sigma_yy, sigma_zz, sigma_xy =', p.stress[0, [0,1,2,3]])
print('Expected: [0, 100, 20, 0] (up to roundoff)')
