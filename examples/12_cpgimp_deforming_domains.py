"""Stage 10B: cpGIMP domain lengths evolve under a grid-imposed affine field."""
import numpy as np
from geopympm.core import ParticleSet, BackgroundGrid
from geopympm.materials import LinearElastic
from geopympm.solvers import ExplicitMPMSolver


def main():
    p=ParticleSet(position=[[2.25,2.75]],mass=[1.],volume=[.2],
                  domain_size=[[.4,.5]])
    g=BackgroundGrid((0.,0.),(1.,1.),(6,6))
    # Prescribed grid velocities v=(a*x,b*y), to isolate kinematic update.
    # This is a kinematic demonstration, not a force-driven benchmark.
    rate=np.array([.2,-.15])
    g.velocity_bc_mask[:]=True
    g.velocity_bc_value[:]=g.node_coordinates*rate
    solver=ExplicitMPMSolver(p,g,LinearElastic(10.,.25),dt=.01,
           interpolation='cpgimp',kinematics_update='exponential',
           advection_update='affine_exponential')
    print('Initial: domain=',p.domain_size[0], 'F=',p.deformation_gradient[0])
    for _ in range(10):
        result=solver.step()
    print('Final t=',result.time)
    print('Final: domain=',p.domain_size[0], 'F=',p.deformation_gradient[0])
    print('Reference domain=',p.domain_size0[0]*np.exp(rate*result.time))

if __name__=='__main__':
    main()
