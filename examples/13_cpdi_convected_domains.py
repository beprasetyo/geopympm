"""Stage 10C: CPDI1 under imposed shear; not a force-driven validation."""
import numpy as np
from geopympm.core import ParticleSet, BackgroundGrid
from geopympm.materials import LinearElastic
from geopympm.solvers import ExplicitMPMSolver
from geopympm.interpolation import evaluate_cpdi, candidate_cpdi_geometry


def main():
    p=ParticleSet([[4.,4.]],[1.],[.2],domain_size=[[.4,.5]])
    g=BackgroundGrid((0.,0.),(1.,1.),(9,9))
    rate=.3
    g.velocity_bc_mask[:]=True
    g.velocity_bc_value[:,0]=rate*(g.node_coordinates[:,1]-4.)
    s=ExplicitMPMSolver(p,g,LinearElastic(1.,.25),dt=.01,
         interpolation='cpdi',kinematics_update='exponential',
         stress_update='corotational',advection_update='affine_exponential')
    print('Initial F:',p.deformation_gradient[0])
    s.run(20)
    ip=evaluate_cpdi(p,g)
    corners,*_=candidate_cpdi_geometry(p,g)
    print('Final time (s):',s.time)
    print('Final F:\n',p.deformation_gradient[0])
    print('Reference F:\n',np.array([[1.,rate*s.time],[0.,1.]]))
    print('Domain corners:\n',corners[0])
    print('Bounding domain_size:',p.domain_size[0])
    print('Basis sum:',ip.weights.sum(), 'gradient sum:',ip.gradients.sum(axis=1))

if __name__=='__main__':main()
