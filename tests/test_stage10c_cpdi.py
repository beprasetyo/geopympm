"""Stage 10C: CPDI1 affine geometry, objective spin, solver and rollback tests."""
import math
import numpy as np
import pytest

from geopympm.core import ParticleSet, BackgroundGrid
from geopympm.interpolation import evaluate_cpdi, evaluate_gimp, update_cpdi_domain
from geopympm.interpolation.cpdi import candidate_cpdi_geometry
from geopympm.interpolation.gimp import validate_current_gimp
from geopympm.materials import LinearElastic
from geopympm.solvers import ExplicitMPMSolver


def case(position=(4.,4.), size=(.6,.4)):
    g=BackgroundGrid((0.,0.),(1.,1.),(9,9))
    p=ParticleSet([position],[1.],[.25],domain_size=[size])
    return p,g


@pytest.mark.parametrize('F', [
    [[1., 0.], [0., 1.]],
    [[1.5, 0.], [0., .7]],
    [[1., 1.8], [0., 1.]],
    [[.6, -.8], [.8, .6]],
    [[0., -1.], [1., 0.]],
    [[-1., 0.], [0., -1.]],
    [[2.5, -.5], [.4, 1.1]],
])
def test_cpdi_basis_reproduces_affine_field_even_under_rotation_shear(F):
    p,g=case();p.deformation_gradient[0]=F
    update_cpdi_domain(p,g)
    ip=evaluate_cpdi(p,g)
    coordinates=g.node_coordinates[ip.node_ids]
    np.testing.assert_allclose(ip.weights.sum(axis=1),1.,rtol=0,atol=2e-13)
    np.testing.assert_allclose(ip.gradients.sum(axis=1),0.,rtol=0,atol=2e-13)
    np.testing.assert_allclose(np.einsum('pa,pad->pd',ip.weights,coordinates),
                               p.position,atol=2e-13)
    np.testing.assert_allclose(np.einsum('pai,paj->pij', coordinates,ip.gradients),
                               np.eye(2)[None],atol=4e-13)
    corners,a,b,bbox=candidate_cpdi_geometry(p,g)
    np.testing.assert_allclose(corners.mean(axis=1), p.position, atol=1e-14)
    np.testing.assert_allclose(bbox,2*(np.abs(a)+np.abs(b)),atol=1e-14)
    assert ip.node_ids.shape==(1,16)


def test_reference_square_domain_matches_ugimp_within_one_grid_cell():
    p,g=case(position=(4.5,4.5),size=(.3,.35))
    im=evaluate_gimp(p,g); ic=evaluate_cpdi(p,g)
    ids=np.arange(g.n_nodes)
    for values in ('weights','gradients'):
        left=np.zeros((g.n_nodes,2)) if values=='gradients' else np.zeros(g.n_nodes)
        right=left.copy()
        for i in range(im.node_ids.shape[1]): left[im.node_ids[0,i]]+=getattr(im,values)[0,i]
        for i in range(ic.node_ids.shape[1]):right[ic.node_ids[0,i]]+=getattr(ic,values)[0,i]
        np.testing.assert_allclose(left,right,atol=4e-14)


def test_cpdi_uses_full_F_and_outgrows_single_cell():
    p,g=case()
    p.deformation_gradient[0]=np.array([[4.0,.4],[0.,1.]])
    update_cpdi_domain(p,g)
    assert p.domain_size[0,0]>g.cell_size[0]
    ip=evaluate_cpdi(p,g)
    assert ip.weights.sum()==pytest.approx(1.)
    assert p.domain_size0[0,0]==pytest.approx(.6)


def test_cpdi_domain_shear_and_rotation_bounding_width():
    p,g=case(size=(.4,.8))
    p.deformation_gradient[0]=[[1.,1.],[0.,1.]]
    update_cpdi_domain(p,g)
    np.testing.assert_allclose(p.domain_size,[[1.2,.8]])
    p.deformation_gradient[0]=[[0.,-1.],[1.,0.]]
    update_cpdi_domain(p,g)
    np.testing.assert_allclose(p.domain_size,[[.8,.4]])


@pytest.mark.parametrize('F', [
    [[1.,0.],[0.,0.]], [[-1.,0.],[0.,1.]],
    [[1.,1e12],[0.,1e-14]], [[float('nan'),0.],[0.,1.]],
])
def test_cpdi_rejects_invalid_F_without_mutating_domain(F):
    p,g=case();p.deformation_gradient[0]=F
    original=p.domain_size.copy()
    with pytest.raises(ValueError):update_cpdi_domain(p,g)
    np.testing.assert_array_equal(p.domain_size,original)


def test_cpdi_uses_valid_corner_support_even_across_many_cells():
    p,g=case(position=(4.1,4.1),size=(.45,.5))
    p.deformation_gradient[0]=[[4.,0.],[0.,3.]]
    update_cpdi_domain(p,g)
    ip=evaluate_cpdi(p,g)
    assert len(np.unique(ip.node_ids[0]))>4
    assert ip.weights.min()>=0
    assert ip.node_ids.max()<g.n_nodes


def test_cpdi_stale_F_detected_before_kinematics():
    p,g=case();ip=evaluate_cpdi(p,g)
    p.deformation_gradient[0,0,1]=.2
    with pytest.raises(ValueError,match='stale'):
        validate_current_gimp(p,g,ip)
    # One specific within-step exception can verify against the captured F.
    validate_current_gimp(p,g,ip,allow_updated_f=True)


def test_solver_rigid_rotation_exponential_updates_corners_not_shrinking():
    p,g=case(size=(.4,.65))
    center=np.array([4.,4.])
    omega=2.
    g.velocity_bc_mask[:]=True
    rel=g.node_coordinates-center
    g.velocity_bc_value[:]=np.column_stack((-omega*rel[:,1],omega*rel[:,0]))
    s=ExplicitMPMSolver(p,g,LinearElastic(1.,.25),dt=.02,cfl=.9,
                        interpolation='cpdi',pic_fraction=0.,
                        stress_update='corotational',
                        kinematics_update='exponential',
                        advection_update='affine_exponential')
    for _ in range(25):s.step()
    theta=omega*.02*25
    ref=np.array([[math.cos(theta),-math.sin(theta)],
                  [math.sin(theta),math.cos(theta)]])
    np.testing.assert_allclose(p.deformation_gradient[0],ref,atol=3e-13)
    np.testing.assert_allclose(p.position,[center],atol=2e-13)
    np.testing.assert_allclose(p.domain_size[0],
          [abs(ref[0,0])*.4+abs(ref[0,1])*.65,
           abs(ref[1,0])*.4+abs(ref[1,1])*.65],atol=2e-13)
    assert np.linalg.det(p.deformation_gradient[0])==pytest.approx(1.,abs=2e-13)
    assert np.max(np.abs(p.stress))<1e-12


def test_solver_pure_shear_full_F():
    p,g=case(size=(.4,.5))
    rate=.3
    g.velocity_bc_mask[:]=True
    g.velocity_bc_value[:,0] = rate*(g.node_coordinates[:,1]-4.)
    g.velocity_bc_value[:,1] = 0.
    s=ExplicitMPMSolver(p,g,LinearElastic(1.,.25),dt=.01,
                        interpolation='cpdi',kinematics_update='exponential',
                        advection_update='affine_exponential')
    for _ in range(20):s.step()
    np.testing.assert_allclose(p.deformation_gradient[0],[[1.,.06],[0.,1.]],atol=2e-13)
    np.testing.assert_allclose(p.domain_size[0],[.43,.5],atol=2e-13)
    assert np.linalg.det(p.deformation_gradient[0])==pytest.approx(1.,abs=2e-13)


def test_solver_domain_leaving_grid_rollback():
    g=BackgroundGrid((0.,0.),(1.,1.),(5,5))
    p=ParticleSet([[.65,2.]], [1.], [.2],velocity=[[-.7,0.]],domain_size=[[1.,.7]])
    s=ExplicitMPMSolver(p,g,LinearElastic(.1,.25),dt=.25,cfl=1.,interpolation='cpdi')
    particle_before={k:getattr(p,k).copy() for k in s._PARTICLE_MUTABLE}
    grid_before={k:getattr(g,k).copy() for k in s._GRID_MUTABLE}
    with pytest.raises(ValueError,match='CPDI particle domain corners'):
        s.step()
    for k,v in particle_before.items():np.testing.assert_array_equal(getattr(p,k),v)
    for k,v in grid_before.items():np.testing.assert_array_equal(getattr(g,k),v)
    assert s.time==0. and s.step_count==0


def test_cpdi_is_opt_in():
    p,g=case();m=LinearElastic(10.,.25)
    assert ExplicitMPMSolver(p,g,m,dt=.001).interpolation=='mpm'
    assert ExplicitMPMSolver(p,g,m,dt=.001,interpolation='cpdi').interpolation=='cpdi'
