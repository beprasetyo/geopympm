"""Stage 9 affine-exponential particle advection (direct G2P and full solver)."""
import math
import numpy as np
import pytest
from geopympm.core import BackgroundGrid, ParticleSet
from geopympm.interpolation import evaluate_quad4
from geopympm.materials import LinearElastic
from geopympm.solvers import (
    particle_to_grid, update_particle_kinematics, grid_to_particle, ExplicitMPMSolver,
)
from geopympm.kernels.advection_exponential import _phi1_2x2
from validation.stage9.trajectory import rigid_rotation_comparison, affine_flow
from validation.stage9.run_advection import main as run_benchmark


def setup_field(A, b=(0.,0.), *, x=(.4,.5), velocity=(0.,0.)):
    p=ParticleSet(position=[list(x)],mass=[10],volume=[.2],velocity=[list(velocity)])
    g=BackgroundGrid((0.,0.),(1.,1.),(1,1))
    ip=evaluate_quad4(p,g)
    particle_to_grid(p,g,ip)
    g.velocity[:]=g.node_coordinates@np.asarray(A,dtype=float).T+np.asarray(b)
    g.acceleration[:]=0
    L=np.einsum('pai,paj->pij',g.velocity[ip.node_ids],ip.gradients)
    return p,g,ip,L


@pytest.mark.parametrize('b',[(0.,0.),(1.,0.),(0.,-1.),(.17,-.2)])
def test_phi1_zero_gradient_is_identity(b):
    result=np.array(_phi1_2x2(0.,0.,0.,0.)).reshape(2,2)
    np.testing.assert_allclose(result,np.eye(2),atol=1e-15)


@pytest.mark.parametrize('w,dt',[(3.,.03),(-2.,.17),(8.,.2)])
def test_phi1_pure_spin_closed_form(w,dt):
    th=w*dt
    f=np.array(_phi1_2x2(0.,-th,th,0.)).reshape(2,2)
    expected=np.array([[math.sin(th)/th,-(1-math.cos(th))/th],
                       [(1-math.cos(th))/th,math.sin(th)/th]])
    np.testing.assert_allclose(f,expected,atol=3e-14,rtol=2e-14)


@pytest.mark.parametrize('rate,dt',[(0.4,0.1),(-.9,.2),(1.,.6)])
def test_phi1_uniform_dilation(rate,dt):
    z=rate*dt
    f=np.array(_phi1_2x2(z,0.,0.,z)).reshape(2,2)
    np.testing.assert_allclose(f,np.eye(2)*math.expm1(z)/z,rtol=1e-13,atol=1e-14)


def test_translation_affine_equals_euler():
    p,g,ip,L=setup_field([[0,0],[0,0]],(.2,-.3))
    old=p.position.copy()
    grid_to_particle(p,g,ip,.2,pic_fraction=0.,
                     advection_update='affine_exponential',velocity_gradient=L)
    np.testing.assert_allclose(p.position,old+.2*np.array([.2,-.3]),atol=1e-14)


def test_simple_shear_with_vertical_translation_quadratic_x():
    # v_x = k*y, v_y=b. x(t)=x0+k*y0*t+0.5*k*b*t**2.
    k=.8; b=.25; dt=.3; x0=np.array([.4,.5]);
    p,g,ip,L=setup_field([[0,k],[0,0]],(0.,b),x=x0)
    grid_to_particle(p,g,ip,dt,pic_fraction=1.,
                     advection_update='affine_exponential',velocity_gradient=L)
    ref=x0+np.array([k*x0[1]*dt+0.5*k*b*dt*dt,b*dt])
    np.testing.assert_allclose(p.position[0],ref,atol=1e-14)


def test_singular_gradient_no_matrix_inverse():
    f=np.array(_phi1_2x2(0.,.6,0.,0.)).reshape(2,2)
    np.testing.assert_allclose(f,[[1.,.3],[0.,1.]],atol=1e-14)


def test_legacy_euler_default_matches_explicit_option():
    p,g,ip,L=setup_field([[0,-.3],[.3,0]],x=(.4,.5))
    x0=p.position.copy()
    grid_to_particle(p,g,ip,.2,pic_fraction=1.)
    default_x=p.position.copy(); p.position[:]=x0
    grid_to_particle(p,g,ip,.2,pic_fraction=1.,advection_update='euler')
    np.testing.assert_array_equal(p.position,default_x)


@pytest.mark.parametrize('bad',[None,True,0,'','strange',[],np.array([2])])
def test_invalid_advection_mode_rejected_and_state_preserved(bad):
    p,g,ip,L=setup_field([[0,0],[0,0]],(.1,0.))
    before=p.position.copy(),p.velocity.copy()
    with pytest.raises(ValueError,match='advection_update'):
        grid_to_particle(p,g,ip,.01,advection_update=bad,velocity_gradient=L)
    np.testing.assert_array_equal(p.position,before[0]);np.testing.assert_array_equal(p.velocity,before[1])


@pytest.mark.parametrize('invalid',[None,np.zeros((2,2)),np.zeros((1,2,2),dtype=np.float32),
                                      np.full((1,2,2),np.nan),np.full((1,2,2),np.inf)])
def test_invalid_velocity_gradient_no_commit(invalid):
    p,g,ip,L=setup_field([[0,0],[0,0]],(.1,0.))
    before=p.position.copy(),p.velocity.copy()
    with pytest.raises(ValueError,match='velocity_gradient'):
        grid_to_particle(p,g,ip,.01,advection_update='affine_exponential',velocity_gradient=invalid)
    np.testing.assert_array_equal(p.position,before[0]);np.testing.assert_array_equal(p.velocity,before[1])


def test_stale_velocity_gradient_rejected():
    p,g,ip,L=setup_field([[0,-.3],[.3,0]])
    before=p.position.copy(),p.velocity.copy()
    with pytest.raises(ValueError,match='velocity_gradient'):
        grid_to_particle(p,g,ip,.01,advection_update='affine_exponential',velocity_gradient=L+1.0)
    np.testing.assert_array_equal(p.position,before[0]);np.testing.assert_array_equal(p.velocity,before[1])


def test_exiting_grid_rolls_back_g2p_position_and_velocity():
    p,g,ip,L=setup_field([[0,0],[0,0]],(2.5,0.),x=(.95,.5))
    before=p.position.copy(),p.velocity.copy()
    with pytest.raises(ValueError,match='outside background grid'):
        grid_to_particle(p,g,ip,.04,advection_update='affine_exponential',velocity_gradient=L)
    np.testing.assert_array_equal(p.position,before[0]);np.testing.assert_array_equal(p.velocity,before[1])


def test_solver_default_and_explicit_new_option():
    p,g,ip,L=setup_field([[0,0],[0,0]])
    m=LinearElastic(1000,.25)
    a=ExplicitMPMSolver(p,g,m,dt=.001)
    b=ExplicitMPMSolver(p,g,m,dt=.001,advection_update='affine_exponential')
    assert a.advection_update=='euler'
    assert b.advection_update=='affine_exponential'
    for bad in ('bad',None,False,1):
        with pytest.raises(ValueError,match='advection_update'):
            ExplicitMPMSolver(p,g,m,dt=.001,advection_update=bad)


def test_solver_transactional_rollback_at_grid_boundary():
    p=ParticleSet(position=[[.99,.5]],mass=[1.],volume=[.25],velocity=[[3.,0.]])
    g=BackgroundGrid((0,0),(1,1),(1,1))
    solver=ExplicitMPMSolver(p,g,LinearElastic(1000,.25),dt=.01,cfl=1.,
                             stress_update='corotational',kinematics_update='exponential',
                             advection_update='affine_exponential')
    snap={k:getattr(p,k).copy() for k in solver._PARTICLE_MUTABLE}
    with pytest.raises(ValueError,match='outside background grid'):
        solver.step()
    for k,v in snap.items():np.testing.assert_array_equal(getattr(p,k),v)
    assert solver.time==0. and solver.step_count==0


def test_benchmark_full_solver_rigid_spin_improvement():
    r=rigid_rotation_comparison()
    assert r.passed
    assert r.metrics['euler_position_error_end_m']>1e-5
    assert r.metrics['affine_exponential_position_error_end_m']<1e-12
    assert r.metrics['exp_j_error_max']<1e-11


@pytest.mark.parametrize('fraction',[0.,.05,1.])
def test_benchmark_affine_flow_fully_integrated(fraction):
    r=affine_flow(pic_fraction=fraction)
    assert r.passed
    assert r.metrics['position_error_m']<1e-12


def test_cli_writes_report(tmp_path):
    assert run_benchmark(['--output',str(tmp_path)])==0
    assert (tmp_path/'9_summary.json').exists()
    assert (tmp_path/'9_rotation_history.csv').exists()
    assert (tmp_path/'9_affine_PIC_history.csv').exists()
    assert (tmp_path/'9_affine_FLIP_history.csv').exists()
