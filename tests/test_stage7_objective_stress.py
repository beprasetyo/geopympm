"""Stage-7 corotational linear-elastic stress integration tests."""
import json
from math import pi

import numpy as np
import pytest

from geopympm.core import ParticleSet
from geopympm.materials import (
    LinearElastic, update_linear_elastic_stress,
    update_objective_linear_elastic_stress,
)
from geopympm.kernels.objective_elastic import objective_elastic_plane_strain_kernel
from geopympm.solvers import ExplicitMPMSolver
from geopympm.core import BackgroundGrid
from validation.stage7.objective_rotation import rotate_stress_reference, rigid_spin
from validation.stage7.run_rotation import main as run_stage7


def particles(stress=None, ids=None):
    n = 1 if ids is None else len(ids)
    return ParticleSet(position=np.array([[0.5, 0.5]]*n),
                       mass=np.ones(n), volume=np.ones(n),
                       stress=stress,
                       material_id=ids)


def linear_kinematics(L, dt):
    matrix = np.asarray(L, dtype=float)
    if matrix.ndim == 2:
        matrix = matrix[None, :, :]
    n = matrix.shape[0]
    de = np.zeros((n, 6))
    de[:, 0] = dt * matrix[:, 0, 0]
    de[:, 1] = dt * matrix[:, 1, 1]
    de[:, 3] = dt * (matrix[:, 0, 1] + matrix[:, 1, 0])
    return de


@pytest.mark.parametrize('angle', [0.0, pi/6, -pi/4, pi/2, 2*pi])
def test_exact_pure_spin_rotation(angle):
    old = np.array([100.0, -20.0, 15.0, 9.0, 0.0, 0.0])
    p = particles(stress=[old]); dt = 0.1
    w = angle / dt
    L = np.array([[[0.0, -w], [w, 0.0]]])
    update_objective_linear_elastic_stress(p, np.zeros((1,6)), L,
                                           LinearElastic(1000.0, 0.25), dt)
    np.testing.assert_allclose(p.stress[0], rotate_stress_reference(old, angle), rtol=0, atol=1e-12)
    assert p.stress[0, 0] + p.stress[0, 1] == pytest.approx(old[0]+old[1], abs=1e-12)
    assert p.stress[0, 2] == old[2]


def test_nondegenerate_principal_stress_invariant_under_spin():
    original = np.array([130., -20., -7., 40., 0., 0.])
    p = particles(stress=[original])
    t0 = np.array([[original[0], original[3]], [original[3], original[1]]])
    eig0 = np.linalg.eigvalsh(t0)
    L = np.array([[[0.0, -3.1],[3.1, 0.0]]])
    dt = 0.002
    for _ in range(1000):
        update_objective_linear_elastic_stress(p, np.zeros((1,6)), L,
                                               LinearElastic(10e3, 0.2), dt)
    t = np.array([[p.stress[0,0], p.stress[0,3]], [p.stress[0,3], p.stress[0,1]]])
    np.testing.assert_allclose(np.linalg.eigvalsh(t), eig0, rtol=0, atol=2e-10)
    np.testing.assert_allclose(p.stress[0], rotate_stress_reference(original,3.1*2.0),rtol=0,atol=3e-10)


@pytest.mark.parametrize('L', [np.array([[.12,.08],[.08,-.24]]),
                                np.diag([.12,-.24]), np.zeros((2,2))])
def test_zero_spin_matches_stage4(L):
    dt=.005
    old=np.array([12.,-5.,11.,7.,0.,0.])
    pa=particles(stress=[old]); pb=particles(stress=[old])
    increment=linear_kinematics(L,dt)
    mat=LinearElastic(2.4e6,.32)
    update_linear_elastic_stress(pa,increment,mat)
    update_objective_linear_elastic_stress(pb,increment,L[None],mat,dt)
    np.testing.assert_allclose(pa.stress,pb.stress,rtol=0,atol=5e-10)


def test_plane_strain_zz_hydrostatic_increment():
    p=particles(stress=[[10.,10.,10.,0.,0.,0.]])
    L=np.array([[[.2,-5.],[5.,-.1]]])
    delta=linear_kinematics(L,.01)
    mat=LinearElastic(3000.,.25)
    update_objective_linear_elastic_stress(p,delta,L,mat,.01)
    np.testing.assert_allclose(p.stress[0,2],10.+mat.lame_lambda*(.2-.1)*.01)


def test_two_materials_dispatch():
    p=particles(ids=[0,2])
    L=np.broadcast_to(np.diag([.2,-.1]),(2,2,2)).copy()
    d=linear_kinematics(L,.01)
    mats={0:LinearElastic(1000.,0.2),2:LinearElastic(2000.,0.2)}
    update_objective_linear_elastic_stress(p,d,L,mats,.01)
    np.testing.assert_allclose(p.stress[1],2*p.stress[0],rtol=1e-12,atol=1e-12)


def test_reference_mixed_rate_second_order_convergence():
    """Independent midpoint quadrature of exact Jaumann stress integral."""
    L=np.array([[.25,-.34],[.46,-.10]])
    w=.5*(L[1,0]-L[0,1])
    m=LinearElastic(4000.,.21)
    D=.5*(L+L.T)
    S=m.lame_lambda*np.trace(D)*np.eye(2)+2*m.shear_modulus*D
    old=np.array([[10.,4.],[4.,-7.]])
    t=.4
    oldvoigt=np.array([old[0,0],old[1,1],7.,old[0,1],0.,0.])
    n_quad=20000
    angles = w*((np.arange(n_quad)+.5)*t/n_quad)
    c,s=np.cos(angles),np.sin(angles)
    # Analytic solution has R(t-s)S R(t-s).T, quadrature over a dummy elapsed time s.
    integral_xx=np.sum(c*c*S[0,0] + s*s*S[1,1] - 2*c*s*S[0,1])*t/n_quad
    integral_yy=np.sum(s*s*S[0,0] + c*c*S[1,1] + 2*c*s*S[0,1])*t/n_quad
    integral_xy=np.sum(c*s*(S[0,0]-S[1,1]) +(c*c-s*s)*S[0,1])*t/n_quad
    reference=rotate_stress_reference(oldvoigt,w*t)
    reference[:2]+=[integral_xx,integral_yy]
    reference[3]+=integral_xy
    reference[2]+=m.lame_lambda*np.trace(D)*t
    errors=[]
    for steps in [4,8,16]:
        dt=t/steps
        p=particles(stress=[oldvoigt])
        for _ in range(steps):
            update_objective_linear_elastic_stress(p,linear_kinematics(L,dt), L[None],m,dt)
        errors.append(np.max(np.abs(p.stress[0]-reference)))
    assert errors[0]>errors[1]>errors[2]
    assert errors[0]/errors[1]>3.8
    assert errors[1]/errors[2]>3.8


@pytest.mark.parametrize('bad_dt', [0.0,-.01,np.nan,np.inf,True])
def test_bad_dt_atomic(bad_dt):
    p=particles(stress=[[10.,20.,7.,3.,0.,0.]])
    old=p.stress.copy()
    with pytest.raises(ValueError,match='dt'):
        update_objective_linear_elastic_stress(p,np.zeros((1,6)),np.zeros((1,2,2)),LinearElastic(1e3,.2),bad_dt)
    np.testing.assert_array_equal(p.stress,old)


@pytest.mark.parametrize('bad_delta', [np.zeros((1,5)),np.full((1,6),np.nan),np.array([[0,0,1,0,0,0.]])])
def test_bad_strain_atomic(bad_delta):
    p=particles(stress=[[10.,20.,7.,3.,0.,0.]])
    old=p.stress.copy()
    with pytest.raises(ValueError):
        update_objective_linear_elastic_stress(p,bad_delta,np.zeros((1,2,2)),LinearElastic(1e3,.2),.01)
    np.testing.assert_array_equal(p.stress,old)


@pytest.mark.parametrize('L', [np.zeros((1,3,3)),np.full((1,2,2),np.inf),np.ones((1,2,2))])
def test_bad_gradient_or_inconsistent_rate_atomic(L):
    p=particles(stress=[[10.,20.,7.,3.,0.,0.]])
    old=p.stress.copy()
    with pytest.raises(ValueError):
        update_objective_linear_elastic_stress(p,np.zeros((1,6)),L,LinearElastic(1e3,.2),.01)
    np.testing.assert_array_equal(p.stress,old)


def test_invalid_material_mapping_atomic():
    p=particles(ids=[0,1]); old=p.stress.copy()
    L=np.zeros((2,2,2))
    with pytest.raises(ValueError,match='material_id'):
        update_objective_linear_elastic_stress(p,np.zeros((2,6)),L,{0:LinearElastic(1e3,.2)},.01)
    np.testing.assert_array_equal(p.stress,old)


def test_bad_existing_stress_atomic():
    p=particles(stress=[[10.,20.,7.,3.,0.,0.]])
    p.stress[0,4]=2.0
    with pytest.raises(ValueError,match='Plane strain'):
        update_objective_linear_elastic_stress(p,np.zeros((1,6)),np.zeros((1,2,2)),LinearElastic(1e3,.2),.01)


def test_solver_option_and_legacy_default():
    p=particles()
    grid=BackgroundGrid((0,0),(1,1),(1,1))
    obj=ExplicitMPMSolver(p,grid,LinearElastic(1000,.2),dt=.01,stress_update='corotational')
    assert obj.stress_update == 'corotational'
    legacy=ExplicitMPMSolver(p,grid,LinearElastic(1000,.2),dt=.01)
    assert legacy.stress_update == 'incremental'
    for mode in ('jaumann','',None,1,True):
        with pytest.raises(ValueError,match='stress_update'):
            ExplicitMPMSolver(p,grid,LinearElastic(1000,.2),dt=.01,stress_update=mode)


def test_solver_rollback_objective_mode_when_particle_exits_grid():
    grid=BackgroundGrid((0,0),(1,1),(1,1))
    p=ParticleSet(position=[[.99,.5]],mass=[1.],volume=[.25],velocity=[[3.,0.]],
                  stress=[[10.,-2.,4.,5.,0.,0.]])
    solver=ExplicitMPMSolver(p,grid,LinearElastic(10e2,.2),dt=.01,stress_update='corotational',cfl=1)
    original={field:getattr(p,field).copy() for field in solver._PARTICLE_MUTABLE}
    with pytest.raises(ValueError):
        solver.step()
    for name,data in original.items():
        np.testing.assert_array_equal(getattr(p,name),data)
    assert solver.step_count==0


def test_rigid_spin_full_solver_and_legacy_comparison():
    a=rigid_spin(stress_update='corotational', n_steps=80)
    b=rigid_spin(stress_update='incremental', n_steps=80)
    assert a.passed
    assert a.metrics['stress_objectivity_error_pa']<1e-8
    assert a.metrics['strain_abs_max']<1e-10
    assert a.metrics['discrete_position_error_m']<1e-10
    assert a.metrics['jacobian_final']>1.0  # Stage3D kinematic limitation exposed
    assert b.metrics['stress_objectivity_error_pa']>10.0


def test_runner_creates_artifacts(tmp_path):
    assert run_stage7(['--output',str(tmp_path)])==0
    d=json.loads((tmp_path/'7_summary.json').read_text())
    assert d['passed']
    assert (tmp_path/'7_corotational_history.csv').exists()
    assert (tmp_path/'7_incremental_history.csv').exists()
    assert d['legacy_comparison']['passed'] is False


def test_kernel_numba_is_compiled():
    p=particles()
    L=np.zeros((1,2,2))
    update_objective_linear_elastic_stress(p,np.zeros((1,6)),L,LinearElastic(1e3,.2),.01)
    assert objective_elastic_plane_strain_kernel.signatures
