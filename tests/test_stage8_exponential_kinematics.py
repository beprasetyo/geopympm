"""Stage 8: exponential-map deformation gradient update verification."""
from math import sin, cos, exp, pi
import numpy as np
import pytest
from geopympm.core import ParticleSet, BackgroundGrid
from geopympm.interpolation import evaluate_quad4
from geopympm.solvers import particle_to_grid, update_particle_kinematics, ExplicitMPMSolver
from geopympm.materials import LinearElastic
from geopympm.kernels.kinematics_exponential import _matrix_exponential_2x2
from validation.stage8.finite_rotation import rotation_comparison
from validation.stage8.run_kinematics import main as run_stage8


def fixture(A, *, n=1, F=None):
    x=np.array([[.3+i*.3/(max(n,2)) , .4] for i in range(n)])
    p=ParticleSet(position=x, mass=np.ones(n), volume=np.full(n,.2))
    g=BackgroundGrid((0,0),(1,1),(1,1))
    ip=evaluate_quad4(p,g)
    particle_to_grid(p,g,ip)
    g.velocity[:]=g.node_coordinates @ np.asarray(A,dtype=float).T
    if F is not None:
        p.deformation_gradient[:]=F
        p.volume[:]=p.volume0 * np.linalg.det(F)
    return p,g,ip


@pytest.mark.parametrize('w,dt',[(1.,.01), (4.,.1), (-3.2,.17), (2*pi,.5)])
def test_pure_spin_orthogonality_and_volume(w,dt):
    p,g,ip=fixture([[0,-w],[w,0]])
    r=update_particle_kinematics(p,g,ip,dt,kinematics_update='exponential')
    angle=w*dt
    R=np.array([[cos(angle),-sin(angle)],[sin(angle),cos(angle)]])
    np.testing.assert_allclose(p.deformation_gradient,[R],rtol=0,atol=1e-14)
    np.testing.assert_allclose(p.deformation_gradient[0].T@p.deformation_gradient[0],np.eye(2),rtol=0,atol=1e-14)
    np.testing.assert_allclose(r.jacobian,[1],rtol=0,atol=1e-14)
    np.testing.assert_allclose(p.volume,p.volume0,rtol=0,atol=1e-14)
    np.testing.assert_allclose(p.strain,0,rtol=0,atol=1e-14)


@pytest.mark.parametrize('rate,dt',[(0.3,.1),(-.9,.07),(2.,.04)])
def test_volumetric_change_matches_exponential_trace(rate,dt):
    p,g,ip=fixture([[rate,0],[0,rate]])
    r=update_particle_kinematics(p,g,ip,dt,kinematics_update='exponential')
    factor=exp(rate*dt)
    np.testing.assert_allclose(p.deformation_gradient,[np.eye(2)*factor],rtol=1e-13)
    np.testing.assert_allclose(r.jacobian,[exp(2*rate*dt)],rtol=1e-13)
    np.testing.assert_allclose(p.volume,p.volume0*exp(2*rate*dt),rtol=1e-13)


def test_nilpotent_simple_shear_agrees_with_euler():
    p,g,ip=fixture([[0,2.],[0,0]])
    update_particle_kinematics(p,g,ip,.1,kinematics_update='exponential')
    np.testing.assert_allclose(p.deformation_gradient,[[[1,.2],[0,1]]],atol=1e-14)
    np.testing.assert_allclose(p.volume,p.volume0,atol=1e-14)
    np.testing.assert_allclose(p.strain,[[0,0,0,.2,0,0]],atol=1e-14)


@pytest.mark.parametrize('A',[ [[.2,.4],[.1,-.3]], [[.2,-1.],[1.4,-.1]], [[-.1,.7],[-.05,.1]], [[.2,.0],[.0,-.2]] ])
def test_general_constant_gradient_matches_independent_series(A):
    A=np.asarray(A,dtype=float); dt=.2
    F=np.array([[1.3,.15],[.4,1.1]])
    p,g,ip=fixture(A,F=F[None])
    update_particle_kinematics(p,g,ip,dt,kinematics_update='exponential')
    # Independent matrix-power series, not the Stage-8 closed form.
    B=dt*A
    term=np.eye(2)
    E=np.eye(2)
    for k in range(1,45):
        term=term@B/k
        E=E+term
    ref=E@F
    np.testing.assert_allclose(p.deformation_gradient[0],ref,rtol=1e-12,atol=2e-14)
    np.testing.assert_allclose(p.volume, [p.volume0[0]*np.linalg.det(ref)],rtol=1e-12)


def test_multi_particle_affine_same_gradient_and_atomic_result():
    p,g,ip=fixture([[.2,-.3],[.4,-.1]],n=2)
    r=update_particle_kinematics(p,g,ip,.01,kinematics_update='exponential')
    np.testing.assert_allclose(r.velocity_gradient,[ [[.2,-.3],[.4,-.1]]]*2,atol=1e-14)
    np.testing.assert_allclose(p.deformation_gradient[0],p.deformation_gradient[1],atol=1e-14)


def test_euler_still_default_legacy_result():
    p,g,ip=fixture([[0,-1],[1,0]])
    update_particle_kinematics(p,g,ip,.01)
    np.testing.assert_allclose(p.volume,[.2*1.0001],atol=1e-15)


@pytest.mark.parametrize('bad',['bad','',None,True,1])
def test_bad_mode_rejected_without_mutation(bad):
    p,g,ip=fixture([[0,-1],[1,0]])
    saved=p.deformation_gradient.copy(),p.volume.copy(),p.strain.copy()
    with pytest.raises(ValueError,match='kinematics_update'):
        update_particle_kinematics(p,g,ip,.01,kinematics_update=bad)
    for a,b in zip((p.deformation_gradient,p.volume,p.strain),saved):
        np.testing.assert_array_equal(a,b)


def test_large_exponential_overflow_rejected_without_partial_commit():
    p,g,ip=fixture([[1000.,0],[0,0]])
    state=[p.deformation_gradient.copy(),p.volume.copy(),p.strain.copy()]
    with pytest.raises((ValueError,OverflowError),match='nonfinite|overflow'):
        update_particle_kinematics(p,g,ip,1.,kinematics_update='exponential')
    for a,b in zip((p.deformation_gradient,p.volume,p.strain),state):
        np.testing.assert_array_equal(a,b)


def test_solver_new_option_and_default_unchanged():
    p,g,_=fixture([[0,-1],[1,0]])
    m=LinearElastic(1000,.25)
    s=ExplicitMPMSolver(p,g,m,dt=.001,stress_update='corotational',kinematics_update='exponential')
    assert s.kinematics_update == 'exponential'
    assert ExplicitMPMSolver(p,g,m,dt=.001).kinematics_update == 'euler'
    for invalid in (None, 'abc', 1, True):
        with pytest.raises(ValueError,match='kinematics_update'):
            ExplicitMPMSolver(p,g,m,dt=.001,kinematics_update=invalid)


def test_solver_rollback_when_exponential_step_leaves_domain():
    p=ParticleSet(position=[[.99,.5]],mass=[1],volume=[.25],velocity=[[3,0]],stress=[[10,-2,4,5,0,0]])
    g=BackgroundGrid((0,0),(1,1),(1,1))
    s=ExplicitMPMSolver(p,g,LinearElastic(1000,.25),dt=.01,cfl=1,
                        stress_update='corotational',kinematics_update='exponential')
    old={k:getattr(p,k).copy() for k in s._PARTICLE_MUTABLE}
    with pytest.raises(ValueError): s.step()
    for k,v in old.items(): np.testing.assert_array_equal(getattr(p,k),v)
    assert s.step_count==0


def test_full_solver_rigid_spin_reference_and_euler_position_limit():
    result=rotation_comparison()
    assert result.passed
    assert result.metrics['euler_j_final'] == pytest.approx(1.0009004,rel=1e-6)
    assert result.metrics['exponential_j_final'] == pytest.approx(1.0,abs=5e-12)
    assert result.metrics['euler_position_error_end_m'] == pytest.approx(result.metrics['exponential_position_error_end_m'],abs=1e-13)


def test_cli_results_written(tmp_path):
    assert run_stage8(['--output',str(tmp_path)])==0
    assert (tmp_path/'8_summary.json').is_file()
    assert (tmp_path/'8_rigid_rotation_history.csv').is_file()
