"""Stage 10E-B tests: independent constrained shear wave reference and MPM integration."""
import math
import numpy as np
import pytest

from validation.stage10eb.nonlinear_shear import (
    AMP, C0, MU, RHO0, T_END, c, characteristics,
    characteristic_jacobian_min, gamma0, make_case, reference, run_one,
    velocity_of_gamma,
)


def test_shear_reference_initial_matches_jaumann_stress_branch():
    y=np.array([3.5,4.5,5.,6.,7.])
    r=reference(y,0.)
    assert np.allclose(r['gamma'],gamma0(y))
    assert np.allclose(r['sig_xy'],MU*np.sin(gamma0(y)))
    assert np.allclose(r['sig_xx'],-r['sig_yy'])
    assert np.allclose(r['F_xy'],0)
    assert np.allclose(r['polar_angle_rad'],0)


def test_velocity_characteristic_invariant():
    a=np.linspace(.01,AMP,18)
    eps=1e-6
    derivative=(velocity_of_gamma(a+eps)-velocity_of_gamma(a-eps))/(2*eps)
    assert np.allclose(derivative,-c(a),atol=1e-9,rtol=1e-8)


def test_implicit_characteristics_are_solved_independently():
    y=np.linspace(2.2,7.8,61)
    t=.7
    xi,gg=characteristics(y,t)
    assert np.allclose(xi+t*c(gg),y,atol=5e-13)
    assert np.allclose(gg,gamma0(xi))
    assert characteristic_jacobian_min(T_END)>.5


def test_reference_satisfies_reduced_pde():
    yy=np.linspace(4.0,6.0,33)
    t=.55
    delta=2e-5
    yp=reference(yy+delta,t)['gamma']
    ym=reference(yy-delta,t)['gamma']
    tp=reference(yy,t+delta)['gamma']
    tm=reference(yy,t-delta)['gamma']
    gt=(tp-tm)/(2*delta)
    gy=(yp-ym)/(2*delta)
    r=reference(yy,t)
    assert np.max(np.abs(gt+c(r['gamma'])*gy))<4e-7
    vp=reference(yy+delta,t)['v_x'];vm=reference(yy-delta,t)['v_x']
    assert np.max(np.abs(gt-(vp-vm)/(2*delta)))<5e-7


def test_characteristics_reject_wrong_time():
    with pytest.raises(ValueError):characteristics(np.array([5.]),-1.)
    with pytest.raises(ValueError):characteristics(np.array([5.]),float('nan'))


@pytest.mark.parametrize('method',['mpm','gimp','cpgimp','cpdi'])
def test_shear_setup_has_only_vertical_grid_constraints(method):
    x,core,p,g,s=make_case(method,h=.5,ppc=2)
    assert p.n_particles==576
    assert np.count_nonzero(core)>0
    assert np.all(g.velocity_bc_mask[:,1])
    assert not np.any(g.velocity_bc_mask[:,0])
    assert np.allclose(g.velocity_bc_value,0)
    assert s.nodal_external_force is None
    assert np.all(s.gravity==0.)
    assert np.allclose(p.deformation_gradient,np.eye(2))
    expected=reference(x[:,1],0)
    assert np.allclose(p.stress[:,3],expected['sig_xy'])
    assert np.allclose(p.velocity[:,0],expected['v_x'])


def test_cpdi_genuine_force_feedback_and_local_rotation():
    r,history,profile=run_one('cpdi',h=.5,nsteps=45,dt=.003)
    assert r['max_internal_force_N']>0
    assert r['max_nodal_ax_m_s2']>0
    assert r['numerical_max_polar_rotation_deg']>0.05
    assert r['max_transverse_displacement_m']<1e-12
    assert r['final_mass_rel_error']<1e-13
    assert r['max_x_momentum_abs_error']<1e-7
    assert r['min_detF']>0
    assert np.isfinite(r['max_reduced_core_energy_drift'])
    assert r['shear_stress_relative_l2']<.40
    assert r['core_max_detF_deviation']<.02
    assert history and profile is None


def test_cpgimp_near_massless_stats_are_only_diagnostics():
    r,_,_=run_one('cpgimp',h=.5,nsteps=18,dt=.003)
    assert r['max_near_massless_node_count']>=0
    assert r['max_near_massless_mass_fraction']>=0.
    assert r['final_mass_rel_error']<1e-12
    assert r['max_x_momentum_abs_error']<1e-7


def test_invalid_benchmark_configuration_rejected():
    with pytest.raises(ValueError):make_case('unknown')
    with pytest.raises(ValueError):make_case('cpdi',ppc=3)
    with pytest.raises(ValueError):make_case('cpdi',h=.3)


def test_edge_diagnostic_mass_threshold_is_observational_only():
    from validation.stage10eb.edge_node_diagnostics import analyze_peak
    a=analyze_peak('cpdi',3,h=.5)
    assert a['peak_step']==3
    assert a['raw_max_ax_m_s2']>=0
    assert a['grid_mass_sum_kg']>0
    assert '1e-08' in a['thresholds']
    assert a['thresholds']['1e-08']['max_abs_low_flip_dvx_upper_m_s']>=0
