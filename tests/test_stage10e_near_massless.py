"""Stage 10E-A.1 diagnostic tests: report only, never change solver."""
import numpy as np
import pytest

from validation.stage10e.near_massless import analyze_step, run_diagnostic, DEFAULT_THRESHOLDS
from validation.stage10e.nonlinear_wave import run_one


def _synthetic():
    # One 2 kg particle deposits 1e-20 of its mass to a near-empty node;
    # a second particle deposits its mass to a regular node.
    node_ids = np.array([[0,1], [0,2]], dtype=np.int64)
    weights = np.array([[1. - 1e-20, 1e-20],[0.,1.]])
    particle_mass = np.array([2.,2.])
    nodal_mass = np.array([2.,2e-20,2.])
    nodal_force = np.array([[2.,0.],[2e-18,0.],[2.,0.]])
    nodal_accel = np.array([[1.,0.],[100.,0.],[1.,0.]])
    velocity = np.array([[1.,0.],[100.,0.],[1.,0.]])
    grad = np.zeros((2,2,2), dtype=float)
    grad[0,1,0] = 2e-20
    return nodal_mass,nodal_force,nodal_accel,velocity,node_ids,weights,grad,particle_mass


def test_diagnostic_reconstructs_weighted_flip_and_gradient():
    m,f,a,v,ids,w,grad,mp = _synthetic()
    d = analyze_step(m,f,a,v,ids,w,grad,dt=.003,particle_mass=mp)
    near = d['thresholds'][str(1e-12)]
    assert near['low_node_count'] == 1
    assert near['mass_cutoff_kg'] > 2e-20
    assert d['raw_max_node_id'] == 1
    assert d['raw_max_ax_m_s2'] == 100
    assert near['filtered_max_ax_m_s2'] == 1
    assert near['max_abs_low_flip_dvx_m_s'] == pytest.approx(3e-21)
    assert near['max_abs_low_Lxx_s_inv'] == pytest.approx(2e-18)
    assert near['low_nodal_mass_fraction'] == pytest.approx(5e-21)


def test_threshold_choice_does_not_modify_inputs():
    args=_synthetic()
    before=[a.copy() for a in args]
    diag=analyze_step(*args[:4],*args[4:7],dt=.003,particle_mass=args[7],thresholds=(1e-8,1e-12,1e-16))
    for now,old in zip(args,before):
        np.testing.assert_array_equal(now,old)
    assert len(diag['thresholds'])==3


def test_reject_inconsistent_mass():
    m,f,a,v,ids,w,grad,mp = _synthetic()
    with pytest.raises(ValueError,match='P2G mass'):
        analyze_step(m*1.1,f,a,v,ids,w,grad,dt=.003,particle_mass=mp)


def test_reject_invalid_threshold():
    m,f,a,v,ids,w,grad,mp = _synthetic()
    with pytest.raises(ValueError,match='threshold'):
        analyze_step(m,f,a,v,ids,w,grad,dt=.003,particle_mass=mp,thresholds=(0.0,))


def test_reject_invalid_id():
    m,f,a,v,ids,w,grad,mp = _synthetic()
    ids[0,0]=100
    with pytest.raises(ValueError,match='indices'):
        analyze_step(m,f,a,v,ids,w,grad,dt=.003,particle_mass=mp)


def test_reproduces_cpgimp_near_massless_node_656_at_step_98():
    c,rows,tail,sample = run_diagnostic('cpgimp', steps=98, sample_step=98)
    worst=c['worst_raw_acceleration']
    assert worst['step'] == 98
    assert worst['node_id'] == 656
    assert worst['accel_x_m_s2'] == pytest.approx(94.05851059136754,rel=1e-9)
    assert worst['nodal_mass_kg'] < 1e-30
    assert sample['mass_kg'][656] == pytest.approx(worst['nodal_mass_kg'])
    assert c['thresholds']['1e-12']['max_filtered_ax_m_s2'] < 1.0
    assert c['thresholds']['1e-12']['max_abs_low_flip_dvx_m_s'] < 1e-25
    assert c['thresholds']['1e-12']['max_abs_low_Lxx_s_inv'] < 1e-25
    assert c['max_mass_rel_error'] < 1e-12
    assert c['max_momentum_x_abs_error'] < 1e-6


def test_diagnostic_does_not_change_particle_solution():
    # The original run_one uses the same solver without diagnostics.
    reference,_,_=run_one('cpgimp',nsteps=5)
    traced,_,_,_=run_diagnostic('cpgimp',steps=5)
    assert traced['final_stress_rel_l2'] == pytest.approx(reference['final_stress_rel_l2'],rel=0,abs=1e-13)
    assert traced['max_abs_energy_relative_drift'] == pytest.approx(reference['maximum_energy_relative_drift'],rel=0,abs=1e-13)
    assert traced['max_momentum_x_abs_error'] == pytest.approx(reference['max_momentum_abs_error'],rel=0,abs=1e-13)
