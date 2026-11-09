"""Stage 10E: independent nonlinear continuum reference and force-driven solver."""
from __future__ import annotations

import numpy as np
import pytest

from validation.stage10e.nonlinear_wave import (
    AMP, C0, MODULUS, T_END, characteristics, minimum_characteristic_jacobian,
    make_case, reference, run_one, stretch_initial,
)


@pytest.fixture(scope='module')
def cases():
    return {name: run_one(name,cell_dx=.25,ppc_x=2,dt=.003,nsteps=150)[0]
            for name in ('mpm','gimp','cpgimp','cpdi')}


def test_initial_reference_stretch_is_finite_and_large():
    X=np.array([1.,4.,7.])
    vals=stretch_initial(X)
    assert vals[1] == pytest.approx(1+AMP)
    assert np.all(vals>=1.)


def test_characteristic_implicit_equation():
    X=np.linspace(2.,7.,43)
    t=0.40
    xi,J=characteristics(X,t)
    np.testing.assert_allclose(xi+C0*t/np.sqrt(J),X,rtol=0,atol=2e-14)
    np.testing.assert_allclose(stretch_initial(xi),J,rtol=0,atol=1e-14)


def test_reference_reduces_to_initial_and_log_stress():
    X=np.linspace(2.,6.,27)
    J,v,sigma=reference(X,0.)
    np.testing.assert_allclose(J,stretch_initial(X))
    np.testing.assert_allclose(v,2*C0*(1-np.sqrt(J)))
    np.testing.assert_allclose(sigma,MODULUS*np.log(J))


def test_exact_characteristics_pre_shock():
    assert minimum_characteristic_jacobian(T_END)>.9


def test_manufactured_exact_wave_satisfies_pde_to_difference_accuracy():
    # The independent implicit analytical fields must solve J_t=v_X and
    # v_t=(M/rho0)*(log J)_X, not merely satisfy an algebraic identity.
    from validation.stage10e.nonlinear_wave import RHO0
    x=np.linspace(3.5,5.5,33)
    t=.25
    h=2e-5
    Jplus,vplus,_=reference(x,t+h)
    Jminus,vminus,_=reference(x,t-h)
    Jsp,vsp,_=reference(x+h,t)
    Jsm,vsm,_=reference(x-h,t)
    np.testing.assert_allclose((Jplus-Jminus)/(2*h), (vsp-vsm)/(2*h),rtol=1e-5,atol=1e-8)
    np.testing.assert_allclose((vplus-vminus)/(2*h),
            (MODULUS/RHO0)*(np.log(Jsp)-np.log(Jsm))/(2*h),rtol=1e-5,atol=1e-8)


@pytest.mark.parametrize('method',['mpm','gimp','cpgimp','cpdi'])
def test_initial_state_in_current_mpm_reference(method):
    XY,p,g,_=make_case(method,cell_dx=.25)
    J0=stretch_initial(XY[:,0])
    np.testing.assert_allclose(p.deformation_gradient,np.broadcast_to(np.eye(2), p.deformation_gradient.shape),rtol=0,atol=0)
    np.testing.assert_allclose(p.volume, p.mass/1200.*J0)
    np.testing.assert_allclose(p.stress[:,0],MODULUS*np.log(J0))
    assert np.all(~g.velocity_bc_mask[:,0])
    assert np.all(g.velocity_bc_mask[:,1])


def test_all_methods_are_genuinely_force_driven(cases):
    for data in cases.values():
        assert data['max_internal_force_N']>1
        assert data['max_axial_acceleration_m_s2']>1e-3
        assert data['particle_cell_crossings']>0
        assert data['reference_peak_stretch']>1.5


def test_positive_deformation_and_conservation(cases):
    for data in cases.values():
        assert data['min_J']>.5
        assert data['max_mass_rel_error']<1e-12
        assert data['max_momentum_abs_error']<1e-8
        assert data['maximum_energy_relative_drift']<.02
        assert data['stress_log_stretch_rms_pa']<1e-8


def test_gimp_cpgimp_cpdi_better_than_standard_mpm_here(cases):
    # Diagnostic for THIS benchmark only, not a global solver ranking.
    assert cases['mpm']['final_stress_rel_l2']>.15
    for method in ('gimp','cpgimp','cpdi'):
        assert cases[method]['final_stress_rel_l2']<.035
        assert cases[method]['final_stretch_rel_l2']<.01


def test_cpdi_spatial_refinement_against_analytic_wave():
    coarse=run_one('cpdi',cell_dx=.5,ppc_x=2)[0]
    fine=run_one('cpdi',cell_dx=.25,ppc_x=2)[0]
    assert fine['final_stress_rel_l2']<.5*coarse['final_stress_rel_l2']
