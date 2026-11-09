"""Stage 10E-B2: general 2D velocity and force feedback, independent linear reference."""
import numpy as np
import pytest

from validation.stage10eb2.general_2d_waves import (
    BOX0, BOX1, CORE0, CORE1, CL, CT, E, LAM, MU, NU, RHO,
    METHODS, _modes, exact_linear, make_case, run_one, cpdi_edge_mismatch,
)


def test_reference_initial_zero_displacement_and_stress():
    X=np.array([[3.,4.],[4.,6.],[7.,5.]])
    r=exact_linear(X,0.)
    np.testing.assert_allclose(r['position'],X,rtol=0,atol=1e-14)
    np.testing.assert_allclose(r['F'],np.repeat(np.eye(2)[None],len(X),axis=0),atol=1e-14)
    assert np.all(r['stress']==0)
    assert np.linalg.norm(r['velocity'])>0


def test_reference_has_nonzero_2d_gradient_and_spin():
    X=np.array([[4.28,5.23],[5.11,6.04],[6.32,4.78]])
    r=exact_linear(X,.2,amplitude_scale=.5)
    assert np.max(np.abs(r['F'][:,0,1]))>1e-4
    assert np.max(np.abs(r['F'][:,1,0]))>1e-4
    assert np.max(np.abs(r['F'][:,0,1]-r['F'][:,1,0]))>1e-4


def test_linear_reference_plane_wave_modes_are_longitudinal_and_transverse():
    modes=_modes(.25)
    assert np.isclose(np.dot(modes[0][1],modes[0][2]),np.linalg.norm(modes[0][1]))
    assert np.isclose(np.dot(modes[1][1],modes[1][2]),0,atol=1e-15)
    assert CL>CT>0


def test_initial_field_varies_in_both_spatial_directions():
    v0=exact_linear(np.array([[4.1,5.2],[4.6,5.2],[4.1,5.7]]),0.)['velocity']
    assert np.linalg.norm(v0[1]-v0[0])>1e-5
    assert np.linalg.norm(v0[2]-v0[0])>1e-5
    assert np.all(np.max(np.abs(v0),axis=0)>1e-6)


@pytest.mark.parametrize('method',METHODS)
def test_mpm_velocity_2d_is_unconstrained(method):
    X,core,p,g,solver=make_case(method,h=1.)
    assert not np.any(g.velocity_bc_mask)
    assert np.any(core)
    assert np.any(p.velocity[:,0] != 0)
    assert np.any(p.velocity[:,1] != 0)
    assert np.count_nonzero(p.stress)==0
    assert solver.stress_update=='corotational'
    assert solver.kinematics_update=='exponential'


@pytest.mark.parametrize('method',METHODS)
def test_force_driven_both_components_and_conservation(method):
    r,_,(X,core,p)=run_one(method,h=1.,steps=25)
    assert r['max_nodal_force_x_N']>1e-5
    assert r['max_nodal_force_y_N']>1e-5
    assert r['mass_relative_error']<1e-12
    assert r['momentum_absolute_error_kg_m_s']<1e-7
    assert r['peak_Fxy']>1e-5
    assert r['peak_Fyx']>1e-5
    assert r['min_detF']>0
    assert r['both_grid_components_unconstrained']
    assert np.isfinite(r['stress_relative_l2'])


def test_refinement_improves_cpdi_reference_stress():
    coarse,_,_=run_one('cpdi',h=1.,steps=25)
    finer,_,_=run_one('cpdi',h=.5,steps=25)
    assert finer['stress_relative_l2'] < coarse['stress_relative_l2']


def test_affine_cpdi_domains_have_matching_edges():
    X,core,p,g,_=make_case('cpdi',h=1.)
    A=np.array([[1.06,.28],[-.09,1.13]])
    p.position[:]=X@A.T
    p.deformation_gradient[:]=A
    n=round((BOX1-BOX0)*2/1.)
    maximum,rms=cpdi_edge_mismatch(p,n)
    assert maximum<1e-12
    assert rms<1e-12


def test_non_affine_domains_can_have_edge_mismatch():
    r,_,_=run_one('cpdi',h=.5,steps=35)
    assert r['cpdi_max_shared_edge_mismatch_m']>0
    assert r['cpdi_rms_shared_edge_mismatch_m']>0


def test_invalid_case_inputs():
    with pytest.raises(ValueError):make_case('broken')
    with pytest.raises(ValueError):make_case('cpdi',h=.3)
    with pytest.raises(ValueError):make_case('cpdi',amplitude_scale=0)
    with pytest.raises(ValueError):exact_linear(np.zeros(2),0)
