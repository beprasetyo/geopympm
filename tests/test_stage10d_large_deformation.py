"""Stage 10D: large shear, rigid rotation, combined affine deformation and tiling."""
import math
import numpy as np
import pytest

from geopympm.materials import LinearElastic
from validation.stage10d.validate_large_deformation import (
    expm_reference, stress_reference, prescribed_rate, run_affine,
    geometry_comparison, nonaffine_domain_diagnostic,
    shared_edge_mismatch, polygon_area, overlap_area_convex,
    pairwise_overlap_area, run_refinement,
)


def test_reference_exponential_handles_defective_simple_shear():
    A,_,_ = prescribed_rate('shear')
    np.testing.assert_allclose(expm_reference(A, 2.), [[1.,2.],[0.,1.]],
                               rtol=0,atol=1e-14)


def test_reference_spin_rotates_half_turn():
    A,_,_=prescribed_rate('spin')
    np.testing.assert_allclose(expm_reference(A,2.),-np.eye(2),rtol=0,atol=1e-14)


def test_independent_jaumann_simple_shear_solution():
    mat=LinearElastic(1200.,.25)
    mu=mat.shear_modulus
    A,_,_=prescribed_rate('shear')
    result=stress_reference(A,2.,mat,np.zeros(4))
    np.testing.assert_allclose(result[:4],
        [mu*(1.-math.cos(2.)),-mu*(1.-math.cos(2.)),0.,mu*math.sin(2.)],
        atol=1e-10,rtol=0)


@pytest.mark.parametrize('mode',["shear", "spin", "combined"])
def test_large_affine_solver_vs_independent_reference(mode):
    result,_ = run_affine(mode, n_side=3, dt=.01)
    assert result['max_F_error'] < 5e-10
    assert result['max_position_error_m'] < 5e-10
    assert result['max_corner_error_m'] < 5e-10
    assert result['max_stress_error_pa'] < .01
    assert result['max_J_error'] < 5e-10
    assert result['max_domain_area_relative_error'] < 5e-10
    assert result['shared_edge_mismatch_m'] < 5e-10
    assert result['mass_relative_error'] == pytest.approx(0.,abs=1e-14)


def test_shear_domains_tiled_by_cpdi_not_by_ugimp_cpgimp():
    r=geometry_comparison(n_side=4,shear=2.)
    assert r['cpdi']['edge_mismatch_m']<1e-10
    assert r['cpdi']['pairwise_overlap_area_m2']<1e-10
    assert r['ugimp']['edge_mismatch_m']>.7
    assert r['cpgimp']['edge_mismatch_m']>.7


def test_nonaffine_local_cpdi_can_generate_gaps_and_overlaps():
    r=nonaffine_domain_diagnostic()
    assert r['edge_mismatch_m']>.05
    assert r['pairwise_overlap_area_m2']>.01


def test_convex_overlap_algorithm_on_known_boxes():
    A=np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
    B=A+np.array([.5,.5])
    assert polygon_area(A)==pytest.approx(1.)
    assert overlap_area_convex(A,B)==pytest.approx(.25,abs=1e-12)
    assert pairwise_overlap_area(np.stack([A,B]))==pytest.approx(.25,abs=1e-12)
    assert overlap_area_convex(A,A+np.array([1.,0.]))==pytest.approx(0.,abs=1e-12)


def test_large_shear_affine_grid_invariance_and_domain_consistency():
    r1,_=run_affine('shear',n_side=2,dt=.01,grid_step=1.)
    r2,_=run_affine('shear',n_side=2,dt=.01,grid_step=.25)
    assert abs(r1['max_stress_error_pa']-r2['max_stress_error_pa'])<1e-8
    assert r1['max_F_error']<1e-10 and r2['max_F_error']<1e-10


def test_time_refinement_second_order_stress(tmp_path):
    result=run_refinement(tmp_path,n_side=2)
    assert all(result['checks'].values()),result
    assert (tmp_path/'10D_refinement_summary.json').exists()
