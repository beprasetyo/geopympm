"""Stage 10A-FD regression: force-driven, independent analytic reference."""
from __future__ import annotations

import numpy as np
import pytest

from validation.stage10.force_driven_crossing import (
    AMP, DX, LAME, M, RHO, WAVE_SPEED, analytical, make_case, one_run,
)


def test_exact_linear_elastic_travelling_wave_initialization():
    X, p, grid, solver = make_case('gimp', drift=1.0, ppc_x=2)
    xa, va, sa = analytical(X[:,0], 0.0, 1.0)
    np.testing.assert_allclose(p.position[:,0], xa, rtol=0, atol=1e-13)
    np.testing.assert_allclose(p.velocity[:,0], va, rtol=0, atol=1e-13)
    np.testing.assert_allclose(p.stress[:,0], sa, rtol=0, atol=1e-13)
    np.testing.assert_allclose(p.stress[:,1], sa*LAME/M, rtol=1e-14, atol=1e-13)
    np.testing.assert_allclose(p.mass.sum(), RHO*10.0*0.5, rtol=1e-14)
    assert np.any(np.abs(p.stress[:,0])>1.0)
    assert np.all(~grid.velocity_bc_mask[:,0])
    assert np.all(grid.velocity_bc_mask[:,1])
    assert WAVE_SPEED > 1.0
    assert solver.interpolation == 'gimp'


@pytest.mark.parametrize('method',('mpm','gimp'))
def test_force_driven_coupling_stress_and_momentum(method):
    X,p,g,solver=make_case(method,drift=1.0,ppc_x=2)
    s0=p.stress[:,0].copy()
    v0=p.velocity[:,0].copy()
    momentum0=float(p.mass@p.velocity[:,0])
    for _ in range(20):
        solver.step()
    assert np.linalg.norm(g.force_internal[:,0])>0
    assert np.linalg.norm(p.stress[:,0]-s0)>1e-5
    assert np.linalg.norm(p.velocity[:,0]-v0)>1e-6
    assert abs((p.mass@p.velocity[:,0])-momentum0) < 1e-9
    assert not np.any(g.velocity_bc_mask[:,0])


@pytest.mark.parametrize('method',('mpm','gimp'))
def test_real_cell_crossing_and_finite_analytic_stress_error(method):
    rows,_,m,stress,_=one_run(method,drift=1.0,ppc_x=2,steps=40,
                              dt=0.002,sample_every=10)
    assert m['n_particle_cell_crossings']>=100
    assert m['force_feedback'] is True
    assert m['velocity_bc_only_y'] is True
    assert m['max_mass_relative_error'] < 1e-12
    assert m['max_momentum_x_relative_error'] < 1e-10
    assert 0 < m['final_stress_relative_l2'] < 1.0
    assert len(rows)==5
    assert np.isfinite(stress).all()


def test_refined_grid_preserves_valid_analytic_setup():
    for h in (0.5,0.25,0.125):
        X,p,g,_=make_case('gimp',drift=0.5,ppc_x=2,cell_dx=h)
        assert len(X)==int(round(10/h))*2*4
        assert np.all(p.domain_size[:,0] == h/2)
        assert np.allclose(g.cell_size,[h,0.25])


def test_analytical_wave_moves_without_stress_change_in_material_frame():
    X=np.linspace(3.0,7.0,21)
    t=0.13
    sigma0=analytical(X,0.0,0.0)[2]
    sigma1=analytical(X+WAVE_SPEED*t,t,1.0)[2]
    np.testing.assert_allclose(sigma0,sigma1,rtol=1e-14,atol=1e-14)


def test_full_benchmark_export_smoke(tmp_path):
    from validation.stage10.force_driven_crossing import run
    result = run(tmp_path/'force_driven',steps=45,ppc_x=2,plot=False)
    assert all(v is True for k,v in result['checks'].items() if v is not None)
    assert result['summary']['mpm_drift_1']['n_particle_cell_crossings'] > 0
    assert result['summary']['gimp_drift_1']['n_particle_cell_crossings'] > 0
    assert (tmp_path/'force_driven'/'10A_FD_timeseries.csv').is_file()
    assert (tmp_path/'force_driven'/'10A_FD_stress_snapshots.csv').is_file()
    assert (tmp_path/'force_driven'/'10A_FD_summary.json').is_file()
