"""Stage-6F regression tests: energy, separated spatial/temporal refinement, CFL."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from geopympm.core.particles import ParticleSet
from validation.stage6.energy_convergence import (
    cfl_guard_check, energy_comparison, full_stage6f, particle_energy,
    spatial_convergence, temporal_convergence,
)
from validation.stage6.run_energy import main


@pytest.fixture(scope='module')
def suite():
    return full_stage6f()


def test_basic_particle_energy_zero():
    p=ParticleSet([[0.,0.]],mass=[4.],volume=[2.],velocity=[[0.,0.]])
    e=particle_energy(p)
    assert e==dict(kinetic_energy_j=0.,elastic_energy_j=0.,total_energy_j=0.)


def test_kinetic_energy_formula():
    p=ParticleSet([[0.,0.]],mass=[4.],volume=[2.],velocity=[[3.,4.]])
    assert particle_energy(p)['kinetic_energy_j']==pytest.approx(50.)


def test_elastic_energy_plane_strain_including_sigma_zz():
    p=ParticleSet([[0.,0.]],mass=[4.],volume=[2.])
    p.stress[0,:]=[10.,20.,30.,0.,0.,0.]
    p.strain[0,:]=[.1,.2,.3,0.,0.,0.]
    # 0.5 V0(1+4+9)=14 J
    assert particle_energy(p)['elastic_energy_j']==pytest.approx(14.)


def test_elastic_energy_engineering_shear_convention():
    p=ParticleSet([[0.,0.]],mass=[4.],volume=[2.])
    p.stress[0,3]=6.0
    p.strain[0,3]=0.1  # gamma_xy, NOT eps_xy
    assert particle_energy(p)['elastic_energy_j']==pytest.approx(0.6)


def test_energy_is_independent_of_deformed_volume():
    p=ParticleSet([[0.,0.]],mass=[4.],volume=[2.])
    p.stress[0,0]=10
    p.strain[0,0]=.1
    base=particle_energy(p)['elastic_energy_j']
    p.volume[0]*=0.5
    assert particle_energy(p)['elastic_energy_j']==base


def test_energy_rejects_nonfinite_inputs():
    p=ParticleSet([[0.,0.]],mass=[4.],volume=[2.])
    p.stress[0,0]=np.nan
    with pytest.raises(ValueError,match='nonfinite'):
        particle_energy(p)


def test_energy_rejects_wrong_particle_type():
    with pytest.raises(TypeError,match='ParticleSet'):
        particle_energy(object())


def test_energy_suite_pass(suite):
    assert suite['energy'].passed
    assert suite['passed']


def test_flip_maintains_near_constant_total_energy(suite):
    m=suite['energy'].metrics['FLIP']
    assert abs(m['final_energy_ratio']-1) < .02
    assert m['max_abs_energy_residual'] < .02


def test_energy_is_initially_elastic_not_kinetic(suite):
    r=suite['energy'].histories['FLIP'][0]
    assert r['kinetic_energy_j']==pytest.approx(0.,abs=1e-14)
    assert r['elastic_energy_j']>0
    assert r['total_energy_j']==pytest.approx(r['elastic_energy_j'])


def test_elastic_kinetic_exchange(suite):
    h=suite['energy'].histories['FLIP']
    assert max(row['kinetic_energy_j'] for row in h)>0.05
    assert min(row['elastic_energy_j'] for row in h)<h[0]['elastic_energy_j']*.1


def test_energy_ratio_comparison(suite):
    m=suite['energy'].metrics
    assert m['FLIP']['final_energy_ratio']>m['BLEND_0p05']['final_energy_ratio']
    assert m['BLEND_0p05']['final_energy_ratio']>m['PIC']['final_energy_ratio']


def test_energy_history_is_finite(suite):
    for history in suite['energy'].histories.values():
        assert len(history)==1201
        assert all(all(np.isfinite(row[key]) for key in (
            'kinetic_energy_j','elastic_energy_j','total_energy_j',
            'relative_energy_residual')) for row in history)


def test_unsupported_energy_transfer_rejected():
    with pytest.raises(ValueError,match='alphas'):
        energy_comparison(alphas=(0.,1.))


def test_space_suite_pass_and_reduces_error(suite):
    space=suite['spatial']
    assert space.passed
    errors=[r['max_stress_l2_error'] for r in space.rows]
    assert errors[0]>errors[1]>errors[2]>0


def test_spatial_dt_constant_nx_changes(suite):
    rows=suite['spatial'].rows
    assert [r['nx'] for r in rows]==[16,32,64]
    assert len({r['dt_s'] for r in rows})==1
    assert [r['dx_m'] for r in rows]==[.25,.125,.0625]


def test_spatial_observed_order_near_one(suite):
    assert all(.8<p<1.2 for p in suite['spatial'].orders)


def test_temporal_suite_pass(suite):
    temporal=suite['temporal']
    assert temporal.passed
    assert [r['dt_s'] for r in temporal.rows]==[.002,.001,.0005,.00025]
    assert len({r['nx'] for r in temporal.rows})==1


def test_temporal_reference_error_decreases(suite):
    rows=suite['temporal'].rows
    for key in ('final_stress_relative_reference_error',
                'final_displacement_relative_reference_error'):
        e=[r[key] for r in rows]
        assert e[0]>e[1]>e[2]>e[3]>0


def test_temporal_self_convergence_about_first_order(suite):
    assert all(.8<p<1.2 for p in suite['temporal'].orders)


def test_distinguishes_temporal_ref_vs_exact_continuum(suite):
    rows=suite['temporal'].rows
    assert rows[-1]['final_stress_relative_reference_error'] < .001
    # Continuum error persists from finite spatial discretization, even at fine dt.
    assert rows[-1]['max_stress_l2_vs_continuum'] > .01


def test_cfl_guard_pass(suite):
    assert suite['cfl']['passed']
    assert suite['cfl']['rejected']
    assert suite['cfl']['rejected_step_state_intact']


def test_cfl_limit_is_physically_consistent(suite):
    cfl=suite['cfl']
    expected=.25*.125/cfl['wave_speed_m_s']
    assert cfl['dt_limit_s']==pytest.approx(expected,rel=1e-13)
    assert cfl['accepted_dt_s']<cfl['dt_limit_s']<cfl['rejected_dt_s']


def test_unsupported_spatial_configuration_rejected():
    with pytest.raises(ValueError,match='nx'):
        spatial_convergence(nx_values=(16,32))


def test_unsupported_temporal_configuration_rejected():
    with pytest.raises(ValueError,match='dt list'):
        temporal_convergence(dt_values=(.001,.0005))


def test_cli_outputs_reproducible_reports(tmp_path,suite,monkeypatch):
    # Reuse the numerically verified fixture to test CLI serialization; avoid
    # re-running all long simulations within the full regression suite.
    monkeypatch.setattr('validation.stage6.run_energy.full_stage6f', lambda: suite)
    code=main(['--output',str(tmp_path)])
    assert code==0
    summary=json.loads((tmp_path/'6F_summary.json').read_text(encoding='utf-8'))
    assert summary['passed']
    assert summary['spatial']['passed']
    assert summary['temporal']['passed']
    assert summary['cfl']['passed']
    for file in ('6F_FLIP_energy_history.csv','6F_BLEND_0p05_energy_history.csv',
                 '6F_PIC_energy_history.csv','6F_spatial_refinement.csv',
                 '6F_temporal_refinement.csv'):
        assert (tmp_path/file).exists()


def test_cfl_guard_reproducible():
    a=cfl_guard_check(); b=cfl_guard_check()
    assert a['passed'] and b['passed']
    assert a['dt_limit_s']==pytest.approx(b['dt_limit_s'])
