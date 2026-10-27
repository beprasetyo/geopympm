"""Stage 6E: analytical checks of point-MPM gradient jumps/force transfer."""
from __future__ import annotations

import json
import numpy as np
import pytest

from validation.stage6.cell_crossing import (
    exact_nodal_force_x,
    moving_stressed_block,
    one_particle_probe,
    particle_refinement,
    _static_internal_force,
)
from validation.stage6.run_crossing import main


@pytest.fixture(scope='module')
def probe():
    return one_particle_probe()


@pytest.fixture(scope='module')
def block4():
    return moving_stressed_block()


@pytest.fixture(scope='module')
def refined():
    return particle_refinement()


def test_probe_matches_analytical_nodal_limit(probe):
    assert probe.passed
    assert probe.metrics['left_force_n'] == pytest.approx(-12.0, abs=1e-12)
    assert probe.metrics['right_force_n'] == pytest.approx(+12.0, abs=1e-12)
    assert probe.metrics['signed_jump_n'] == pytest.approx(24.0, abs=1e-12)


def test_probe_jump_scales_linearly_with_stress_and_volume():
    result=one_particle_probe(sigma_xx=240.0,volume=0.1)
    assert result.passed
    assert result.metrics['absolute_jump_n']==pytest.approx(24.0)


def test_probe_compressive_stress_reverses_force_sign():
    result=one_particle_probe(sigma_xx=-120.0)
    assert result.passed
    assert result.metrics['signed_jump_n']==pytest.approx(-24.0)


def test_zero_stress_has_zero_internal_force():
    assert _static_internal_force(1.99,volume=0.2,sigma_xx=0.0)==pytest.approx(0.0)
    assert _static_internal_force(2.01,volume=0.2,sigma_xx=0.0)==pytest.approx(0.0)


def test_exact_integral_zero_when_support_covered_on_both_sides():
    result=exact_nodal_force_x(node_x=4.0,x_left=2.0,x_right=6.0,
                               h=1.0,thickness=1.0,sigma_xx=120.0)
    assert result==pytest.approx(0.0)


def test_exact_integral_partial_overlap_left():
    result=exact_nodal_force_x(node_x=5.0,x_left=1.22,x_right=4.22,
                               h=1.0,thickness=1.0,sigma_xx=120.0)
    assert result==pytest.approx(-13.2)


def test_exact_integral_partial_overlap_right():
    result=exact_nodal_force_x(node_x=5.0,x_left=4.8,x_right=5.4,
                               h=1.0,thickness=1.0,sigma_xx=120.0)
    assert result==pytest.approx(12.0)


def test_exact_force_vanishes_outside_support():
    result=exact_nodal_force_x(node_x=6.0,x_left=1.0,x_right=3.0,
                               h=1.0,thickness=1.0,sigma_xx=120.0)
    assert result==0.0


def test_block4_is_continuum_integral_consistent(block4):
    assert block4.benchmark.passed
    m=block4.benchmark.metrics
    assert m['max_force_error_n']==pytest.approx(7.5,abs=1e-9)
    assert m['normalized_max_force_error'] < 0.25
    assert m['max_force_change_noise_n'] > 1
    assert m['crossing_particles'] > 0


def test_motion_is_prescribed_and_stress_constant(block4):
    m=block4.benchmark.metrics
    assert m['final_position_error_m'] < 1e-10
    assert m['max_stress_drift_pa'] < 1e-7
    assert all(r['min_jacobian']>0.999999999 for r in block4.benchmark.history)


def test_force_series_has_real_stress_and_crossing_noise(block4):
    h=block4.benchmark.history
    assert len(h)==150
    assert abs(h[0]['continuum_force_n']) > 1
    assert abs(h[0]['numerical_force_n']) > 1
    assert len({round(r['numerical_force_n'],4) for r in h}) >= 3
    assert max(r['abs_error_n'] for r in h)>1
    assert h[0]['time']==0.0
    assert h[-1]['time']==pytest.approx(1.49)


def test_particle_refinement_decreases_force_error(refined):
    assert refined['passed']
    rows=refined['rows']
    assert [r['particles_per_cell'] for r in rows]==[2,4,8]
    for key in ['normalized_max_force_error','normalized_rms_force_error','max_force_change_noise_n']:
        a,b,c=[r[key] for r in rows]
        assert 0<c<b<a


@pytest.mark.parametrize('kwargs', [
    {'epsilon':0.0}, {'epsilon':1e-15}, {'epsilon':0.1},
    {'volume':-1}, {'sigma_xx':0.0}, {'sigma_xx':float('nan')},
])
def test_probe_rejects_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        one_particle_probe(**kwargs)


@pytest.mark.parametrize('kwargs', [
    {'particles_per_cell':0}, {'particles_per_cell':True},
    {'dt':-1}, {'dt':float('nan')}, {'dt':0.07},
    {'t_end':0}, {'speed':-1}, {'speed':100},
    {'sigma_xx':0},
])
def test_block_rejects_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        moving_stressed_block(**kwargs)


def test_refinement_rejects_single_resolution():
    with pytest.raises(ValueError):
        particle_refinement(ppc_values=(4,))


def test_cli_writes_reports(tmp_path):
    exit_code=main(['--output',str(tmp_path),'--refinement','--plots'])
    assert exit_code==0
    data=json.loads((tmp_path/'6E_summary.json').read_text(encoding='utf-8'))
    assert data['passed']
    assert data['probe']['passed']
    assert data['primary_ppc4']['benchmark']['passed']
    assert data['particle_refinement']['passed']
    for name in (
        '6E_probe.csv', '6E_ppc4_history.csv', '6E_ppc2_history.csv',
        '6E_ppc8_history.csv','6E_particle_refinement.csv',
        '6E_single_particle_jump.png', '6E_nodal_force_history.png',
        '6E_force_error_history.png', '6E_particle_refinement.png',
    ):
        assert (tmp_path/name).is_file(),name
