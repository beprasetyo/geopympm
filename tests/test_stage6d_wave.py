"""Stage 6D regression tests: force-driven elastic wave, not prescribed interior motion."""
from __future__ import annotations

import json
import numpy as np
import pytest

from validation.stage6.elastic_wave import vibrating_bar, grid_refinement
from validation.stage6.run_wave import main


@pytest.fixture(scope='module')
def flip():
    return vibrating_bar()


@pytest.fixture(scope='module')
def pic():
    return vibrating_bar(pic_fraction=1.0)


def test_standing_wave_matches_continuum_with_flip(flip):
    assert flip.benchmark.passed
    m = flip.benchmark.metrics
    assert m['displacement_relative_l2_max'] < 0.03
    assert m['velocity_relative_l2_max'] < 0.04
    assert m['stress_relative_l2_max'] < 0.08


def test_period_is_physically_consistent(flip):
    m = flip.benchmark.metrics
    assert m['period_analytical_s'] == pytest.approx(2 * 4.0 / np.sqrt(144000.0 / 1200.0))
    assert m['relative_period_error'] < 0.015


def test_internal_forces_drive_motion(flip):
    m = flip.benchmark.metrics
    assert m['max_internal_force_n'] > 1
    assert m['min_jacobian'] > 0.99
    assert any(abs(r['modal_velocity_m_s']) > 0.01 for r in flip.benchmark.history)
    assert abs(flip.benchmark.history[0]['modal_velocity_m_s']) < 1e-15


def test_energy_is_nearly_conserved_with_flip(flip):
    m = flip.benchmark.metrics
    assert m['energy_drift_abs_max'] < 0.12
    assert m['mass_relative_error'] < 1e-12
    assert 0.9 < m['energy_ratio_final'] < 1.1


def test_analytical_initial_state(flip):
    start = flip.benchmark.history[0]
    assert start['modal_displacement_m'] == pytest.approx(0.002, rel=1e-12)
    assert start['stress_relative_l2'] == pytest.approx(0.0, abs=1e-14)
    assert start['energy_relative_drift'] == pytest.approx(0.0, abs=1e-14)


def test_pic_has_measurable_numerical_damping(flip, pic):
    assert not pic.benchmark.passed  # intentional against undamped reference
    assert pic.benchmark.metrics['energy_ratio_final'] < 0.3
    assert flip.benchmark.metrics['energy_ratio_final'] > 0.9
    assert pic.benchmark.metrics['displacement_relative_l2_max'] > flip.benchmark.metrics['displacement_relative_l2_max']


def test_profiles_have_correct_reference_information(flip):
    assert {r['snapshot'] for r in flip.profiles} == {
        'initial', 'quarter_period', 'half_period', 'final'
    }
    assert len(flip.profiles) == 4 * flip.benchmark.metrics['particles']


def test_joint_refinement_reduces_stress_error():
    result = grid_refinement()
    assert result['passed']
    errors = [r['stress_relative_l2_max'] for r in result['rows']]
    assert errors[2] < errors[1] < errors[0]


@pytest.mark.parametrize('kwargs', [
    {'nx': 3}, {'ny': 0}, {'dt': 0}, {'dt': -1}, {'t_end': -1},
    {'dt': 0.01, 't_end': 0.065}, {'amplitude': 0.05},
    {'pic_fraction': -0.2}, {'pic_fraction': 1.2},
])
def test_rejects_invalid_cases(kwargs):
    with pytest.raises(ValueError):
        vibrating_bar(**kwargs)


def test_cli_writes_reports_without_core_modification(tmp_path):
    code = main(['--output', str(tmp_path), '--compare-pic'])
    assert code == 0
    summary = json.loads((tmp_path / '6D_summary.json').read_text(encoding='utf-8'))
    assert summary['passed']
    assert summary['primary_FLIP']['benchmark']['passed']
    assert summary['comparison_PIC']['benchmark']['passed'] is False
    assert (tmp_path / '6D_FLIP_history.csv').exists()
    assert (tmp_path / '6D_FLIP_profiles.csv').exists()
    assert (tmp_path / '6D_PIC_history.csv').exists()
