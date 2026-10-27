"""Stage 6A–6C reproducible verification regression tests, using Stage-5 API."""
import csv
import json

import numpy as np
import pytest

from validation.stage6.benchmarks import (
    affine_elastic,
    free_fall,
    free_fall_refinement,
    rigid_translation,
)
from validation.stage6.run_all import main as run_stage6


@pytest.mark.parametrize('pic_fraction', [0.0, 0.25, 1.0])
def test_rigid_translation_is_deformation_free(pic_fraction):
    case = rigid_translation(pic_fraction=pic_fraction)
    assert case.passed, case.metrics
    assert case.metrics['particles'] == 24
    assert case.metrics['crossing_particles'] > 0
    assert case.metrics['position_error'] < 1e-10
    assert case.metrics['stress_abs_max'] < 1e-6
    assert case.metrics['deformation_error'] < 1e-10
    assert case.metrics['momentum_error'] < 1e-8
    assert case.history[-1]['time'] == pytest.approx(0.5)


@pytest.mark.parametrize('dt', [0.02, 0.01, 0.005])
def test_free_fall_matches_discrete_euler_and_continuum_error(dt):
    case = free_fall(dt=dt)
    assert case.passed, case.metrics
    assert case.metrics['velocity_error'] < 1e-10
    assert case.metrics['euler_position_error'] < 1e-10
    np.testing.assert_allclose(case.metrics['continuum_position_error_end'],
                               0.5 * 9.81 * 0.2 * dt, atol=1e-10, rtol=0)
    assert case.metrics['strain_abs_max'] < 1e-10
    assert case.metrics['stress_abs_max'] < 1e-6


def test_free_fall_convergence_order_first():
    study = free_fall_refinement()
    assert study['passed']
    np.testing.assert_allclose(study['observed_orders'], [1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(study['errors'], [0.01962, 0.00981, 0.004905], atol=1e-10)


@pytest.mark.parametrize('n_steps', [1, 5, 8])
def test_affine_patch_stress_strain_deformation_and_volume(n_steps):
    case = affine_elastic(n_steps=n_steps)
    assert case.passed, case.metrics
    assert case.metrics['particles'] == 48
    assert case.metrics['stress_error'] < 1e-6
    assert case.metrics['strain_error'] < 1e-11
    assert case.metrics['position_error'] < 1e-11
    assert case.metrics['deformation_error'] < 1e-11
    assert case.metrics['volume_error'] < 1e-11
    assert case.metrics['sigma_zz_expected_pa'] == pytest.approx(-800.0 * n_steps)
    assert case.metrics['sigma_xy_expected_pa'] == pytest.approx(1600.0 * n_steps)


def test_affine_patch_reports_kinematic_scope():
    case = affine_elastic(n_steps=1)
    assert 'no traction equilibrium' in case.notes.lower()
    assert 'traction' in case.notes.lower()


def test_stage6_runner_writes_reports(tmp_path):
    assert run_stage6(['--output', str(tmp_path)]) == 0
    summary_path = tmp_path / 'summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    assert summary['passed'] is True
    assert len(summary['cases']) == 3
    assert summary['free_fall_refinement']['passed'] is True
    for name, expected_rows in [('6A', 50), ('6B', 20), ('6C', 5)]:
        csv_path = tmp_path / f'{name}_history.csv'
        with csv_path.open(newline='', encoding='utf-8') as file:
            rows = list(csv.DictReader(file))
        assert len(rows) == expected_rows


def test_free_fall_rejects_nonintegral_time_horizon():
    with pytest.raises(ValueError, match='integer multiple'):
        free_fall(dt=0.03, t_end=0.2)
